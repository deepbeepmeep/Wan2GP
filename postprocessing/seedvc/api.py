from __future__ import annotations

import os
import time

from types import SimpleNamespace
from typing import Optional, Tuple, Iterable

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

from queue import Queue
from .Models.audio import AudioData
from .inference import load_models as load_models_v1, adjust_f0_semitones, crossfade
from .inference_v2 import load_v2_models
from .inference_realtime import load_models as load_models_realtime


# Reuse the same device policy as the inference scripts
if torch.cuda.is_available():
    _device = torch.device("cuda")
elif torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")

# Global cache for V1 models and a lightweight streaming state
v1_models_cache = None  # (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args)

def get_audio_numpy(audio_segment: AudioData) -> np.ndarray:
    samples = audio_segment.samples
    arr_int16 = np.array(samples).astype("int16")
    arr_fltp = arr_int16.astype(np.float32)
    # normalization. AudioData use int16, so the max value is  `1 << 8*2 - 1`
    arr_fltp = arr_fltp / (1 << 8 * 2 - 1)
    
    return arr_fltp


class _V1StreamState:
    """Holds precomputed target features and overlap buffer for streaming V1 inference."""

    def __init__(self, args: SimpleNamespace, target: AudioData=None, new_target_name: str=None, realtime=True):
        if realtime:
            self.v1_models_cache = load_models_realtime(args)
        else:
            self.v1_models_cache = load_models_v1(args)
        (
            self.model,
            self.semantic_fn,
            self.f0_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.mel_fn,
            self.mel_fn_args,
        ) = self.v1_models_cache

        self.sr = int(self.mel_fn_args["sampling_rate"])  # 22050 or 44100
        self.hop_length = int(self.mel_fn_args["hop_size"])  # 256 or 512
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length

        self.target_name = new_target_name
        if target is not None:
            self.prepare_target(args.f0_condition, target, new_target_name)

        # Streaming overlap buffer and accumulator
        self._previous_chunk = None  # torch.Tensor on device with shape [overlap_wave_len]

    def prepare_target(self, f0_condition: bool, target: AudioData, new_target_name: str=None):
        self.target_name = new_target_name

        # Prepare target once (limit to 25s)
        target_wave = get_audio_numpy(target)
        if int(target.sample_rate) != self.sr:
            target_wave = librosa.resample(target_wave, orig_sr=int(target.sample_rate), target_sr=self.sr)
        target_wave_t = torch.tensor(target_wave, dtype=torch.float32, device=_device)[None, :]
        target_wave_t = target_wave_t[:, : self.sr * 25]

        # 16k features for target
        ori_waves_16k = torchaudio.functional.resample(target_wave_t, self.sr, 16000)
        self.S_ori = self.semantic_fn(ori_waves_16k)

        # Target mel and style
        self.mel2 = self.mel_fn(target_wave_t.float())
        self.target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        self.style2 = self.campplus_model(feat2.unsqueeze(0))

        # Optional F0 for target
        if f0_condition:
            F0_ori = self.f0_fn(ori_waves_16k[0], thred=0.03)
            self.F0_ori = torch.from_numpy(F0_ori).to(_device)[None]
        else:
            self.F0_ori = None

        # Prompt condition once
        self.prompt_condition, _, _, _, _ = self.model.length_regulator(
            self.S_ori, ylens=self.target2_lengths, n_quantizers=3, f0=self.F0_ori
        )

    def process_chunk(
        self,
        source: AudioData,
        length_adjust: float,
        diffusion_steps: int,
        inference_cfg_rate: float,
        f0_condition: bool,
        auto_f0_adjust: bool,
        semi_tone_shift: int,
        fp16_flag: bool,
        end_of_stream: bool = False,
    ) -> np.ndarray:
        # Prepare source chunk at model SR
        src_wave = get_audio_numpy(source)
        if int(source.sample_rate) != self.sr:
            src_wave = librosa.resample(src_wave, orig_sr=int(source.sample_rate), target_sr=self.sr)
        source_wave_t = torch.tensor(src_wave, dtype=torch.float32, device=_device)[None, :]

        # Content features (usually < 30s for a chunk)
        converted_waves_16k = torchaudio.functional.resample(source_wave_t, self.sr, 16000)
        S_alt = self.semantic_fn(converted_waves_16k)

        # Mel for source (to determine target length for regulator)
        mel = self.mel_fn(source_wave_t.float())
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)

        # F0 for source chunk if enabled
        if f0_condition:
            F0_alt = self.f0_fn(converted_waves_16k[0], thred=0.03)
            F0_alt = torch.from_numpy(F0_alt).to(_device)[None]
            shifted_f0_alt = F0_alt.clone()
            if auto_f0_adjust and self.F0_ori is not None:
                voiced_F0_ori = self.F0_ori[self.F0_ori > 1]
                voiced_F0_alt = F0_alt[F0_alt > 1]
                if voiced_F0_ori.numel() > 0 and voiced_F0_alt.numel() > 0:
                    log_f0_alt = torch.log(F0_alt + 1e-5)
                    median_log_f0_ori = torch.median(torch.log(voiced_F0_ori + 1e-5))
                    median_log_f0_alt = torch.median(torch.log(voiced_F0_alt + 1e-5))
                    shifted_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
                    shifted_f0_alt = torch.exp(shifted_f0_alt)
            if semi_tone_shift != 0:
                mask = F0_alt > 1
                shifted_vals = adjust_f0_semitones(shifted_f0_alt[mask], semi_tone_shift)
                shifted_f0_alt[mask] = shifted_vals
        else:
            shifted_f0_alt = None

        # Length regulation -> conditions for this chunk
        cond, _, _, _, _ = self.model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

        # VC inference for this chunk
        with torch.autocast(device_type=_device.type, dtype=torch.float16 if fp16_flag else torch.float32):
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
        vc_wave = self.vocoder_fn(vc_target.float()).squeeze()[None]

        # Streaming crossfade logic
        if self._previous_chunk is None:
            if end_of_stream:
                # First and last chunk: return all
                output_wave = vc_wave[0].detach().cpu().numpy()
                return output_wave
            # Hold back overlap for future crossfade
            head = vc_wave[0, :-self.overlap_wave_len].detach().cpu().numpy()
            self._previous_chunk = vc_wave[0, -self.overlap_wave_len:]
            return head
        else:
            if end_of_stream:
                # Crossfade previous tail with entire current chunk
                output_wave = crossfade(
                    self._previous_chunk.detach().cpu().numpy(),
                    vc_wave[0].detach().cpu().numpy(),
                    self.overlap_wave_len,
                )
                # Reset state for next session
                self._previous_chunk = None
                return output_wave
            # Middle chunk: crossfade prev tail with current head excluding new tail
            head = vc_wave[0, :-self.overlap_wave_len]
            output_wave = crossfade(
                self._previous_chunk.detach().cpu().numpy(),
                head.detach().cpu().numpy(),
                self.overlap_wave_len,
            )
            # Update tail buffer
            self._previous_chunk = vc_wave[0, -self.overlap_wave_len:]
            return output_wave


@torch.no_grad()
def inference(
    source: AudioData,
    target: AudioData,
    new_target_name: Optional[str] = None,
    output: Optional[str] = None,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = False,
    auto_f0_adjust: bool = False,
    semi_tone_shift: int = 0,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    fp16: bool = True,
    # New optional streaming parameters
    streaming: bool = False,
    stream_state: Optional[_V1StreamState] = None,
    end_of_stream: bool = False,
    realtime: bool = True
) -> AudioData:
    """
    Run Seed-VC V1 inference.

    Default: non-streaming full-clip conversion (original behavior).
    Streaming mode: models are loaded once; each call treats `source` as a chunk and
    returns the streamable audio segment. Maintain `stream_state` across calls.

    Returns: (sample_rate, waveform_np)
    Optionally writes a file if `output` directory is provided (non-streaming mode).
    """
    # Build an args-like namespace for loader
    args = SimpleNamespace(
        f0_condition=f0_condition,
        checkpoint=checkpoint,
        config=config,
        fp16=fp16,
    )

    if streaming:
        # Initialize stream state on first chunk
        if stream_state is None:
            stream_state = _V1StreamState(args, target, new_target_name, realtime)
        elif(new_target_name != stream_state.target_name):
            stream_state.prepare_target(f0_condition, target, new_target_name)
        sr = stream_state.sr
        chunk_audio = stream_state.process_chunk(
            source=source,
            length_adjust=length_adjust,
            diffusion_steps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            semi_tone_shift=semi_tone_shift,
            fp16_flag=fp16,
            end_of_stream=end_of_stream,
        )

        if source.sample_rate != sr:
            chunk_audio = librosa.resample(chunk_audio, orig_sr=sr, target_sr=source.sample_rate)

        arr_fltp = chunk_audio * (1 << 8 * 2 - 1)
        arr_int16 = arr_fltp.astype("int16")

        output_audio = AudioData (
            arr_int16,
            source.mel_chunks,
            source.duration,
            source.samples_count,
            source.sample_rate,
            source.metadata,
        )
        return output_audio        

    # ---- Original non-streaming path below ----
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models_realtime(args)
    sr = int(mel_fn_args["sampling_rate"])  # 22050 or 44100 depending on f0_condition

    # Prepare source/target audio at model SR
    def _to_tensor_at_sr(wave: np.ndarray, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr != target_sr:
            wave = librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr)
        wave_t = torch.tensor(wave, dtype=torch.float32, device=_device)[None, :]
        return wave_t

    # Limit target to 25s like CLI (context len - safety)
    source_wave_t = _to_tensor_at_sr(get_audio_numpy(source), int(source.sample_rate), sr)
    target_wave_t = _to_tensor_at_sr(get_audio_numpy(target), int(target.sample_rate), sr)
    target_wave_t = target_wave_t[:, : sr * 25]

    # Resample to 16k for content (Whisper/xlsr)
    converted_waves_16k = torchaudio.functional.resample(source_wave_t, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(target_wave_t, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    # Mels
    mel = mel_fn(source_wave_t.float())
    mel2 = mel_fn(target_wave_t.float())

    hop_length = int(mel_fn_args["hop_size"])  # 256 or 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # Style vector via CAMPPlus on 16k fbank
    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # F0
    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(_device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(_device)[None]
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if semi_tone_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], semi_tone_shift)
    else:
        F0_ori = None
        shifted_f0_alt = None

    # Length regulation -> conditions
    cond, _, _, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )

    # Chunked generation with crossfade
    processed_frames = 0
    generated_wave_chunks = []
    start_time = time.time()
    while processed_frames < cond.size(1):
        max_source_window = max_context_window - mel2.size(2)
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=_device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave_np = np.concatenate(generated_wave_chunks)
    elapsed = time.time() - start_time
    if vc_wave_np.size > 0:
        print(f"RTF: {elapsed / vc_wave_np.size * sr}")

    # Optionally save
    if output:
        os.makedirs(output, exist_ok=True)
        src_name = "source"
        tgt_name = "target"
        out_path = os.path.join(
            output,
            f"vc_{src_name}_{tgt_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav",
        )
        sf.write(out_path, vc_wave_np, sr)

    if source.sample_rate != sr:
        vc_wave_np = librosa.resample(vc_wave_np, orig_sr=sr, target_sr=source.sample_rate)

    arr_fltp = vc_wave_np * (1 << 8 * 2 - 1)
    arr_int16 = arr_fltp.astype("int16")

    output_audio = AudioData (
        arr_int16,
        source.mel_chunks,
        source.duration,
        source.samples_count,
        source.sample_rate,
        source.metadata,
    )
    return output_audio      


@torch.no_grad()
def inference_v2(
    source: AudioData,
    target: AudioData,
    output: Optional[str] = None,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    intelligibility_cfg_rate: float = 0.7,
    similarity_cfg_rate: float = 0.7,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    convert_style: bool = False,
    anonymization_only: bool = False,
    compile: bool = False,
    ar_checkpoint_path: Optional[str] = None,
    cfm_checkpoint_path: Optional[str] = None,
) -> Tuple[int, np.ndarray]:
    """
    Run Seed-VC V2 inference given in-memory audio (uses the v2 wrapper under the hood).

    Returns: (sample_rate, waveform_np)
    Optionally writes a file if `output` directory is provided.
    """
    # Build args for v2 loader and conversion call
    args = SimpleNamespace(
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligibility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
        compile=compile,
        ar_checkpoint_path=ar_checkpoint_path,
        cfm_checkpoint_path=cfm_checkpoint_path,
    )

    # Ensure models are loaded
    from . import inference_v2 as _infv2
    if _infv2.vc_wrapper_v2 is None:
        _infv2.vc_wrapper_v2 = load_v2_models(args)

    # Call the in-memory V2 wrapper directly
    sr_v2, audio_np = _infv2.vc_wrapper_v2.convert_voice_with_streaming_arrays(
        source_wave=get_audio_numpy(source),
        target_wave=get_audio_numpy(target),
        source_sr=int(source.sample_rate),
        target_sr=int(target.sample_rate),
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligebility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
        device=_device,
        dtype=torch.float16,
        stream_output=False,
    )

    # Optionally save
    if output:
        os.makedirs(output, exist_ok=True)
        src_name = "source"
        tgt_name = "target"
        out_path = os.path.join(
            output,
            f"vc_v2_{src_name}_{tgt_name}_{length_adjust}_{diffusion_steps}_{similarity_cfg_rate}.wav",
        )
        sf.write(out_path, audio_np, sr_v2)

    return sr_v2, audio_np


# ---------------- Convenience helpers for V1 streaming ----------------

def create_v1_stream_state(
    target: AudioData,
    new_target_name: Optional[str] = None,
    f0_condition: bool = False,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    fp16: bool = True,
    realtime: bool = True
) -> _V1StreamState:
    """Create and return a reusable V1 streaming state.

    Preloads models (once) and precomputes target conditioning.
    Keep the returned state and reuse it across chunk calls.
    """
    args = SimpleNamespace(
        f0_condition=f0_condition,
        checkpoint=checkpoint,
        config=config,
        fp16=fp16,
    )
    return _V1StreamState(args, target, new_target_name, realtime)


def inference_v1_streaming(
    source_chunks: Queue[AudioData],
    target: AudioData,
    new_target_name: Optional[str] = None,
    output: Optional[str] = None,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = False,
    auto_f0_adjust: bool = False,
    semi_tone_shift: int = 0,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    fp16: bool = True,
    yield_full_audio: bool = False,
    stream_state: Optional[_V1StreamState] = None,
    realtime: bool = True
):
    """
    Generator wrapper for V1 streaming, similar in spirit to V2's streaming API.

    Yields tuples per chunk: (sample_rate, chunk_audio_np, full_audio_np_or_None)
    - chunk_audio_np is the streamable segment for this input chunk
    - full_audio_np_or_None is the concatenated audio-so-far if yield_full_audio=True, else None

    Notes:
    - `target` is used to precompute prompt/style once and reused for all chunks.
    - `source_chunks` should yield AudioData chunks in order.
    - The last yielded item includes the crossfaded tail (set internally via end_of_stream).
    - Optionally writes the final full audio if `output` is provided and yield_full_audio=True.
    """
    # Initialize stream state on first chunk
    if stream_state is None:
        stream_state = create_v1_stream_state(
            target=target,
            new_target_name=new_target_name,
            f0_condition=f0_condition,
            checkpoint=checkpoint,
            config=config,
            fp16=fp16,
            realtime=realtime
        )
    elif(new_target_name != stream_state.target_name):
        stream_state.prepare_target(f0_condition, target, new_target_name)

    prev = None
    # Iterate with lookahead to know when we're at the last chunk
    if source_chunks.empty():
        return  # empty iterator

    full_chunks = []
    prev = source_chunks.get()

    while not source_chunks.empty():
        cur = source_chunks.get()
        chunk_audio = inference(
            source=prev,
            target=target,
            new_target_name=new_target_name,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            semi_tone_shift=semi_tone_shift,
            checkpoint=checkpoint,
            config=config,
            fp16=fp16,
            streaming=True,
            stream_state=stream_state,
            end_of_stream=False,
            realtime=realtime
        )
        full_chunks.append(chunk_audio.samples)
        if yield_full_audio:
            yield chunk_audio, np.concatenate(full_chunks) if len(full_chunks) > 0 else np.array([], dtype=np.float32)
        else:
            yield chunk_audio, None
        prev = cur

    # Handle last chunk
    last_audio = inference(
        source=prev,
        target=target,
        new_target_name=new_target_name,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        semi_tone_shift=semi_tone_shift,
        checkpoint=checkpoint,
        config=config,
        fp16=fp16,
        streaming=True,
        stream_state=stream_state,
        end_of_stream=True,
        realtime=realtime
    )
    full_chunks.append(last_audio.samples)

    full_audio = np.concatenate(full_chunks) if len(full_chunks) > 0 else np.array([], dtype=np.float32)

    if yield_full_audio:
        # Optionally save final output
        if output:
            os.makedirs(output, exist_ok=True)
            src_name = "source"
            tgt_name = "target"
            out_path = os.path.join(
                output,
                f"vc_v1_stream_{src_name}_{tgt_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav",
            )
            sf.write(out_path, full_audio, last_audio.sample_rate)

        yield last_audio, full_audio
    else:
        yield last_audio, None

