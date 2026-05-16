import torch
import librosa
import torchaudio
import numpy as np
from pydub import AudioSegment
from ...hf_utils import load_custom_model_from_hf

DEFAULT_REPO_ID = "Plachta/Seed-VC"
DEFAULT_CFM_CHECKPOINT = "v2/cfm_small.pth"
DEFAULT_AR_CHECKPOINT = "v2/ar_base.pth"

DEFAULT_CE_REPO_ID = "Plachta/ASTRAL-quantization"
DEFAULT_CE_NARROW_CHECKPOINT = "bsq32/bsq32_light.pth"
DEFAULT_CE_WIDE_CHECKPOINT = "bsq2048/bsq2048_light.pth"

DEFAULT_SE_REPO_ID = "funasr/campplus"
DEFAULT_SE_CHECKPOINT = "campplus_cn_common.bin"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class VoiceConversionWrapper(torch.nn.Module):
    def __init__(
            self,
            sr: int,
            hop_size: int,
            mel_fn: callable,
            cfm: torch.nn.Module,
            cfm_length_regulator: torch.nn.Module,
            content_extractor_narrow: torch.nn.Module,
            content_extractor_wide: torch.nn.Module,
            ar_length_regulator: torch.nn.Module,
            ar: torch.nn.Module,
            style_encoder: torch.nn.Module,
            vocoder: torch.nn.Module,
            ):
        super(VoiceConversionWrapper, self).__init__()
        self.sr = sr
        self.hop_size = hop_size
        self.mel_fn = mel_fn
        self.cfm = cfm
        self.cfm_length_regulator = cfm_length_regulator
        self.content_extractor_narrow = content_extractor_narrow
        self.content_extractor_wide = content_extractor_wide
        self.vocoder = vocoder
        self.ar_length_regulator = ar_length_regulator
        self.ar = ar
        self.style_encoder = style_encoder
        # Set streaming parameters
        self.overlap_frame_len = 16
        self.bitrate = "320k"
        self.compiled_decode_fn = None
        self.dit_compiled = False
        self.dit_max_context_len = 30  # in seconds
        self.ar_max_content_len = 1500  # in num of narrow tokens
        self.compile_len = 87 * self.dit_max_context_len

    def forward_cfm(self, content_indices_wide, content_lens, mels, mel_lens, style_vectors):
        device = content_indices_wide.device
        B = content_indices_wide.size(0)
        cond, _ = self.cfm_length_regulator(content_indices_wide, ylens=mel_lens)

        # randomly set a length as prompt
        prompt_len_max = mel_lens - 1
        prompt_len = (torch.rand([B], device=device) * prompt_len_max).floor().to(dtype=torch.long)
        prompt_len[torch.rand([B], device=device) < 0.1] = 0

        loss = self.cfm(mels, mel_lens, prompt_len, cond, style_vectors)
        return loss

    def forward_ar(self, content_indices_narrow, content_indices_wide, content_lens):
        device = content_indices_narrow.device
        duration_reduced_narrow_tokens = []
        duration_reduced_narrow_lens = []
        for bib in range(content_indices_narrow.size(0)):
            reduced, reduced_len = self.duration_reduction_func(content_indices_narrow[bib])
            duration_reduced_narrow_tokens.append(reduced)
            duration_reduced_narrow_lens.append(reduced_len)
        duration_reduced_narrow_tokens = torch.nn.utils.rnn.pad_sequence(duration_reduced_narrow_tokens,
            batch_first=True, padding_value=0).to(device)
        duration_reduced_narrow_lens = torch.LongTensor(duration_reduced_narrow_lens).to(device)

        # interpolate speech token to match acoustic feature length
        cond, _ = self.ar_length_regulator(duration_reduced_narrow_tokens)
        loss = self.ar(cond, duration_reduced_narrow_lens, content_indices_wide, content_lens)
        return loss

    def forward(self, waves_16k, mels, wave_lens_16k, mel_lens, forward_ar=False, forward_cfm=True):
        """
        Forward pass for the model.
        """
        # extract wide content features as both AR and CFM models use them
        with torch.no_grad():
            _, content_indices_wide, content_lens = self.content_extractor_wide(waves_16k, wave_lens_16k)
        if forward_ar:
            # extract narrow content features for AR model
            _, content_indices_narrow, _ = self.content_extractor_narrow(waves_16k, wave_lens_16k, ssl_model=self.content_extractor_wide.ssl_model)
            loss_ar = self.forward_ar(content_indices_narrow.clone(), content_indices_wide.clone(), content_lens)
        else:
            loss_ar = torch.tensor(0.0, device=waves_16k.device, dtype=waves_16k.dtype)
        if forward_cfm:
            style_vectors = self.compute_style(waves_16k, wave_lens_16k)
            loss_cfm = self.forward_cfm(content_indices_wide, content_lens, mels, mel_lens, style_vectors)
        else:
            loss_cfm = torch.tensor(0.0, device=waves_16k.device, dtype=waves_16k.dtype)
        return loss_ar, loss_cfm

    def compile_ar(self):
        """
        Compile the AR model for inference.
        """
        self.compiled_decode_fn = torch.compile(
            self.ar.model.forward_generate,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    def compile_cfm(self):
        self.cfm.estimator.transformer = torch.compile(
            self.cfm.estimator.transformer,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )
        self.dit_compiled = True

    @staticmethod
    def strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
        """
        Strip the prefix from the state_dict keys.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    @staticmethod
    def duration_reduction_func(token_seq, n_gram=1):
        """
        Args:
            token_seq: (T,)
        Returns:
            reduced_token_seq: (T')
            reduced_token_seq_len: T'
        """
        n_gram_seq = token_seq.unfold(0, n_gram, 1)
        mask = torch.all(n_gram_seq[1:] != n_gram_seq[:-1], dim=1)
        reduced_token_seq = torch.cat(
            (n_gram_seq[0, :n_gram], n_gram_seq[1:, -1][mask])
        )
        return reduced_token_seq, len(reduced_token_seq)

    @staticmethod
    def crossfade(chunk1, chunk2, overlap):
        """Apply crossfade between two audio chunks."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def _stream_wave_chunks(self, vc_wave, processed_frames, vc_mel, overlap_wave_len,
                           generated_wave_chunks, previous_chunk, is_last_chunk, stream_output):
        """
        Helper method to handle streaming wave chunks.

        Args:
            vc_wave: The current wave chunk
            processed_frames: Number of frames processed so far
            vc_mel: The mel spectrogram
            overlap_wave_len: Length of overlap between chunks
            generated_wave_chunks: List of generated wave chunks
            previous_chunk: Previous wave chunk for crossfading
            is_last_chunk: Whether this is the last chunk
            stream_output: Whether to stream the output

        Returns:
            Tuple of (processed_frames, previous_chunk, should_break, mp3_bytes, full_audio)
            where should_break indicates if processing should stop
            mp3_bytes is the MP3 bytes if streaming, None otherwise
            full_audio is the full audio if this is the last chunk, None otherwise
        """
        mp3_bytes = None
        full_audio = None

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)

                if stream_output:
                    output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = AudioSegment(
                        output_wave_int16.tobytes(), frame_rate=self.sr,
                        sample_width=output_wave_int16.dtype.itemsize, channels=1
                    ).export(format="mp3", bitrate=self.bitrate).read()
                    full_audio = (self.sr, np.concatenate(generated_wave_chunks))
                else:
                    return processed_frames, previous_chunk, True, None, np.concatenate(generated_wave_chunks)

                return processed_frames, previous_chunk, True, mp3_bytes, full_audio

            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_mel.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()

        elif is_last_chunk:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_mel.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()
                full_audio = (self.sr, np.concatenate(generated_wave_chunks))
            else:
                return processed_frames, previous_chunk, True, None, np.concatenate(generated_wave_chunks)

            return processed_frames, previous_chunk, True, mp3_bytes, full_audio

        else:
            output_wave = self.crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_mel.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=self.bitrate).read()

        return processed_frames, previous_chunk, False, mp3_bytes, full_audio

    def load_checkpoints(
            self,
            cfm_checkpoint_path = None,
            ar_checkpoint_path = None,
    ):
        if cfm_checkpoint_path is None:
            cfm_checkpoint_path = load_custom_model_from_hf(
                repo_id=DEFAULT_REPO_ID,
                model_filename=DEFAULT_CFM_CHECKPOINT,
            )
        else:
            print(f"Loading CFM checkpoint from {cfm_checkpoint_path}...")
        if ar_checkpoint_path is None:
            ar_checkpoint_path = load_custom_model_from_hf(
                repo_id=DEFAULT_REPO_ID,
                model_filename=DEFAULT_AR_CHECKPOINT,
            )
        else:
            print(f"Loading AR checkpoint from {ar_checkpoint_path}...")
        # cfm
        cfm_checkpoint = torch.load(cfm_checkpoint_path, map_location="cpu")
        cfm_length_regulator_state_dict = self.strip_prefix(cfm_checkpoint["net"]['length_regulator'], "module.")
        cfm_state_dict = self.strip_prefix(cfm_checkpoint["net"]['cfm'], "module.")
        missing_keys, unexpected_keys = self.cfm.load_state_dict(cfm_state_dict, strict=False)
        missing_keys, unexpected_keys = self.cfm_length_regulator.load_state_dict(cfm_length_regulator_state_dict, strict=False)

        # ar
        ar_checkpoint = torch.load(ar_checkpoint_path, map_location="cpu")
        ar_length_regulator_state_dict = self.strip_prefix(ar_checkpoint["net"]['length_regulator'], "module.")
        ar_state_dict = self.strip_prefix(ar_checkpoint["net"]['ar'], "module.")
        missing_keys, unexpected_keys = self.ar.load_state_dict(ar_state_dict, strict=False)
        missing_keys, unexpected_keys = self.ar_length_regulator.load_state_dict(ar_length_regulator_state_dict, strict=False)

        # content extractor
        content_extractor_narrow_checkpoint_path = load_custom_model_from_hf(
            repo_id=DEFAULT_CE_REPO_ID,
            model_filename=DEFAULT_CE_NARROW_CHECKPOINT,
        )
        content_extractor_narrow_checkpoint = torch.load(content_extractor_narrow_checkpoint_path, map_location="cpu")
        self.content_extractor_narrow.load_state_dict(
            content_extractor_narrow_checkpoint, strict=False
        )

        content_extractor_wide_checkpoint_path = load_custom_model_from_hf(
            repo_id=DEFAULT_CE_REPO_ID,
            model_filename=DEFAULT_CE_WIDE_CHECKPOINT,
        )
        content_extractor_wide_checkpoint = torch.load(content_extractor_wide_checkpoint_path, map_location="cpu")
        self.content_extractor_wide.load_state_dict(
            content_extractor_wide_checkpoint, strict=False
        )

        # style encoder
        style_encoder_checkpoint_path = load_custom_model_from_hf(DEFAULT_SE_REPO_ID, DEFAULT_SE_CHECKPOINT, config_filename=None)
        style_encoder_checkpoint = torch.load(style_encoder_checkpoint_path, map_location="cpu")
        self.style_encoder.load_state_dict(style_encoder_checkpoint, strict=False)

    def setup_ar_caches(self, max_batch_size=1, max_seq_len=4096, dtype=torch.float32, device=torch.device("cpu")):
        self.ar.setup_caches(max_batch_size=max_batch_size, max_seq_len=max_seq_len, dtype=dtype, device=device)

    @torch.no_grad()
    def compute_style(self, waves_16k: torch.Tensor, wave_lens_16k: torch.Tensor = None):
        if wave_lens_16k is None:
            wave_lens_16k = torch.tensor([waves_16k.size(-1)], dtype=torch.int32).to(waves_16k.device)
        feat_list = []
        for bib in range(waves_16k.size(0)):
            feat = torchaudio.compliance.kaldi.fbank(waves_16k[bib:bib + 1, :wave_lens_16k[bib]],
                               num_mel_bins=80,
                               dither=0,
                               sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        max_feat_len = max([feat.size(0) for feat in feat_list])
        feat_lens = torch.tensor([feat.size(0) for feat in feat_list], dtype=torch.int32).to(waves_16k.device) // 2
        feat_list = [
            torch.nn.functional.pad(feat, (0, 0, 0, max_feat_len - feat.size(0)), value=float(feat.min().item()))
            for feat in feat_list
        ]
        feat = torch.stack(feat_list, dim=0)
        style = self.style_encoder(feat, feat_lens)
        return style

    @torch.no_grad()
    @torch.inference_mode()
    def convert_timbre(
            self,
            source_audio_path: str,
            target_audio_path: str,
            diffusion_steps: int = 30,
            length_adjust: float = 1.0,
            inference_cfg_rate: float = 0.5,
            use_sway_sampling: bool = False,
            use_amo_sampling: bool = False,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ):
        source_wave = librosa.load(source_audio_path, sr=self.sr)[0]
        target_wave = librosa.load(target_audio_path, sr=self.sr)[0]
        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).to(device)
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).to(device)

        # get 16khz audio
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.sr, target_sr=16000)
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.sr, target_sr=16000)
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

        # compute mel spectrogram
        source_mel = self.mel_fn(source_wave_tensor)
        target_mel = self.mel_fn(target_wave_tensor)
        source_mel_len = source_mel.size(2)
        target_mel_len = target_mel.size(2)

        with torch.autocast(device_type=device.type, dtype=dtype):
            # compute content features
            _, source_content_indices, _ = self.content_extractor_wide(source_wave_16k_tensor, [source_wave_16k.size])
            _, target_content_indices, _ = self.content_extractor_wide(target_wave_16k_tensor, [target_wave_16k.size])

            # compute style features
            target_style = self.compute_style(target_wave_16k_tensor)

            # Length regulation
            cond, _ = self.cfm_length_regulator(source_content_indices, ylens=torch.LongTensor([source_mel_len]).to(device))
            prompt_condition, _, = self.cfm_length_regulator(target_content_indices, ylens=torch.LongTensor([target_mel_len]).to(device))

            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            # generate mel spectrogram
            vc_mel = self.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                target_mel, target_style, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                sway_sampling=use_sway_sampling,
                amo_sampling=use_amo_sampling,
            )
        vc_mel = vc_mel[:, :, target_mel_len:]
        vc_wave = self.vocoder(vc_mel.float()).squeeze()[None]
        return vc_wave.cpu().numpy()

    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice(
            self,
            source_audio_path: str,
            target_audio_path: str,
            diffusion_steps: int = 30,
            length_adjust: float = 1.0,
            inference_cfg_rate: float = 0.5,
            top_p: float = 0.7,
            temperature: float = 0.7,
            repetition_penalty: float = 1.5,
            use_sway_sampling: bool = False,
            use_amo_sampling: bool = False,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ):
        source_wave = librosa.load(source_audio_path, sr=self.sr)[0]
        target_wave = librosa.load(target_audio_path, sr=self.sr)[0]
        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).to(device)
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).to(device)

        # get 16khz audio
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.sr, target_sr=16000)
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.sr, target_sr=16000)
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

        # compute mel spectrogram
        source_mel = self.mel_fn(source_wave_tensor)
        target_mel = self.mel_fn(target_wave_tensor)
        source_mel_len = source_mel.size(2)
        target_mel_len = target_mel.size(2)

        with torch.autocast(device_type=device.type, dtype=dtype):
            # compute content features
            _, source_content_indices, _ = self.content_extractor_wide(source_wave_16k_tensor, [source_wave_16k.size])
            _, target_content_indices, _ = self.content_extractor_wide(target_wave_16k_tensor, [target_wave_16k.size])

            _, source_narrow_indices, _ = self.content_extractor_narrow(source_wave_16k_tensor,
                                                                         [source_wave_16k.size], ssl_model=self.content_extractor_wide.ssl_model)
            _, target_narrow_indices, _ = self.content_extractor_narrow(target_wave_16k_tensor,
                                                                         [target_wave_16k.size], ssl_model=self.content_extractor_wide.ssl_model)

            src_narrow_reduced, src_narrow_len = self.duration_reduction_func(source_narrow_indices[0], 1)
            tgt_narrow_reduced, tgt_narrow_len = self.duration_reduction_func(target_narrow_indices[0], 1)

            ar_cond = self.ar_length_regulator(torch.cat([tgt_narrow_reduced, src_narrow_reduced], dim=0)[None])[0]

            ar_out = self.ar.generate(ar_cond, target_content_indices, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)
            ar_out_mel_len = torch.LongTensor([int(source_mel_len / source_content_indices.size(-1) * ar_out.size(-1) * length_adjust)]).to(device)
            # compute style features
            target_style = self.compute_style(target_wave_16k_tensor)

            # Length regulation
            cond, _ = self.cfm_length_regulator(ar_out, ylens=torch.LongTensor([ar_out_mel_len]).to(device))
            prompt_condition, _, = self.cfm_length_regulator(target_content_indices, ylens=torch.LongTensor([target_mel_len]).to(device))

            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            # generate mel spectrogram
            vc_mel = self.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                target_mel, target_style, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                sway_sampling=use_sway_sampling,
                amo_sampling=use_amo_sampling,
            )
        vc_mel = vc_mel[:, :, target_mel_len:]
        vc_wave = self.vocoder(vc_mel.float()).squeeze()[None]
        return vc_wave.cpu().numpy()

    def _process_content_features(self, audio_16k_tensor, is_narrow=False):
        """Process audio through Whisper model to extract features."""
        content_extractor_fn = self.content_extractor_narrow if is_narrow else self.content_extractor_wide
        if audio_16k_tensor.size(-1) <= 16000 * 30:
            # Compute content features
            _, content_indices, _ = content_extractor_fn(audio_16k_tensor, [audio_16k_tensor.size(-1)], ssl_model=self.content_extractor_wide.ssl_model)
        else:
            # Process long audio in chunks
            overlapping_time = 5  # 5 seconds
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k_tensor.size(-1):
                if buffer is None:  # first chunk
                    chunk = audio_16k_tensor[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer,
                        audio_16k_tensor[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                _, chunk_content_indices, _ = content_extractor_fn(chunk, [chunk.size(-1)], ssl_model=self.content_extractor_wide.ssl_model)
                if traversed_time == 0:
                    features_list.append(chunk_content_indices)
                else:
                    features_list.append(chunk_content_indices[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            content_indices = torch.cat(features_list, dim=1)

        return content_indices

    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice_with_streaming(
            self,
            source_audio_path: str,
            target_audio_path: str,
            diffusion_steps: int = 30,
            length_adjust: float = 1.0,
            intelligebility_cfg_rate: float = 0.7,
            similarity_cfg_rate: float = 0.7,
            top_p: float = 0.7,
            temperature: float = 0.7,
            repetition_penalty: float = 1.5,
            convert_style: bool = False,
            anonymization_only: bool = False,
            device: torch.device = device,
            dtype: torch.dtype = torch.float16,
            stream_output: bool = True,
    ):
        """
        Convert voice with streaming support for long audio files.

        Args:
            source_audio_path: Path to source audio file
            target_audio_path: Path to target audio file
            diffusion_steps: Number of diffusion steps (default: 30)
            length_adjust: Length adjustment factor (default: 1.0)
            intelligebility_cfg_rate: CFG rate for intelligibility (default: 0.7)
            similarity_cfg_rate: CFG rate for similarity (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.7)
            temperature: Temperature for sampling (default: 0.7)
            repetition_penalty: Repetition penalty (default: 1.5)
            device: Device to use (default: cpu)
            dtype: Data type to use (default: float32)
            stream_output: Whether to stream the output (default: True)

        Returns:
            If stream_output is True, yields (mp3_bytes, full_audio) tuples
            If stream_output is False, returns the full audio as a numpy array
        """
        # Load audio
        source_wave = librosa.load(source_audio_path, sr=self.sr)[0]
        target_wave = librosa.load(target_audio_path, sr=self.sr)[0]

        # Limit target audio to 25 seconds
        target_wave = target_wave[:self.sr * (self.dit_max_context_len - 5)]

        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(device)
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(device)

        # Resample to 16kHz for feature extraction
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.sr, target_sr=16000)
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.sr, target_sr=16000)
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

        # Compute mel spectrograms
        source_mel = self.mel_fn(source_wave_tensor)
        target_mel = self.mel_fn(target_wave_tensor)
        source_mel_len = source_mel.size(2)
        target_mel_len = target_mel.size(2)

        # Set up chunk processing parameters
        max_context_window = self.sr // self.hop_size * self.dit_max_context_len
        overlap_wave_len = self.overlap_frame_len * self.hop_size

        with torch.autocast(device_type=device.type, dtype=dtype):
            # Compute content features
            source_content_indices = self._process_content_features(source_wave_16k_tensor, is_narrow=False)
            target_content_indices = self._process_content_features(target_wave_16k_tensor, is_narrow=False)
            # Compute style features
            target_style = self.compute_style(target_wave_16k_tensor)
            prompt_condition, _, = self.cfm_length_regulator(
                target_content_indices,
                ylens=torch.LongTensor([target_mel_len]).to(device),
            )

        # prepare for streaming
        generated_wave_chunks = []
        processed_frames = 0
        previous_chunk = None
        if convert_style:
            with torch.autocast(device_type=device.type, dtype=dtype):
                source_narrow_indices = self._process_content_features(source_wave_16k_tensor, is_narrow=True)
                target_narrow_indices = self._process_content_features(target_wave_16k_tensor, is_narrow=True)
            src_narrow_reduced, src_narrow_len = self.duration_reduction_func(source_narrow_indices[0], 1)
            tgt_narrow_reduced, tgt_narrow_len = self.duration_reduction_func(target_narrow_indices[0], 1)
            # Process src_narrow_reduced in chunks of max 1000 tokens
            max_chunk_size = self.ar_max_content_len - tgt_narrow_len

            # Process src_narrow_reduced in chunks
            for i in range(0, len(src_narrow_reduced), max_chunk_size):
                is_last_chunk = i + max_chunk_size >= len(src_narrow_reduced)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    chunk = src_narrow_reduced[i:i + max_chunk_size]
                    if anonymization_only:
                        chunk_ar_cond = self.ar_length_regulator(chunk[None])[0]
                        chunk_ar_out = self.ar.generate(
                            chunk_ar_cond,
                            torch.zeros([1, 0]).long().to(device),
                            compiled_decode_fn=self.compiled_decode_fn,
                            top_p=top_p, temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    else:
                        # For each chunk, we need to include tgt_narrow_reduced as context
                        chunk_ar_cond = self.ar_length_regulator(torch.cat([tgt_narrow_reduced, chunk], dim=0)[None])[0]
                        chunk_ar_out = self.ar.generate(
                            chunk_ar_cond,
                            target_content_indices,
                            compiled_decode_fn=self.compiled_decode_fn,
                            top_p=top_p, temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    chunkar_out_mel_len = torch.LongTensor([
                        int(source_mel_len / source_content_indices.size(-1) * chunk_ar_out.size(-1) * length_adjust)
                    ]).to(device)
                    # Length regulation
                    chunk_cond, _ = self.cfm_length_regulator(chunk_ar_out, ylens=torch.LongTensor([chunkar_out_mel_len]).to(device))
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    original_len = cat_condition.size(1)
                    # pad cat_condition to compile_len
                    if self.dit_compiled:
                        cat_condition = torch.nn.functional.pad(cat_condition,
                                                                (0, 0, 0, self.compile_len - cat_condition.size(1),),
                                                                value=0)
                    # Voice Conversion
                    vc_mel = self.cfm.inference(
                        cat_condition,
                        torch.LongTensor([original_len]).to(device),
                        target_mel, target_style, diffusion_steps,
                        inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                        random_voice=anonymization_only,
                    )
                    vc_mel = vc_mel[:, :, target_mel_len:original_len]
                vc_wave = self.vocoder(vc_mel).squeeze()[None]
                processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = self._stream_wave_chunks(
                    vc_wave, processed_frames, vc_mel, overlap_wave_len,
                    generated_wave_chunks, previous_chunk, is_last_chunk, stream_output
                )

                if stream_output and mp3_bytes is not None:
                    yield mp3_bytes, full_audio
                if should_break:
                    break
        else:
            cond, _ = self.cfm_length_regulator(source_content_indices, ylens=torch.LongTensor([source_mel_len]).to(device))

            # Process in chunks for streaming
            max_source_window = max_context_window - target_mel.size(2)

            # Generate chunk by chunk and stream the output
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                original_len = cat_condition.size(1)
                # pad cat_condition to compile_len
                if self.dit_compiled:
                    cat_condition = torch.nn.functional.pad(cat_condition,
                                                            (0, 0, 0, self.compile_len - cat_condition.size(1),), value=0)
                with torch.autocast(device_type=device.type, dtype=torch.float32):  # force CFM to use float32
                    # Voice Conversion
                    vc_mel = self.cfm.inference(
                        cat_condition,
                        torch.LongTensor([original_len]).to(device),
                        target_mel, target_style, diffusion_steps,
                        inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                        random_voice=anonymization_only,
                    )
                vc_mel = vc_mel[:, :, target_mel_len:original_len]
                vc_wave = self.vocoder(vc_mel).squeeze()[None]

                processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = self._stream_wave_chunks(
                    vc_wave, processed_frames, vc_mel, overlap_wave_len,
                    generated_wave_chunks, previous_chunk, is_last_chunk, stream_output
                )

                if stream_output and mp3_bytes is not None:
                    yield mp3_bytes, full_audio
                if should_break:
                    break

        # If not streaming, return the concatenated audio
        if not stream_output:
            return self.sr, np.concatenate(generated_wave_chunks) if len(generated_wave_chunks) > 0 else np.array([], dtype=np.float32)


    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice_with_streaming_arrays(
        self,
        source_wave: np.ndarray,
        target_wave: np.ndarray,
        source_sr: int,
        target_sr: int,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        intelligebility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        convert_style: bool = False,
        anonymization_only: bool = False,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        stream_output: bool = False,
    ):
        """
        In-memory variant of convert_voice_with_streaming that accepts raw arrays and sample rates.
        If stream_output is False, returns (sr, full_audio_np). If True, yields (mp3_bytes, full_audio_np) chunks.
        """
        # Normalize input arrays
        source_wave = np.asarray(source_wave, dtype=np.float32).reshape(-1)
        target_wave = np.asarray(target_wave, dtype=np.float32).reshape(-1)

        # Resample to model SR if needed
        if source_sr != self.sr:
            source_wave = librosa.resample(source_wave, orig_sr=source_sr, target_sr=self.sr)
        if target_sr != self.sr:
            target_wave = librosa.resample(target_wave, orig_sr=target_sr, target_sr=self.sr)

        # Limit target audio duration to protect context window
        target_wave = target_wave[: self.sr * (self.dit_max_context_len - 5)]

        # Torch tensors
        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(device)
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(device)

        # 16k content
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.sr, target_sr=16000)
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.sr, target_sr=16000)
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

        # Mels
        source_mel = self.mel_fn(source_wave_tensor)
        target_mel = self.mel_fn(target_wave_tensor)
        source_mel_len = source_mel.size(2)
        target_mel_len = target_mel.size(2)

        # Chunk params
        max_context_window = self.sr // self.hop_size * self.dit_max_context_len
        overlap_wave_len = self.overlap_frame_len * self.hop_size

        with torch.autocast(device_type=device.type, dtype=dtype):
            source_content_indices = self._process_content_features(source_wave_16k_tensor, is_narrow=False)
            target_content_indices = self._process_content_features(target_wave_16k_tensor, is_narrow=False)
            target_style = self.compute_style(target_wave_16k_tensor)
            prompt_condition, _, = self.cfm_length_regulator(
                target_content_indices, ylens=torch.LongTensor([target_mel_len]).to(device)
            )

        generated_wave_chunks = []
        processed_frames = 0
        previous_chunk = None

        if convert_style:
            with torch.autocast(device_type=device.type, dtype=dtype):
                source_narrow_indices = self._process_content_features(source_wave_16k_tensor, is_narrow=True)
                target_narrow_indices = self._process_content_features(target_wave_16k_tensor, is_narrow=True)
            src_narrow_reduced, _ = self.duration_reduction_func(source_narrow_indices[0], 1)
            tgt_narrow_reduced, tgt_narrow_len = self.duration_reduction_func(target_narrow_indices[0], 1)

            max_chunk_size = self.ar_max_content_len - tgt_narrow_len
            for i in range(0, len(src_narrow_reduced), max_chunk_size):
                is_last_chunk = i + max_chunk_size >= len(src_narrow_reduced)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    chunk = src_narrow_reduced[i : i + max_chunk_size]
                    if anonymization_only:
                        chunk_ar_cond = self.ar_length_regulator(chunk[None])[0]
                        chunk_ar_out = self.ar.generate(
                            chunk_ar_cond,
                            torch.zeros([1, 0]).long().to(device),
                            compiled_decode_fn=self.compiled_decode_fn,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    else:
                        chunk_ar_cond = self.ar_length_regulator(
                            torch.cat([tgt_narrow_reduced, chunk], dim=0)[None]
                        )[0]
                        chunk_ar_out = self.ar.generate(
                            chunk_ar_cond,
                            target_content_indices,
                            compiled_decode_fn=self.compiled_decode_fn,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    chunkar_out_mel_len = torch.LongTensor([
                        int(source_mel_len / source_content_indices.size(-1) * chunk_ar_out.size(-1) * length_adjust)
                    ]).to(device)
                    chunk_cond, _ = self.cfm_length_regulator(
                        chunk_ar_out, ylens=torch.LongTensor([chunkar_out_mel_len]).to(device)
                    )
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    original_len = cat_condition.size(1)
                    if self.dit_compiled:
                        cat_condition = torch.nn.functional.pad(
                            cat_condition,
                            (0, 0, 0, self.compile_len - cat_condition.size(1)),
                            value=0,
                        )
                    vc_mel = self.cfm.inference(
                        cat_condition,
                        torch.LongTensor([original_len]).to(device),
                        target_mel,
                        target_style,
                        diffusion_steps,
                        inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                        random_voice=anonymization_only,
                    )
                    vc_mel = vc_mel[:, :, target_mel_len:original_len]
                vc_wave = self.vocoder(vc_mel).squeeze()[None]
                processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = self._stream_wave_chunks(
                    vc_wave,
                    processed_frames,
                    vc_mel,
                    overlap_wave_len,
                    generated_wave_chunks,
                    previous_chunk,
                    is_last_chunk,
                    stream_output,
                )
                if stream_output and mp3_bytes is not None:
                    yield mp3_bytes, full_audio
                if should_break:
                    break
        else:
            cond, _ = self.cfm_length_regulator(
                source_content_indices, ylens=torch.LongTensor([source_mel_len]).to(device)
            )
            max_source_window = max_context_window - target_mel.size(2)
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                original_len = cat_condition.size(1)
                if self.dit_compiled:
                    cat_condition = torch.nn.functional.pad(
                        cat_condition,
                        (0, 0, 0, self.compile_len - cat_condition.size(1)),
                        value=0,
                    )
                with torch.autocast(device_type=device.type, dtype=torch.float32):
                    vc_mel = self.cfm.inference(
                        cat_condition,
                        torch.LongTensor([original_len]).to(device),
                        target_mel,
                        target_style,
                        diffusion_steps,
                        inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                        random_voice=anonymization_only,
                    )
                vc_mel = vc_mel[:, :, target_mel_len:original_len]
                vc_wave = self.vocoder(vc_mel).squeeze()[None]

                processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = self._stream_wave_chunks(
                    vc_wave,
                    processed_frames,
                    vc_mel,
                    overlap_wave_len,
                    generated_wave_chunks,
                    previous_chunk,
                    is_last_chunk,
                    stream_output,
                )
                if stream_output and mp3_bytes is not None:
                    yield mp3_bytes, full_audio
                if should_break:
                    break

        if not stream_output:
            return self.sr, np.concatenate(generated_wave_chunks) if len(generated_wave_chunks) > 0 else np.array([], dtype=np.float32)


    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice_arrays(
        self,
        source_wave: np.ndarray,
        target_wave: np.ndarray,
        source_sr: int,
        target_sr: int,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        intelligebility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        convert_style: bool = False,
        anonymization_only: bool = False,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        """
        Non-streaming convenience wrapper for in-memory voice conversion.
        Returns: (sample_rate, waveform_np)
        """
        return self.convert_voice_with_streaming_arrays(
            source_wave=source_wave,
            target_wave=target_wave,
            source_sr=source_sr,
            target_sr=target_sr,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            intelligebility_cfg_rate=intelligebility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            convert_style=convert_style,
            anonymization_only=anonymization_only,
            device=device,
            dtype=dtype,
            stream_output=False,
        )
