from __future__ import annotations

import importlib.util
import os
import random
import re
import sys
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as taF
from torchaudio.transforms import Resample
from tqdm import tqdm
from einops import rearrange
from transformers import LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf

from mmgp import offload
from shared.utils import files_locator as fl
from .codecmanipulator import CodecManipulator
from .mmtokenizer import _MMSentencePieceTokenizer
from .llama_patched import LlamaForCausalLM


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


YUE_STAGE1_COT_REPO = "m-a-p/YuE-s1-7B-anneal-en-cot"
YUE_STAGE1_ICL_REPO = "m-a-p/YuE-s1-7B-anneal-en-icl"
YUE_STAGE2_REPO = "m-a-p/YuE-s2-1B-general"
YUE_TOKENIZER_FOLDER = "mm_tokenizer_v0.2_hf"
YUE_XCODEC_ROOT = "xcodec_mini_infer"


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id: int, end_id: int):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_text_or_file(value: str, label: str) -> str:
    if not value:
        return ""
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as fp:
            return fp.read()
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value)}")
    return value


def _split_lyrics(lyrics: str) -> list[str]:
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\\n{seg[1].strip()}\\n\\n" for seg in segments]
    return structured_lyrics


def _ensure_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.append(path)


def _load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class YuePipeline:
    def __init__(
        self,
        stage1_weights_path: str,
        stage2_weights_path: str,
        *,
        use_audio_prompt: bool = False,
        max_new_tokens: int = 3000,
        run_n_segments: int = 2,
        stage2_batch_size: int = 4,
        segment_duration: int = 6,
        prompt_start_time: float = 0.0,
        prompt_end_time: float = 30.0,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._interrupt = False

        assets = self._resolve_assets(
            stage1_weights_path,
            stage2_weights_path,
            use_audio_prompt,
        )
        stage1_weights_path = assets["stage1_weights"]
        stage1_config_path = assets["stage1_config"]
        stage2_weights_path = assets["stage2_weights"]
        stage2_config_path = assets["stage2_config"]
        tokenizer_path = assets["tokenizer_path"]
        xcodec_root = assets["xcodec_root"]
        xcodec_config_path = assets["xcodec_config"]
        xcodec_ckpt_path = assets["xcodec_ckpt"]
        vocoder_config_path = assets["vocoder_config"]
        vocoder_vocal_path = assets["vocoder_vocal"]
        vocoder_inst_path = assets["vocoder_inst"]

        self.stage1_weights_path = stage1_weights_path
        self.stage1_config_path = stage1_config_path
        self.stage2_weights_path = stage2_weights_path
        self.stage2_config_path = stage2_config_path

        self.max_new_tokens = int(max_new_tokens)
        self.run_n_segments = int(run_n_segments)
        self.stage2_batch_size = int(stage2_batch_size)
        self.segment_duration = int(segment_duration)
        self.prompt_start_time = float(prompt_start_time)
        self.prompt_end_time = float(prompt_end_time)

        if attn_implementation is None:
            attn_implementation = "flash_attention_2" if _flash_attn_available() else "sdpa"
        self.attn_implementation = attn_implementation
        print(f"[YUE] Attention backend: {self.attn_implementation}")

        self.mmtokenizer = _MMSentencePieceTokenizer(tokenizer_path)
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)

        self.model_stage1 = offload.fast_load_transformers_model(
            stage1_weights_path,
            modelClass=LlamaForCausalLM,
            defaultConfigPath=stage1_config_path,
            configKwargs={"attn_implementation": attn_implementation},
            default_dtype=torch.bfloat16,
            ignore_unused_weights=True,
        )
        self._set_attn_implementation(self.model_stage1, attn_implementation)
        self.model_stage1.eval()
        self.model_stage1._validate_model_kwargs = lambda *_args, **_kwargs: None
        self.model_stage1._offload_hooks = ["generate"]
        self.model = self.model_stage1

        self.model_stage2 = offload.fast_load_transformers_model(
            stage2_weights_path,
            modelClass=LlamaForCausalLM,
            defaultConfigPath=stage2_config_path,
            configKwargs={"attn_implementation": attn_implementation},
            default_dtype=torch.float16,
            ignore_unused_weights=True,
        )
        self._set_attn_implementation(self.model_stage2, attn_implementation)
        self.model_stage2.eval()
        self.model_stage2._validate_model_kwargs = lambda *_args, **_kwargs: None
        self.model_stage2._offload_hooks = ["generate"]
        self.model2 = self.model_stage2

        _ensure_sys_path(xcodec_root)
        _ensure_sys_path(os.path.join(xcodec_root, "descriptaudiocodec"))

        soundstream_path = os.path.join(xcodec_root, "models", "soundstream_hubert_new.py")
        soundstream_module = _load_module_from_path(
            "xcodec_soundstream_hubert_new", soundstream_path
        )
        SoundStream = soundstream_module.SoundStream
        sys.modules["models.soundstream_hubert_new"] = soundstream_module
        from vocoder import build_codec_model

        model_config = OmegaConf.load(xcodec_config_path)
        generator_name = model_config.generator.name
        generator_cls = getattr(soundstream_module, generator_name, None)
        if generator_cls is None:
            raise ValueError(f"Unsupported xcodec generator '{generator_name}'")

        prev_cwd = os.getcwd()
        os.chdir(os.path.dirname(xcodec_root))
        try:
            self.codec_model = generator_cls(**model_config.generator.config)
        finally:
            os.chdir(prev_cwd)
        parameter_dict = torch.load(
            xcodec_ckpt_path, map_location="cpu", weights_only=False
        )
        self.codec_model.load_state_dict(parameter_dict["codec_model"])
        self.codec_model.eval()
        self.codec_model._offload_hooks = ["encode", "decode", "get_embed"]

        self.vocoder_vocal, self.vocoder_inst = build_codec_model(
            vocoder_config_path, vocoder_vocal_path, vocoder_inst_path
        )
        self.vocoder_vocal.eval()
        self.vocoder_inst.eval()

        self.codec_fps = self.codectool.fps or 50
        self.codec_sample_rate = getattr(self.codec_model, "sample_rate", 16000)

    @staticmethod
    def _set_attn_implementation(model, attn_implementation: str) -> None:
        if not hasattr(model, "config"):
            return
        model.config._attn_implementation = attn_implementation
        model.config.attn_implementation = attn_implementation
        if hasattr(model, "generation_config"):
            model.generation_config._attn_implementation = attn_implementation
            model.generation_config.attn_implementation = attn_implementation

    @staticmethod
    def _resolve_assets(
        stage1_weights: str,
        stage2_weights: str,
        use_audio_prompt: bool,
    ) -> dict[str, str]:
        stage1_weights = stage1_weights or ""
        stage2_weights = stage2_weights or ""

        if stage1_weights:
            stage1_weights = fl.locate_file(stage1_weights, error_if_none=False) or stage1_weights
        if stage2_weights:
            stage2_weights = fl.locate_file(stage2_weights, error_if_none=False) or stage2_weights

        stage1_repo = YUE_STAGE1_ICL_REPO if use_audio_prompt else YUE_STAGE1_COT_REPO
        stage1_folder = os.path.basename(stage1_repo)
        stage2_folder = os.path.basename(YUE_STAGE2_REPO)

        stage1_dir = os.path.dirname(stage1_weights) if stage1_weights else ""
        stage2_dir = os.path.dirname(stage2_weights) if stage2_weights else ""

        stage1_config = os.path.join(stage1_dir, "config.json") if stage1_dir else ""
        stage2_config = os.path.join(stage2_dir, "config.json") if stage2_dir else ""

        stage1_config = fl.locate_file(stage1_config, error_if_none=False) or stage1_config
        stage2_config = fl.locate_file(stage2_config, error_if_none=False) or stage2_config

        if not stage1_config or not os.path.isfile(stage1_config):
            stage1_config = (
                fl.locate_file(os.path.join(stage1_folder, "config.json"), error_if_none=False)
                or stage1_config
            )
        if not stage2_config or not os.path.isfile(stage2_config):
            stage2_config = (
                fl.locate_file(os.path.join(stage2_folder, "config.json"), error_if_none=False)
                or stage2_config
            )

        tokenizer_path = None
        if stage1_weights:
            tokenizer_path = fl.locate_file(
                os.path.join(os.path.dirname(stage1_weights), "tokenizer.model"),
                error_if_none=False,
            )
        if tokenizer_path is None:
            tokenizer_path = fl.locate_file(
                os.path.join(YUE_TOKENIZER_FOLDER, "tokenizer.model"),
                error_if_none=False,
            )

        xcodec_root = fl.locate_folder(YUE_XCODEC_ROOT, error_if_none=False)

        xcodec_config = fl.locate_file(
            os.path.join(YUE_XCODEC_ROOT, "final_ckpt", "config.yaml"),
            error_if_none=False,
        )
        if xcodec_config is None and xcodec_root:
            xcodec_config = os.path.join(xcodec_root, "final_ckpt", "config.yaml")

        xcodec_ckpt = fl.locate_file(
            os.path.join(YUE_XCODEC_ROOT, "final_ckpt", "ckpt_00360000.pth"),
            error_if_none=False,
        )
        if xcodec_ckpt is None and xcodec_root:
            xcodec_ckpt = os.path.join(xcodec_root, "final_ckpt", "ckpt_00360000.pth")

        vocoder_config = fl.locate_file(
            os.path.join(YUE_XCODEC_ROOT, "decoders", "config.yaml"),
            error_if_none=False,
        )
        if vocoder_config is None and xcodec_root:
            vocoder_config = os.path.join(xcodec_root, "decoders", "config.yaml")

        vocoder_vocal = fl.locate_file(
            os.path.join(YUE_XCODEC_ROOT, "decoders", "decoder_131000.pth"),
            error_if_none=False,
        )
        if vocoder_vocal is None and xcodec_root:
            vocoder_vocal = os.path.join(xcodec_root, "decoders", "decoder_131000.pth")

        vocoder_inst = fl.locate_file(
            os.path.join(YUE_XCODEC_ROOT, "decoders", "decoder_151000.pth"),
            error_if_none=False,
        )
        if vocoder_inst is None and xcodec_root:
            vocoder_inst = os.path.join(xcodec_root, "decoders", "decoder_151000.pth")

        missing = []
        for name, value, is_dir in [
            ("stage1_weights", stage1_weights, False),
            ("stage1_config", stage1_config, False),
            ("stage2_weights", stage2_weights, False),
            ("stage2_config", stage2_config, False),
            ("tokenizer_path", tokenizer_path, False),
            ("xcodec_root", xcodec_root, True),
            ("xcodec_config", xcodec_config, False),
            ("xcodec_ckpt", xcodec_ckpt, False),
            ("vocoder_config", vocoder_config, False),
            ("vocoder_vocal", vocoder_vocal, False),
            ("vocoder_inst", vocoder_inst, False),
        ]:
            if not value:
                missing.append(name)
            elif is_dir and not os.path.isdir(value):
                missing.append(name)
            elif not is_dir and not os.path.isfile(value):
                missing.append(name)
        if missing:
            raise FileNotFoundError(f"Missing Yue assets: {missing}")

        return {
            "stage1_weights": stage1_weights,
            "stage1_config": stage1_config,
            "stage2_weights": stage2_weights,
            "stage2_config": stage2_config,
            "tokenizer_path": tokenizer_path,
            "xcodec_root": xcodec_root,
            "xcodec_config": xcodec_config,
            "xcodec_ckpt": xcodec_ckpt,
            "vocoder_config": vocoder_config,
            "vocoder_vocal": vocoder_vocal,
            "vocoder_inst": vocoder_inst,
        }

    def _check_abort(self) -> None:
        if self._interrupt:
            raise RuntimeError("Abort requested")

    def _load_audio_mono(self, filepath: str, sampling_rate: int) -> torch.Tensor:
        audio, sr = torchaudio.load(filepath)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != sampling_rate:
            resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    def _encode_audio(self, audio_prompt: torch.Tensor, target_bw: float = 0.5) -> np.ndarray:
        if len(audio_prompt.shape) < 3:
            audio_prompt = audio_prompt.unsqueeze(0)
        with torch.no_grad():
            raw_codes = self.codec_model.encode(audio_prompt.to(self.device), target_bw=target_bw)
        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int16)
        return raw_codes

    def _build_audio_prompt_ids(
        self,
        audio_prompt: Optional[str],
        audio_prompt2: Optional[str],
        prompt_start_time: float,
        prompt_end_time: float,
        use_dual_tracks: bool,
    ) -> list[int]:
        if use_dual_tracks:
            vocals_ids = self._load_audio_mono(audio_prompt, self.codec_sample_rate)
            instrumental_ids = self._load_audio_mono(audio_prompt2, self.codec_sample_rate)
            vocals_ids = self._encode_audio(vocals_ids, target_bw=0.5)
            instrumental_ids = self._encode_audio(instrumental_ids, target_bw=0.5)
            vocals_ids = self.codectool.npy2ids(vocals_ids[0])
            instrumental_ids = self.codectool.npy2ids(instrumental_ids[0])
            min_size = min(len(vocals_ids), len(instrumental_ids))
            vocals_ids = vocals_ids[0:min_size]
            instrumental_ids = instrumental_ids[0:min_size]
            ids_segment_interleaved = rearrange(
                [np.array(vocals_ids), np.array(instrumental_ids)], "b n -> (n b)"
            )
            audio_prompt_codec = ids_segment_interleaved[
                int(prompt_start_time * self.codec_fps * 2) : int(prompt_end_time * self.codec_fps * 2)
            ]
            audio_prompt_codec = audio_prompt_codec.tolist()
        else:
            audio_prompt_tensor = self._load_audio_mono(audio_prompt, self.codec_sample_rate)
            raw_codes = self._encode_audio(audio_prompt_tensor, target_bw=0.5)
            code_ids = self.codectool.npy2ids(raw_codes[0])
            audio_prompt_codec = code_ids[
                int(prompt_start_time * self.codec_fps) : int(prompt_end_time * self.codec_fps)
            ]

        audio_prompt_codec_ids = (
            [self.mmtokenizer.soa] + self.codectool.sep_ids + audio_prompt_codec + [self.mmtokenizer.eoa]
        )
        sentence_ids = (
            self.mmtokenizer.tokenize("[start_of_reference]")
            + audio_prompt_codec_ids
            + self.mmtokenizer.tokenize("[end_of_reference]")
        )
        return sentence_ids

    def _stage1_inference(
        self,
        genres: str,
        lyrics_input: str,
        run_n_segments: int,
        max_new_tokens: int,
        temperature: float,
        use_audio_prompt: bool,
        use_dual_tracks_prompt: bool,
        audio_prompt: Optional[str],
        audio_prompt2: Optional[str],
        prompt_start_time: float,
        prompt_end_time: float,
        callback=None,
        set_header_text=None,
    ):
        genres = genres.strip()
        lyrics = _split_lyrics(lyrics_input)
        if not lyrics:
            lyrics = [lyrics_input.strip()]
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [
            f"Generate music from the given lyrics segment by segment.\\n[Genre] {genres}\\n{full_lyrics}"
        ]
        prompt_texts += lyrics

        top_p = 0.93
        repetition_penalty = 1.2
        start_of_segment = self.mmtokenizer.tokenize("[start_of_segment]")
        end_of_segment = self.mmtokenizer.tokenize("[end_of_segment]")

        run_n_segments = min(max(1, run_n_segments), len(lyrics))
        raw_output = None

        for i, p in enumerate(tqdm(prompt_texts[1 : run_n_segments + 1]), 1):
            self._check_abort()
            segment_label = f"Segment {i}/{run_n_segments}"
            def hf_callback(step_idx: int, total_tokens: int, label: str = segment_label):
                self._check_abort()
                if callback is not None:
                    callback(
                        step_idx=step_idx,
                        override_num_inference_steps=total_tokens,
                        pass_no=1,
                        denoising_extra=label,
                    )

            if set_header_text is not None:
                set_header_text(f"Stage 1: Segment {i} of {run_n_segments}")

            section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
            guidance_scale = 1.5 if i <= 1 else 1.2

            if i == 1:
                if use_audio_prompt or use_dual_tracks_prompt:
                    sentence_ids = self._build_audio_prompt_ids(
                        audio_prompt,
                        audio_prompt2,
                        prompt_start_time,
                        prompt_end_time,
                        use_dual_tracks_prompt,
                    )
                    head_id = self.mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
                else:
                    head_id = self.mmtokenizer.tokenize(prompt_texts[0])
                prompt_ids = (
                    head_id
                    + start_of_segment
                    + self.mmtokenizer.tokenize(section_text)
                    + [self.mmtokenizer.soa]
                    + self.codectool.sep_ids
                )
            else:
                prompt_ids = (
                    end_of_segment
                    + start_of_segment
                    + self.mmtokenizer.tokenize(section_text)
                    + [self.mmtokenizer.soa]
                    + self.codectool.sep_ids
                )

            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self.device)
            input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
            max_context = 16384 - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                input_ids = input_ids[:, -max_context:]

            with torch.no_grad():
                output_seq = self.model_stage1.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=LogitsProcessorList(
                        [
                            BlockTokenRangeProcessor(0, 32002),
                            BlockTokenRangeProcessor(32016, 32017),
                        ]
                    ),
                    guidance_scale=guidance_scale,
                    callback=hf_callback,
                )

                if output_seq[0][-1].item() != self.mmtokenizer.eoa:
                    tensor_eoa = torch.as_tensor([[self.mmtokenizer.eoa]]).to(self.device)
                    output_seq = torch.cat((output_seq, tensor_eoa), dim=1)

            if i > 1:
                raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
            else:
                raw_output = output_seq

        if raw_output is None:
            return None

        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(
                f"Invalid pairs of soa and eoa. Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}"
            )

        vocals = []
        instrumentals = []
        range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0
        for idx in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[idx] + 1 : eoa_idx[idx]]
            if len(codec_ids) == 0:
                continue
            if codec_ids[0] == self.codectool.sep_ids[0]:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            vocals_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)

        if not vocals or not instrumentals:
            return None

        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)
        return [vocals, instrumentals]

    def _stage2_generate(self, prompt, batch_size: int, segment_duration: int, callback=None):
        codec_ids = self.codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codectool.offset_tok_ids(
            codec_ids,
            global_offset=self.codectool.global_offset,
            codebook_size=self.codectool.codebook_size,
            num_codebooks=self.codectool.num_codebooks,
        ).astype(np.int32)

        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * segment_duration * self.codec_fps
                idx_end = (i + 1) * segment_duration * self.codec_fps
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([self.mmtokenizer.soa, self.mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([self.mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1,
            )
        else:
            prompt_ids = np.concatenate(
                [
                    np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                    codec_ids.flatten(),
                    np.array([self.mmtokenizer.stage_2]),
                ]
            ).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(self.device)
        prompt_ids = torch.as_tensor(prompt_ids).to(self.device)
        len_prompt = prompt_ids.shape[-1]

        block_list = LogitsProcessorList(
            [
                BlockTokenRangeProcessor(0, 46358),
                BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size),
            ]
        )

        max_tokens = codec_ids.shape[1] * 8
        i = 0
        real_max_length = codec_ids.shape[1] * 8 + prompt_ids.shape[1]
        session_cache = {"real_max_length": real_max_length}

        for frames_idx in range(codec_ids.shape[1]):
            if i % 96 == 0 and callback is not None:
                callback(
                    step_idx=i,
                    override_num_inference_steps=real_max_length,
                    pass_no=2,
                    denoising_extra="Stage 2",
                )
            self._check_abort()

            cb0 = codec_ids[:, frames_idx : frames_idx + 1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = self.model_stage2.generate(
                    input_ids=input_ids,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=block_list,
                    session_cache=session_cache,
                )
            if stage2_output.shape[1] - prompt_ids.shape[1] != 7:
                raise ValueError(
                    f"Stage2 output new tokens={stage2_output.shape[1] - prompt_ids.shape[1]}"
                )
            prompt_ids = stage2_output
            i += 8

        del session_cache

        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def _stage2_inference(
        self,
        stage1_output_set,
        batch_size: int,
        segment_duration: int,
        callback=None,
        set_header_text=None,
    ):
        stage2_result = []
        for i in tqdm(range(len(stage1_output_set))):
            self._check_abort()
            prefix = "Stage 2.1: Sampling Vocal track" if i == 0 else "Stage 2.2: Sampling Instrumental track"
            if set_header_text is not None:
                set_header_text(prefix)

            prompt = stage1_output_set[i].astype(np.int32)
            output_duration = prompt.shape[-1] // self.codec_fps // segment_duration * segment_duration
            num_batch = output_duration // segment_duration
            any_trail = output_duration * self.codec_fps != prompt.shape[-1]

            if num_batch <= batch_size:
                if set_header_text is not None:
                    if any_trail:
                        set_header_text(prefix + ", segment 1 of 2")
                    else:
                        set_header_text(prefix)
                output = self._stage2_generate(
                    prompt[:, : output_duration * self.codec_fps],
                    batch_size=num_batch,
                    segment_duration=segment_duration,
                    callback=callback,
                )
            else:
                segments = []
                num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
                max_segments = num_segments + 1 if any_trail else num_segments
                for seg in range(num_segments):
                    if set_header_text is not None:
                        set_header_text(prefix + f", segment {seg + 1} of {max_segments}")
                    start_idx = seg * batch_size * segment_duration * self.codec_fps
                    end_idx = min((seg + 1) * batch_size * segment_duration * self.codec_fps, output_duration * self.codec_fps)
                    current_batch_size = (
                        batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                    )
                    segment = self._stage2_generate(
                        prompt[:, start_idx:end_idx],
                        batch_size=current_batch_size,
                        segment_duration=segment_duration,
                        callback=callback,
                    )
                    segments.append(segment)
                output = np.concatenate(segments, axis=0)

            if any_trail:
                if set_header_text is not None:
                    set_header_text(prefix + f", segment {max_segments} of {max_segments}")
                ending = self._stage2_generate(
                    prompt[:, output_duration * self.codec_fps :],
                    batch_size=1,
                    segment_duration=segment_duration,
                    callback=callback,
                )
                output = np.concatenate([output, ending], axis=0)

            output = self.codectool_stage2.ids2npy(output)
            fixed_output = output.copy()
            for row_idx, line in enumerate(output):
                for col_idx, element in enumerate(line):
                    if element < 0 or element > 1023:
                        values, counts = np.unique(line, return_counts=True)
                        fixed_output[row_idx, col_idx] = values[np.argmax(counts)]

            stage2_result.append(fixed_output)
        return stage2_result

    def _decode_track(self, codec_result: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            decoded = self.codec_model.decode(
                torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(self.device)
            )
        decoded = decoded.cpu().squeeze(0)
        return decoded

    def _vocoder_decode(self, decoder, codec_result: np.ndarray) -> torch.Tensor:
        compressed = torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(1)
        embeds = self.codec_model.get_embed(compressed.to(self.device))
        if not torch.is_tensor(embeds):
            embeds = torch.tensor(embeds)
        embeds = embeds.to(self.device)
        with torch.no_grad():
            out = decoder(embeds)
        return out.detach().cpu()

    def _replace_low_freq_with_energy_matched(
        self, wave_a: torch.Tensor, sr_a: int, wave_b: torch.Tensor, sr_b: int, cutoff_freq: float = 5500.0
    ) -> torch.Tensor:
        wave_a = wave_a.float()
        wave_b = wave_b.float()
        if sr_a != sr_b:
            resampler = Resample(orig_freq=sr_a, new_freq=sr_b)
            wave_a = resampler(wave_a)

        wave_a_low = taF.lowpass_biquad(wave_a, sample_rate=sr_b, cutoff_freq=cutoff_freq)
        wave_b_low = taF.lowpass_biquad(wave_b, sample_rate=sr_b, cutoff_freq=cutoff_freq)

        a_rms = wave_a_low.pow(2).mean().sqrt().item() + 1e-10
        b_rms = wave_b_low.pow(2).mean().sqrt().item() + 1e-10

        scale_factor = b_rms / a_rms
        wave_a_low_matched = wave_a_low * scale_factor

        wave_b_high = taF.highpass_biquad(wave_b, sample_rate=sr_b, cutoff_freq=cutoff_freq)

        min_length = min(wave_a_low_matched.size(1), wave_b_high.size(1))
        wave_a_low_matched = wave_a_low_matched[:, :min_length]
        wave_b_high = wave_b_high[:, :min_length]

        return wave_a_low_matched + wave_b_high

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        alt_prompt: Optional[str] = None,
        audio_guide2: Optional[str] = None,
        audio_prompt_type: str = "",
        temperature: float = 1.0,
        yue_max_new_tokens: Optional[int] = None,
        yue_run_n_segments: Optional[int] = None,
        yue_stage2_batch_size: Optional[int] = None,
        yue_segment_duration: Optional[int] = None,
        yue_prompt_start_time: Optional[float] = None,
        yue_prompt_end_time: Optional[float] = None,
        seed: int = -1,
        callback=None,
        set_header_text=None,
        **kwargs,
    ):
        self._interrupt = False
        lyrics_text = _read_text_or_file(input_prompt, "Lyrics prompt")
        if not lyrics_text.strip():
            raise ValueError("Lyrics prompt cannot be empty for Yue generation.")

        genres = _read_text_or_file(alt_prompt, "Genres prompt")
        if not genres.strip():
            raise ValueError("Genres prompt cannot be empty for Yue generation.")

        audio_prompt_type = audio_prompt_type or ""
        use_dual_tracks_prompt = "A" in audio_prompt_type and "B" in audio_prompt_type and audio_guide and audio_guide2
        use_audio_prompt = "A" in audio_prompt_type and not use_dual_tracks_prompt and audio_guide

        max_new_tokens = self.max_new_tokens if yue_max_new_tokens is None else int(yue_max_new_tokens)
        run_n_segments = self.run_n_segments if yue_run_n_segments is None else int(yue_run_n_segments)
        stage2_batch_size = (
            self.stage2_batch_size if yue_stage2_batch_size is None else int(yue_stage2_batch_size)
        )
        segment_duration = (
            self.segment_duration if yue_segment_duration is None else int(yue_segment_duration)
        )
        prompt_start_time = (
            self.prompt_start_time if yue_prompt_start_time is None else float(yue_prompt_start_time)
        )
        prompt_end_time = (
            self.prompt_end_time if yue_prompt_end_time is None else float(yue_prompt_end_time)
        )

        if seed is None or seed < 0:
            seed = random.randint(0, 999999999)
        _seed_everything(seed)

        stage1_output_set = self._stage1_inference(
            genres,
            lyrics_text,
            run_n_segments,
            max_new_tokens,
            float(temperature),
            use_audio_prompt,
            use_dual_tracks_prompt,
            audio_guide,
            audio_guide2,
            prompt_start_time,
            prompt_end_time,
            callback=callback,
            set_header_text=set_header_text,
        )
        if stage1_output_set is None:
            return None

        stage2_result = self._stage2_inference(
            stage1_output_set,
            batch_size=stage2_batch_size,
            segment_duration=segment_duration,
            callback=callback,
            set_header_text=set_header_text,
        )
        if stage2_result is None or len(stage2_result) < 2:
            return None

        vocal_codes, inst_codes = stage2_result[0], stage2_result[1]
        vocal_16k = self._decode_track(vocal_codes)
        inst_16k = self._decode_track(inst_codes)
        min_len = min(vocal_16k.shape[-1], inst_16k.shape[-1])
        mix_16k = vocal_16k[:, :min_len] + inst_16k[:, :min_len]

        vocal_hi = self._vocoder_decode(self.vocoder_vocal, vocal_codes)
        inst_hi = self._vocoder_decode(self.vocoder_inst, inst_codes)
        min_len_hi = min(vocal_hi.shape[-1], inst_hi.shape[-1])
        mix_hi = vocal_hi[:, :min_len_hi] + inst_hi[:, :min_len_hi]

        final_audio = self._replace_low_freq_with_energy_matched(
            mix_16k, 16000, mix_hi, 44100
        )
        final_audio = final_audio.clamp(-0.99, 0.99)
        return {"x": final_audio, "audio_sampling_rate": 44100}

    def release(self) -> None:
        for module in [self.model_stage1, self.model_stage2, self.codec_model]:
            if hasattr(module, "to"):
                module.to("cpu")
        for module in [self.vocoder_vocal, self.vocoder_inst]:
            if hasattr(module, "to"):
                module.to("cpu")
