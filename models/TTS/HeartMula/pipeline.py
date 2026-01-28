from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import contextlib

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from shared.utils import files_locator as fl

from .heartcodec.configuration_heartcodec import HeartCodecConfig
from .heartcodec.modeling_heartcodec import HeartCodec
from .heartmula.configuration_heartmula import HeartMuLaConfig
from .heartmula.modeling_heartmula import HeartMuLa


def _resolve_paths(
    pretrained_path: Path, version: str, heartmula_weights_path: Optional[str] = None
):
    config_path = ( Path(__file__).resolve().parent / "config" / f"heartmula_{version}.json" )

    heartcodec_dir = fl.locate_folder("HeartMula")
    tokenizer_path = fl.locate_file( os.path.join("HeartMula", "tokenizer.json"))
    gen_config_path = fl.locate_file( os.path.join("HeartMula", "gen_config.json"))

    weights_path = heartmula_weights_path

    return (
        weights_path,
        Path(config_path),
        Path(heartcodec_dir),
        Path(tokenizer_path),
        Path(gen_config_path),
    )


def _resolve_codec_names(codec_version: Optional[str]) -> tuple[str, str]:
    if codec_version:
        suffix = f"_{codec_version}"
    else:
        suffix = ""
    return f"HeartMula_codec{suffix}.safetensors", f"codec_config{suffix}.json"


def _strip_heartmula_rope_cache(state_dict):
    remove_keys = (
        "backbone.layers.0.attn.pos_embeddings.theta",
        "backbone.layers.0.attn.pos_embeddings.cache",
        "decoder.layers.0.attn.pos_embeddings.theta",
        "decoder.layers.0.attn.pos_embeddings.cache",
    )
    for key in remove_keys:
        state_dict.pop(key, None)
    return state_dict


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = fp.read()
        import json

        return cls(**json.loads(data))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HeartMuLaPipeline:
    def __init__(
        self,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
        version: str = "3B",
        heartmula_dtype: Optional[torch.dtype] = None,
        heartcodec_dtype: Optional[torch.dtype] = None,
        heartmula_weights_path: Optional[str] = None,
        cfg_scale: float = 1.5,
        topk: int = 50,
        max_audio_length_ms: int = 120000,
        codec_steps: int = 10,
        codec_guidance_scale: float = 1.25,
        codec_version: str = "",
        VAE_dtype = torch.float32,
    ):
        self.device = torch.device("cpu")
        self.mula_device = self.device
        self.codec_device = self.device
        self.mula_dtype = None
        self.codec_dtype = None

        self.cfg_scale = cfg_scale
        self.topk = topk
        self.max_audio_length_ms = max_audio_length_ms
        self.codec_steps = codec_steps
        self.codec_guidance_scale = codec_guidance_scale
        self.codec_version = codec_version
        self.heartmula_weights_path = heartmula_weights_path
        self.VAE_dtype = VAE_dtype

        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else Path(
            fl.get_download_location()
        )
        self.version = version
        self._interrupt = False
        self._early_stop = False

        self._parallel_number = 8 + 1
        self._muq_dim = 512

        self._load_models()

    def _load_models(self):
        (
            mula_weights_path,
            mula_config_path,
            codec_path,
            tokenizer_path,
            gen_config_path,
        ) = _resolve_paths(
            self.ckpt_root,
            self.version,
            heartmula_weights_path=self.heartmula_weights_path,
        )
        from accelerate import init_empty_weights
        from mmgp import offload
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.gen_config = HeartMuLaGenConfig.from_file(str(gen_config_path))
        with open(mula_config_path, encoding="utf-8") as fp:
            mula_config = HeartMuLaConfig(**json.load(fp))

        with init_empty_weights():
            self.mula = HeartMuLa(mula_config)
        offload.load_model_data(
            self.mula,
            str(mula_weights_path),
            default_dtype=None,
            writable_tensors=False,
            preprocess_sd=_strip_heartmula_rope_cache,
        )

        decoder = self.mula.decoder
        delattr(self.mula, "decoder")
        self.mula.decoder = [decoder]

        if hasattr(self.mula, "_interrupt_check"):
            self.mula._interrupt_check = self._abort_requested
        self.model = self.mula
        self.mula.eval()
        first_param = next(self.mula.parameters(), None)
        if first_param is not None:
            self.mula_dtype = first_param.dtype
        codec_weights_name, codec_config_name = _resolve_codec_names(self.codec_version)
        codec_weights_path = fl.locate_file(
            os.path.join("HeartMula", codec_weights_name), error_if_none=False
        )
        codec_weights_path = (
            Path(codec_weights_path)
            if codec_weights_path
            else Path(codec_path) / codec_weights_name
        )
        if not codec_weights_path.is_file():
            raise FileNotFoundError(
                f"Expected HeartCodec weights at {codec_weights_path} but not found."
            )
        codec_config_path = fl.locate_file(
            os.path.join("HeartMula", codec_config_name), error_if_none=False
        )
        codec_config_path = (
            Path(codec_config_path)
            if codec_config_path
            else Path(codec_path) / codec_config_name
        )
        if not codec_config_path.is_file():
            raise FileNotFoundError(
                f"Expected HeartCodec config at {codec_config_path} but not found."
            )
        with open(codec_config_path, encoding="utf-8") as fp:
            codec_config = HeartCodecConfig(**json.load(fp))
        with init_empty_weights():
            self.codec = HeartCodec(codec_config)
        self.codec._offload_hooks = ["detokenize"]

        self.codec._model_dtype = self.VAE_dtype
        offload.load_model_data(
            self.codec,
            str(codec_weights_path),
            default_dtype=self.VAE_dtype,
            writable_tensors=False,
        )
        self.codec.eval()
        first_param = next(self.codec.parameters(), None)
        if first_param is not None:
            self.codec_dtype = first_param.dtype

        self.sample_rate = getattr(self.codec, "sample_rate", 48000)

    def _get_mula_device(self) -> torch.device:
        if self.mula is None:
            return self.device
        text_embed = getattr(self.mula, "text_embeddings", None)
        if text_embed is not None and hasattr(text_embed, "weight"):
            return text_embed.weight.device
        first_param = next(self.mula.parameters(), None)
        return first_param.device if first_param is not None else self.device

    def _ensure_mula_loaded(self) -> None:
        if self.mula is None:
            return
        mm_manager = getattr(self.mula, "_mm_manager", None)
        mm_id = getattr(self.mula, "_mm_id", None)
        if mm_manager is None or mm_id is None:
            return
        mm_manager.ensure_model_loaded(mm_id)

    def _move_model_inputs(
        self, model_inputs: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        non_blocking = device.type == "cuda"
        for key, value in model_inputs.items():
            if torch.is_tensor(value) and value.device != device:
                model_inputs[key] = value.to(device, non_blocking=non_blocking)
        return model_inputs

    def _abort_requested(self) -> bool:
        return bool(self._interrupt)

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _read_text_or_file(self, value: str, label: str) -> str:
        if os.path.isfile(value):
            with open(value, encoding="utf-8") as fp:
                return fp.read()
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string, got {type(value)}")
        return value

    def _build_model_inputs(
        self, lyrics: str, tags: str, cfg_scale: float
    ) -> Dict[str, Any]:
        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.tokenizer.encode(tags).ids
        if tags_ids[0] != self.gen_config.text_bos_id:
            tags_ids = [self.gen_config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.gen_config.text_eos_id:
            tags_ids = tags_ids + [self.gen_config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.mula_dtype)
        muq_idx = len(tags_ids)

        lyrics = lyrics.lower()
        lyrics_ids = self.tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.gen_config.text_bos_id:
            lyrics_ids = [self.gen_config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.gen_config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.gen_config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1
        muq_idx_tensor = torch.full((bs_size,), muq_idx, dtype=torch.long)

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": muq_idx_tensor,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        callback=None,
    ):
        prompt_tokens = model_inputs["tokens"]
        mula_device = prompt_tokens.device
        prompt_tokens_mask = model_inputs["tokens_mask"]
        continuous_segment = model_inputs["muq_embed"]
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"]
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.mula.setup_caches(bs_size)
        self.mula.move_causal_masks(mula_device)
        flash_dtype = self.mula_dtype
        if flash_dtype is None:
            first_param = next(self.mula.parameters(), None)
            if first_param is not None:
                flash_dtype = first_param.dtype
        if flash_dtype is not None:
            self.mula.prepare_flash(mula_device, flash_dtype)
        if self._abort_requested():
            return None
        try:
            curr_token = self.mula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
            if curr_token is None:
                return None
            frames.append(curr_token[0:1,])
            early_stop_now = self._early_stop_requested()

            def _pad_audio_token(token: torch.Tensor):
                padded_token = (
                    torch.ones(
                        (token.shape[0], self._parallel_number),
                        device=token.device,
                        dtype=torch.long,
                    )
                    * self.gen_config.empty_id
                )
                padded_token[:, :-1] = token
                padded_token = padded_token.unsqueeze(1)
                padded_token_mask = torch.ones_like(
                    padded_token, device=token.device, dtype=torch.bool
                )
                padded_token_mask[..., -1] = False
                return padded_token, padded_token_mask

            frame_duration_ms = 80  # 80 ms per audio token frame.
            max_audio_frames = max_audio_length_ms // frame_duration_ms
            progress_total_seconds = max(1, max_audio_length_ms // 1000)
            if callback is not None:
                callback(
                    step_idx=-1,
                    override_num_inference_steps=progress_total_seconds,
                    denoising_extra=f"0s/{progress_total_seconds}s",
                    progress_unit="seconds",
                )

            if not early_stop_now:
                for i in tqdm(range(max_audio_frames)):
                    if self._abort_requested():
                        return None
                    curr_token, curr_token_mask = _pad_audio_token(curr_token)
                    curr_token = self.mula.generate_frame(
                        tokens=curr_token,
                        tokens_mask=curr_token_mask,
                        input_pos=prompt_pos[..., -1:] + i + 1,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )
                    if curr_token is None:
                        return None
                    if torch.any(curr_token[0:1, :] >= self.gen_config.audio_eos_id):
                        break
                    frames.append(curr_token[0:1,])
                    if self._early_stop_requested():
                        break
                    if i % 10 == 0 and callback is not None:
                        generated_ms = len(frames) * frame_duration_ms
                        generated_seconds_int = min(
                            progress_total_seconds,
                            generated_ms // 1000,
                        )
                        callback(
                            step_idx=generated_seconds_int - 1,
                            override_num_inference_steps=progress_total_seconds,
                            denoising_extra=(
                                f"{generated_seconds_int}s/{progress_total_seconds}s"
                            ),
                            progress_unit="seconds",
                        )
            frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
            return {"frames": frames}
        finally:
            # Drop KV cache tensors as soon as we're done with generation.
            try:
                self.mula.move_causal_masks(torch.device("cpu"))
                self.mula.release_caches()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _decode(self, frames: torch.Tensor, callback=None):
        if self._abort_requested():
            return None
        wav = self.codec.detokenize(
            frames.to(self.codec_device),
            num_steps=self.codec_steps,
            guidance_scale=self.codec_guidance_scale,
            disable_progress=False,
            abort_signal=self._abort_requested,
        )
        if wav is None:
            return None
        if callback is not None:
            callback(step_idx=-1, force_refresh=True)
        return wav

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        alt_prompt: Optional[str] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        self._interrupt = False
        self._early_stop = False
        seed = kwargs.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        if seed is not None and seed >= 0:
            _seed_everything(seed)
        if self.mula is not None:
            self.mula_device = torch.device("cpu")
            text_embed = getattr(self.mula, "text_embeddings", None)
            if text_embed is not None and hasattr(text_embed, "weight"):
                self.mula_dtype = text_embed.weight.dtype
            else:
                first_param = next(self.mula.parameters(), None)
                if first_param is not None:
                    self.mula_dtype = first_param.dtype
        if self.codec is not None:
            first_param = next(self.codec.parameters(), None)
            if first_param is not None:
                self.codec_device = first_param.device
                self.codec_dtype = first_param.dtype
        if not input_prompt or not input_prompt.strip():
            raise ValueError("Lyrics prompt cannot be empty for HeartMuLa generation.")
        if alt_prompt is None or not str(alt_prompt).strip():
            raise ValueError("Keywords prompt cannot be empty for HeartMuLa generation.")
        if audio_guide or kwargs.get("audio_guide2"):
            raise ValueError("HeartMuLa does not support reference audio yet.")

        lyrics = self._read_text_or_file(input_prompt, "Lyrics prompt")
        tags = self._read_text_or_file(str(alt_prompt), "Keywords prompt")
        if not lyrics.strip():
            raise ValueError("Lyrics prompt cannot be empty for HeartMuLa generation.")
        if not tags.strip():
            raise ValueError("Keywords prompt cannot be empty for HeartMuLa generation.")

        cfg_scale = float(kwargs.get("cfg_scale", self.cfg_scale))
        topk_value = kwargs.get("topk", None)
        if topk_value is None:
            topk_value = kwargs.get("top_k", self.topk)
        try:
            topk = int(topk_value)
        except (TypeError, ValueError):
            topk = int(self.topk)
        duration_seconds = kwargs.get("duration_seconds", None)
        if duration_seconds is not None:
            try:
                duration_seconds = float(duration_seconds)
            except (TypeError, ValueError):
                duration_seconds = None
        if duration_seconds is not None and duration_seconds > 0:
            max_audio_length_ms = int(round(duration_seconds * 1000.0))
        else:
            max_audio_length_ms = int(
                kwargs.get("max_audio_length_ms", self.max_audio_length_ms)
            )
        callback = kwargs.get("callback")

        model_inputs = self._build_model_inputs(lyrics, tags, cfg_scale=cfg_scale)
        self._ensure_mula_loaded()
        target_device = self._get_mula_device()
        if target_device.type != "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")
        model_inputs = self._move_model_inputs(model_inputs, target_device)
        outputs = self._forward(
            model_inputs,
            max_audio_length_ms=max_audio_length_ms,
            temperature=float(temperature),
            topk=topk,
            cfg_scale=cfg_scale,
            callback=callback,
        )
        if outputs is None:
            return None
        wav = self._decode(outputs["frames"], callback=callback)
        if wav is None:
            return None
        return {"x": wav, "audio_sampling_rate": self.sample_rate}

    def release(self) -> None:
        if hasattr(self, "mula") and self.mula is not None:
            self.mula = None
        if hasattr(self, "model"):
            self.model = None
        if hasattr(self, "codec") and self.codec is not None:
            self.codec = None
