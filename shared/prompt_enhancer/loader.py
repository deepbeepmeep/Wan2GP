from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from safetensors import safe_open

from .florence2 import Florence2Config, Florence2ForConditionalGeneration, Florence2Processor
from .florence2.image_processing_florence2 import Florence2ImageProcessorLite

from transformers import BartTokenizer, BartTokenizerFast


def _load_state_dict(weights_path: Path) -> dict:
    if weights_path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    return torch.load(str(weights_path), map_location="cpu")


def _resolve_weights_path(model_path: Path) -> Path:
    # Prefer fp32 weights for stability/quality when available.
    preferred = model_path / "xmodel.safetensors"
    if preferred.exists():
        return preferred
    fallback = model_path / "model.safetensors"
    if fallback.exists():
        return fallback
    fallback = model_path / "pytorch_model.bin"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No Florence2 weights found in {model_path} (expected model.safetensors/xmodel.safetensors/pytorch_model.bin)"
    )


def load_florence2(
    model_dir: str,
    attn_implementation: str = "sdpa",
) -> Tuple[Florence2ForConditionalGeneration, Florence2Processor]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Florence2 folder not found: {model_path}")

    config = Florence2Config.from_pretrained(str(model_path))
    if attn_implementation:
        config._attn_implementation = attn_implementation
    weights_path = _resolve_weights_path(model_path)
    state_dict = _load_state_dict(weights_path)

    model = Florence2ForConditionalGeneration(config)
    load_info = model.load_state_dict(state_dict, strict=False)
    del state_dict
    if load_info.missing_keys:
        allowed_missing = {
            "language_model.model.encoder.embed_tokens.weight",
            "language_model.model.decoder.embed_tokens.weight",
        }
        extra_missing = [k for k in load_info.missing_keys if k not in allowed_missing]
        if extra_missing:
            print(f"Florence2 missing keys: {extra_missing}")
    if load_info.unexpected_keys:
        print(f"Florence2 unexpected keys: {len(load_info.unexpected_keys)}")
    model.eval()

    image_processor = Florence2ImageProcessorLite.from_preprocessor_config(model_path)
    tokenizer = None
    tokenizer_errors = []
    for tok_cls in (BartTokenizerFast, BartTokenizer):
        try:
            tokenizer = tok_cls.from_pretrained(str(model_path))
            break
        except Exception as exc:
            tokenizer_errors.append(exc)
    if tokenizer is None:
        raise RuntimeError(f"Unable to load Florence2 tokenizer: {tokenizer_errors}")
    try:
        processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)
    except TypeError as exc:
        if "CLIPImageProcessor" not in str(exc):
            raise
        try:
            from transformers import CLIPImageProcessor
        except Exception:
            from transformers.models.clip import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(str(model_path))
        processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    return model, processor
