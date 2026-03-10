from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import torch
from mmgp import offload
from safetensors import safe_open
from shared.utils import files_locator as fl

from .florence2 import Florence2Config, Florence2ForConditionalGeneration, Florence2Processor
from .florence2.image_processing_florence2 import Florence2ImageProcessorLite

from transformers import AutoTokenizer, BartTokenizer, BartTokenizerFast


FLORENCE2_FOLDER = "Florence2"
LLAMA32_FOLDER = "Llama3_2"
LLAMAJOY_FOLDER = "llama-joycaption-beta-one-hf-llava"
PROMPT_ENHANCER_REPO = "DeepBeepMeep/LTX_Video"


@dataclass(slots=True)
class PromptEnhancerRuntime:
    image_caption_model: Any = None
    image_caption_processor: Any = None
    llm_model: Any = None
    llm_tokenizer: Any = None
    pipe_models: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, int] = field(default_factory=dict)


def ensure_prompt_enhancer_assets(process_files_def, enhancer_enabled: int, qwen_backend: str = "quanto_int8"):
    enhancer_enabled = int(enhancer_enabled)
    if enhancer_enabled == 1:
        process_files_def(
            repoId=PROMPT_ENHANCER_REPO,
            sourceFolderList=[FLORENCE2_FOLDER, LLAMA32_FOLDER],
            fileList=[
                ["config.json", "configuration_florence2.py", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"],
                ["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
            ],
        )
        return
    if enhancer_enabled == 2:
        process_files_def(
            repoId=PROMPT_ENHANCER_REPO,
            sourceFolderList=[FLORENCE2_FOLDER, LLAMAJOY_FOLDER],
            fileList=[
                ["config.json", "configuration_florence2.py", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"],
                ["config.json", "llama_config.json", "llama_joycaption_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
            ],
        )
        return
    if enhancer_enabled in (3, 4):
        from .qwen35_vl import ensure_qwen35_prompt_enhancer_assets, get_qwen35_prompt_enhancer_variant

        ensure_qwen35_prompt_enhancer_assets(process_files_def, backend=qwen_backend, variant=get_qwen35_prompt_enhancer_variant(enhancer_enabled))


def unload_prompt_enhancer_models(*models):
    seen = set()
    for model in models:
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()


def _set_pad_token_from_tokenizer(model, tokenizer):
    model.generation_config.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
        model.generation_config.pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id


def _load_llama32_prompt_enhancer():
    llm_model = offload.fast_load_transformers_model(
        fl.locate_file(f"{LLAMA32_FOLDER}/Llama3_2_quanto_bf16_int8.safetensors"),
        defaultConfigPath=fl.locate_file(f"{LLAMA32_FOLDER}/config.json", error_if_none=False),
        configKwargs={"attn_implementation": "sdpa", "hidden_act": "silu"},
        writable_tensors=False,
    )
    llm_model._validate_model_kwargs = lambda *_args, **_kwargs: None
    llm_model._offload_hooks = ["generate"]
    llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder(LLAMA32_FOLDER))
    _set_pad_token_from_tokenizer(llm_model, llm_tokenizer)
    llm_model.eval()
    return llm_model, llm_tokenizer, 5000


def _load_joycaption_prompt_enhancer():
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        rules = {"model.language_model": "model", "model.vision_tower": None, "model.multi_modal_projector": None}
        return tuple(offload.map_state_dict([sd, quant_map, tied_map], rules))

    llm_model = offload.fast_load_transformers_model(
        fl.locate_file(f"{LLAMAJOY_FOLDER}/llama_joycaption_quanto_bf16_int8.safetensors"),
        forcedConfigPath=fl.locate_file(f"{LLAMAJOY_FOLDER}/llama_config.json", error_if_none=False),
        configKwargs={"attn_implementation": "sdpa", "hidden_act": "silu"},
        preprocess_sd=preprocess_sd,
        writable_tensors=False,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder(LLAMAJOY_FOLDER))
    _set_pad_token_from_tokenizer(llm_model, llm_tokenizer)
    llm_model.eval()
    return llm_model, llm_tokenizer, 10000


def load_prompt_enhancer_runtime(process_files_def, enhancer_enabled: int, lm_decoder_engine: str = "", qwen_backend: str = "quanto_int8") -> PromptEnhancerRuntime:
    enhancer_enabled = int(enhancer_enabled)
    runtime = PromptEnhancerRuntime()
    if enhancer_enabled <= 0:
        return runtime

    ensure_prompt_enhancer_assets(process_files_def, enhancer_enabled=enhancer_enabled, qwen_backend=qwen_backend)

    if enhancer_enabled in (3, 4):
        from .qwen35_text import load_qwen35_text_prompt_enhancer
        from .qwen35_vl import (
            enhancer_quantization_QUANTO_INT8,
            get_qwen35_assets_dir_name,
            get_qwen35_prompt_enhancer_variant,
            load_qwen35_vl_prompt_enhancer,
        )

        backend = qwen_backend or enhancer_quantization_QUANTO_INT8
        qwen35_variant = get_qwen35_prompt_enhancer_variant(enhancer_enabled)
        assets_dir_name = get_qwen35_assets_dir_name(qwen35_variant)
        assets_dir = fl.locate_folder(assets_dir_name, error_if_none=False) or fl.get_download_location(assets_dir_name)
        runtime.llm_model = load_qwen35_text_prompt_enhancer(
            assets_dir=assets_dir,
            backend=backend,
            attn_implementation="sdpa",
            requested_lm_engine=lm_decoder_engine,
            variant=qwen35_variant,
        )
        runtime.llm_tokenizer = getattr(runtime.llm_model, "_prompt_enhancer_tokenizer", None)
        runtime.llm_model.eval()
        runtime.image_caption_model, vision_tower_model = load_qwen35_vl_prompt_enhancer(
            assets_dir=assets_dir,
            attn_implementation="sdpa",
            text_model=runtime.llm_model,
            backend=backend,
            variant=qwen35_variant,
        )
        runtime.image_caption_processor = getattr(runtime.image_caption_model, "_prompt_enhancer_processor", None)
        runtime.image_caption_model.eval()
        runtime.pipe_models["prompt_enhancer_image_caption_vision_tower_model"] = vision_tower_model
        runtime.pipe_models["prompt_enhancer_llm_model"] = runtime.llm_model
        runtime.budgets["prompt_enhancer_image_caption_vision_tower_model"] = 3000
        runtime.budgets["prompt_enhancer_llm_model"] = 10000
        return runtime

    runtime.image_caption_model, runtime.image_caption_processor = load_florence2(fl.locate_folder(FLORENCE2_FOLDER), attn_implementation="sdpa")
    runtime.image_caption_model._model_dtype = torch.float
    runtime.image_caption_model.eval()
    runtime.pipe_models["prompt_enhancer_image_caption_model"] = runtime.image_caption_model
    if enhancer_enabled == 1:
        runtime.llm_model, runtime.llm_tokenizer, budget = _load_llama32_prompt_enhancer()
    else:
        runtime.llm_model, runtime.llm_tokenizer, budget = _load_joycaption_prompt_enhancer()
    runtime.pipe_models["prompt_enhancer_llm_model"] = runtime.llm_model
    runtime.budgets["prompt_enhancer_llm_model"] = budget
    return runtime


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
