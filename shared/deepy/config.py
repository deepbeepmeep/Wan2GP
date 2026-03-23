from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEEPY_ENABLED_KEY = "deepy_enabled"
DEEPY_VRAM_MODE_KEY = "deepy_vram_mode"
DEEPY_TOOL_GEN_IMAGE_KEY = "deepy_tool_gen_image"
DEEPY_TOOL_EDIT_IMAGE_KEY = "deepy_tool_edit_image"
DEEPY_TOOL_GEN_VIDEO_KEY = "deepy_tool_gen_video"
DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY = "deepy_tool_gen_video_with_speech"
DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY = "deepy_tool_gen_speech_from_description"
DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY = "deepy_tool_gen_speech_from_sample"
DEEPY_CONTEXT_TOKENS_KEY = "deepy_context_tokens"
DEEPY_CUSTOM_SYSTEM_PROMPT_KEY = "deepy_custom_system_prompt"
DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY = "deepy_auto_cancel_queue_tasks"

DEEPY_VRAM_MODE_UNLOAD = "unload"
DEEPY_VRAM_MODE_ALWAYS_LOADED = "always_loaded"
DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST = "unload_on_request"
DEEPY_DEFAULT_GEN_IMAGE = "Z Image Turbo"
DEEPY_DEFAULT_EDIT_IMAGE = "Flux Klein 9B"
DEEPY_DEFAULT_GEN_VIDEO = "LTX-2 2.3 Distilled"
DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH = "Infinitalk"
DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION = "Qwen3 1.7B"
DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE = "Index TTS 2"
DEEPY_CONTEXT_TOKENS_MIN = 4096
DEEPY_CONTEXT_TOKENS_MAX = 256000
DEEPY_CONTEXT_TOKENS_DEFAULT = 32768
DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT = True
DEEPY_CONFIG_FILENAME = "wgp_config.json"

DEEPY_QWEN_ENHANCER_IDS = {3, 4}
_DEEPY_QWEN_VARIANT_LABELS = {3: "Qwen3.5-4B", 4: "Qwen3.5-9B"}
_DEEPY_QWEN_KV_CACHE_SPECS = {
    3: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    4: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
}
_DEEPY_DEFAULT_GEN_IMAGE_ALIASES = {"Z_Image_Turbo": DEEPY_DEFAULT_GEN_IMAGE}
_DEEPY_DEFAULT_EDIT_IMAGE_ALIASES = {"Qwen_Edit": DEEPY_DEFAULT_EDIT_IMAGE}
_DEEPY_DEFAULT_GEN_VIDEO_ALIASES = {"ltx2_22B_distilled": DEEPY_DEFAULT_GEN_VIDEO}
_DEEPY_RUNTIME_CONFIG: dict[str, Any] | None = None
_DEEPY_RUNTIME_CONFIG_FILENAME = ""


def normalize_deepy_enabled(value: Any) -> int:
    return 1 if int(value or 0) != 0 else 0


def normalize_deepy_vram_mode(value: Any) -> str:
    text = str(value or "").strip()
    if text == DEEPY_VRAM_MODE_ALWAYS_LOADED:
        return DEEPY_VRAM_MODE_ALWAYS_LOADED
    if text == DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST:
        return DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST
    return DEEPY_VRAM_MODE_UNLOAD


def _normalize_deepy_variant(value: Any, aliases: dict[str, str], default: str) -> str:
    text = str(value or "").strip()
    return aliases.get(text, text) or default


def normalize_deepy_tool_gen_image(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_GEN_IMAGE_ALIASES, DEEPY_DEFAULT_GEN_IMAGE)


def normalize_deepy_tool_edit_image(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_EDIT_IMAGE_ALIASES, DEEPY_DEFAULT_EDIT_IMAGE)


def normalize_deepy_tool_gen_video(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_GEN_VIDEO_ALIASES, DEEPY_DEFAULT_GEN_VIDEO)


def normalize_deepy_tool_gen_video_with_speech(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH)


def normalize_deepy_tool_gen_speech_from_description(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION)


def normalize_deepy_tool_gen_speech_from_sample(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE)


def normalize_deepy_context_tokens(value: Any) -> int:
    try:
        tokens = int(value or DEEPY_CONTEXT_TOKENS_DEFAULT)
    except Exception:
        tokens = DEEPY_CONTEXT_TOKENS_DEFAULT
    return max(DEEPY_CONTEXT_TOKENS_MIN, min(DEEPY_CONTEXT_TOKENS_MAX, tokens))


def normalize_deepy_custom_system_prompt(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def normalize_deepy_auto_cancel_queue_tasks(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def estimate_deepy_kv_cache_mb(enhancer_enabled: Any, context_tokens: Any) -> tuple[str | None, int | None]:
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except Exception:
        enhancer_no = 0
    spec = _DEEPY_QWEN_KV_CACHE_SPECS.get(enhancer_no)
    if spec is None:
        return None, None
    tokens = normalize_deepy_context_tokens(context_tokens)
    block_size = max(1, int(spec.get("kvcache_block_size", 256) or 256))
    num_blocks = (tokens + block_size - 1) // block_size
    total_bytes = (
        2
        * int(spec["num_kv_cache_layers"])
        * num_blocks
        * block_size
        * int(spec["num_key_value_heads"])
        * int(spec["head_dim"])
        * int(spec["dtype_bytes"])
    )
    return _DEEPY_QWEN_VARIANT_LABELS.get(enhancer_no, "Qwen3.5"), int(round(total_bytes / (1024 * 1024)))


def format_deepy_context_tokens_label(enhancer_enabled: Any, context_tokens: Any) -> str:
    variant_label, cache_mb = estimate_deepy_kv_cache_mb(enhancer_enabled, context_tokens)
    if variant_label is None or cache_mb is None:
        return "Context Window Tokens (KV cache: N/A)"
    return f"Context Window Tokens (KV cache ~{cache_mb} MB on {variant_label})"


def normalize_deepy_runtime_config(server_config: dict[str, Any] | None) -> dict[str, Any]:
    runtime_config = dict(server_config or {})
    runtime_config[DEEPY_ENABLED_KEY] = normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0))
    runtime_config[DEEPY_VRAM_MODE_KEY] = normalize_deepy_vram_mode(runtime_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))
    runtime_config[DEEPY_TOOL_GEN_IMAGE_KEY] = normalize_deepy_tool_gen_image(runtime_config.get(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE))
    runtime_config[DEEPY_TOOL_EDIT_IMAGE_KEY] = normalize_deepy_tool_edit_image(runtime_config.get(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_KEY] = normalize_deepy_tool_gen_video(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY] = normalize_deepy_tool_gen_video_with_speech(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY] = normalize_deepy_tool_gen_speech_from_description(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY] = normalize_deepy_tool_gen_speech_from_sample(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE))
    runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = normalize_deepy_context_tokens(runtime_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
    runtime_config[DEEPY_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_custom_system_prompt(runtime_config.get(DEEPY_CUSTOM_SYSTEM_PROMPT_KEY, ""))
    runtime_config[DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY] = normalize_deepy_auto_cancel_queue_tasks(runtime_config.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT))
    return runtime_config


def get_deepy_default_runtime_config() -> dict[str, Any]:
    return {
        DEEPY_ENABLED_KEY: 0,
        DEEPY_VRAM_MODE_KEY: DEEPY_VRAM_MODE_UNLOAD,
        DEEPY_TOOL_EDIT_IMAGE_KEY: DEEPY_DEFAULT_EDIT_IMAGE,
        DEEPY_TOOL_GEN_IMAGE_KEY: DEEPY_DEFAULT_GEN_IMAGE,
        DEEPY_TOOL_GEN_VIDEO_KEY: DEEPY_DEFAULT_GEN_VIDEO,
        DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH,
        DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION,
        DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE,
        DEEPY_CONTEXT_TOKENS_KEY: DEEPY_CONTEXT_TOKENS_DEFAULT,
        DEEPY_CUSTOM_SYSTEM_PROMPT_KEY: "",
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
    }


def set_deepy_runtime_config(server_config: dict[str, Any] | None, server_config_filename: str = "") -> dict[str, Any]:
    global _DEEPY_RUNTIME_CONFIG, _DEEPY_RUNTIME_CONFIG_FILENAME
    normalized = normalize_deepy_runtime_config(server_config)
    if isinstance(server_config, dict):
        server_config.update(normalized)
    _DEEPY_RUNTIME_CONFIG = normalized
    _DEEPY_RUNTIME_CONFIG_FILENAME = str(server_config_filename or "").strip()
    return normalized


def get_deepy_runtime_config() -> dict[str, Any]:
    global _DEEPY_RUNTIME_CONFIG
    if isinstance(_DEEPY_RUNTIME_CONFIG, dict) and len(_DEEPY_RUNTIME_CONFIG) > 0:
        return _DEEPY_RUNTIME_CONFIG
    candidate_paths = []
    if len(_DEEPY_RUNTIME_CONFIG_FILENAME) > 0:
        candidate_paths.append(Path(_DEEPY_RUNTIME_CONFIG_FILENAME))
    candidate_paths.append(Path.cwd() / DEEPY_CONFIG_FILENAME)
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as reader:
                loaded = json.load(reader)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return set_deepy_runtime_config(loaded, str(path))
    _DEEPY_RUNTIME_CONFIG = {}
    return _DEEPY_RUNTIME_CONFIG


def get_deepy_config_value(key: str, default: Any = None) -> Any:
    return get_deepy_runtime_config().get(key, default)


def deepy_requirement_met(server_config: dict[str, Any] | None) -> bool:
    runtime_config = server_config or {}
    try:
        enhancer_enabled = int(runtime_config.get("enhancer_enabled", 0) or 0)
    except Exception:
        enhancer_enabled = 0
    return enhancer_enabled in DEEPY_QWEN_ENHANCER_IDS


def deepy_enabled(server_config: dict[str, Any] | None) -> bool:
    runtime_config = server_config or {}
    return normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0)) == 1


def deepy_available(server_config: dict[str, Any] | None) -> bool:
    return deepy_enabled(server_config) and deepy_requirement_met(server_config)


def deepy_requirement_message(server_config: dict[str, Any] | None) -> str:
    if deepy_requirement_met(server_config):
        return (
            "<div style='color:#1b6d44; font-weight:600;'>"
            "Deepy is available because Prompt Enhancer is set to a Qwen3.5VL mode."
            "</div>"
        )
    return (
        "<div style='color:#8a4a14; font-weight:600;'>"
        "Deepy requires Prompt Enhancer to be set to Qwen3.5VL Abliterated 4B or 9B in the Extensions tab."
        "</div>"
    )
