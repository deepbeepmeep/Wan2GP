from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version


DEEPY_ENABLED_KEY = "deepy_enabled"
DEEPY_TYPE_KEY = "deepy_type"
DEEPY_VRAM_MODE_KEY = "deepy_vram_mode"
DEEPY_TOOL_GEN_IMAGE_KEY = "deepy_tool_gen_image"
DEEPY_TOOL_EDIT_IMAGE_KEY = "deepy_tool_edit_image"
DEEPY_TOOL_GEN_VIDEO_KEY = "deepy_tool_gen_video"
DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY = "deepy_tool_gen_video_with_speech"
DEEPY_TOOL_GEN_SONG_KEY = "deepy_tool_gen_song"
DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY = "deepy_tool_gen_speech_from_description"
DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY = "deepy_tool_gen_speech_from_sample"
DEEPY_CONTEXT_TOKENS_KEY = "deepy_context_tokens"
DEEPY_KV_CACHE_QUANTIZATION_KEY = "deepy_kv_cache_quantization"
DEEPY_COMPACTION_TYPE_KEY = "deepy_compaction_type"
DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY = "deepy_zero_custom_system_prompt"
DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY = "deepy_prime_custom_system_prompt"
DEEPY_PRIME_MCP_SERVERS_KEY = "deepy_prime_mcp_servers"
DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY = "deepy_mcp_auto_discover_paths"
DEEPY_ALLOW_READ_FILE_SYSTEM_KEY = "deepy_allow_read_file_system"
DEEPY_FILE_SYSTEM_PATHS_KEY = "deepy_file_system_paths"
DEEPY_READ_EVERYWHERE_KEY = "deepy_read_everywhere"
DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY = "deepy_auto_cancel_queue_tasks"
DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY = "deepy_separate_requests_with_empty_line"
DEEPY_TEMPLATE_CONFIG_MIGRATIONS = {
    DEEPY_TOOL_GEN_VIDEO_KEY: {
        "MiniMax H3 FL2VA Turbo Lightx2v 8 Steps": "MiniMax H3 FL2VA Pruned Turbo Lightx2v 8 Steps",
    },
    DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: {
        "MiniMax H3 FL2VA Turbo Lightx2v 8 Steps With Sound": "MiniMax H3 FL2VA Pruned Turbo Lightx2v 8 Steps With Sound",
    },
}

DEEPY_VRAM_MODE_UNLOAD = "unload"
DEEPY_VRAM_MODE_ALWAYS_LOADED = "always_loaded"
DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST = "unload_on_request"
DEEPY_TYPE_ZERO = "zero"
DEEPY_TYPE_PRIME = "prime"
DEEPY_TYPE_DISABLED = "disabled"
DEEPY_TYPE_DEFAULT = DEEPY_TYPE_ZERO
DEEPY_PRIME_GUIDANCE_DEFAULT = "When several models can satisfy the request, prefer the highest-quality base or full model unless the user explicitly prioritizes speed or names another model."
DEEPY_COMPACTION_TYPE_DISCARD = "discard"
DEEPY_COMPACTION_TYPE_SUMMARIZE = "summarize"
DEEPY_DEFAULT_GEN_IMAGE = "Krea 2 Turbo (8 Steps)"
DEEPY_DEFAULT_EDIT_IMAGE = "Flux Klein 9B"
DEEPY_DEFAULT_GEN_VIDEO = "LTX-2 2.5 Distilled"
DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH = "LTX-2.5 Distilled With Sound"
DEEPY_DEFAULT_GEN_SONG = "ACE-Step 1.5 Turbo LM 1.7B"
DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION = "Qwen3 1.7B"
DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE = "Index TTS 2"
DEEPY_CONTEXT_TOKENS_MIN = 8192
DEEPY_CONTEXT_TOKENS_MAX = 256000
DEEPY_CONTEXT_TOKENS_DEFAULT = 16386
DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS = 32000
DEEPY_COMPACTION_TYPE_DEFAULT = DEEPY_COMPACTION_TYPE_DISCARD
DEEPY_KV_CACHE_QUANTIZATION_AUTO = "auto"
DEEPY_KV_CACHE_QUANTIZATION_DEFAULT = DEEPY_KV_CACHE_QUANTIZATION_AUTO
DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION = "1.0.11"
DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT = False
DEEPY_FILE_SYSTEM_ACCESS_DISABLED = "disabled"
DEEPY_FILE_SYSTEM_ACCESS_READ = "read"
DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE = "read_write"
DEEPY_FILE_SYSTEM_ACCESS_DEFAULT = DEEPY_FILE_SYSTEM_ACCESS_DISABLED
DEEPY_FILE_SYSTEM_PATHS_DEFAULT: list[str] = []
DEEPY_READ_EVERYWHERE_DEFAULT = False
DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT = False
DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT = True
DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT = True
DEEPY_CONFIG_FILENAME = "wgp_config.json"

DEEPY_QWEN_ENHANCER_IDS = {3, 4, 5}
_DEEPY_QWEN_VARIANT_LABELS = {3: "Qwen3.5-4B", 4: "Qwen3.5-9B", 5: "Qwen3.8-27B"}
_DEEPY_QWEN_KV_CACHE_SPECS = {
    3: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    4: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    5: {"num_kv_cache_layers": 16, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
}
_DEEPY_DEFAULT_GEN_IMAGE_ALIASES = {"Z_Image_Turbo": "Z Image Turbo"}
_DEEPY_DEFAULT_EDIT_IMAGE_ALIASES = {"Qwen_Edit": DEEPY_DEFAULT_EDIT_IMAGE}
_DEEPY_DEFAULT_GEN_VIDEO_ALIASES = {
    "ltx2_22B_distilled": "LTX-2 2.3 Distilled 1.0",
    "LTX-2 2.3 Distilled": "LTX-2 2.3 Distilled 1.0",
    **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_VIDEO_KEY],
}
_DEEPY_GEN_VIDEO_WITH_SPEECH_ALIASES = {
    "LTX-2.3 Distilled With Sound": "LTX-2.3 Distilled 1.0 With Sound",
    **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY],
}
_DEEPY_RUNTIME_CONFIG: dict[str, Any] | None = None
_DEEPY_RUNTIME_CONFIG_FILENAME = ""


def normalize_deepy_enabled(value: Any) -> int:
    return 1 if int(value or 0) != 0 else 0


def normalize_deepy_type(value: Any) -> str:
    return DEEPY_TYPE_PRIME if str(value or "").strip().lower() == DEEPY_TYPE_PRIME else DEEPY_TYPE_ZERO


def deepy_mode_from_config(enabled: Any, deepy_type: Any) -> str:
    return normalize_deepy_type(deepy_type) if normalize_deepy_enabled(enabled) else DEEPY_TYPE_DISABLED


def split_deepy_mode(value: Any) -> tuple[int, str]:
    text = str(value or "").strip().lower()
    if text == DEEPY_TYPE_PRIME:
        return 1, DEEPY_TYPE_PRIME
    if text == DEEPY_TYPE_ZERO:
        return 1, DEEPY_TYPE_ZERO
    return 0, DEEPY_TYPE_ZERO


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
    return _normalize_deepy_variant(value, _DEEPY_GEN_VIDEO_WITH_SPEECH_ALIASES, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH)


def normalize_deepy_tool_gen_song(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SONG)


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


def normalize_deepy_kv_cache_quantization(value: Any) -> str:
    text = str(DEEPY_KV_CACHE_QUANTIZATION_DEFAULT if value is None else value).strip().lower()
    if text == "int8":
        return "int8"
    if text == DEEPY_KV_CACHE_QUANTIZATION_AUTO:
        return DEEPY_KV_CACHE_QUANTIZATION_AUTO
    return ""


@lru_cache(maxsize=1)
def get_gguf_kernels_version() -> str:
    try:
        return version("llamacpp-gguf-cuda")
    except PackageNotFoundError:
        return ""


def resolve_deepy_kv_cache_quantization(value: Any) -> tuple[str, str]:
    mode = normalize_deepy_kv_cache_quantization(value)
    if mode != DEEPY_KV_CACHE_QUANTIZATION_AUTO:
        return mode, ""
    installed_version = get_gguf_kernels_version()
    try:
        supported = bool(installed_version) and Version(installed_version) >= Version(DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION)
    except InvalidVersion:
        supported = False
    if supported:
        return "int8", f"Fast KV Cache quantization enabled automatically because GGUF kernels {installed_version} were found."
    found = f" {installed_version}" if installed_version else ""
    return "", f"Fast KV Cache quantization disabled automatically because GGUF kernels{found} do not meet the {DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION}+ requirement."


def normalize_deepy_compaction_type(value: Any) -> str:
    return DEEPY_COMPACTION_TYPE_SUMMARIZE if str(value or "").strip().lower() == DEEPY_COMPACTION_TYPE_SUMMARIZE else DEEPY_COMPACTION_TYPE_DISCARD


def validate_deepy_compaction_config(compaction_type: Any, context_tokens: Any) -> str:
    normalized_type = normalize_deepy_compaction_type(compaction_type)
    normalized_tokens = normalize_deepy_context_tokens(context_tokens)
    if normalized_type == DEEPY_COMPACTION_TYPE_SUMMARIZE and normalized_tokens < DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS:
        raise ValueError(f"Deepy Summarize compaction requires at least {DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS:,} context tokens.")
    return normalized_type


def validate_deepy_version_config(deepy_type: Any, compaction_type: Any, context_tokens: Any, enhancer_enabled: Any) -> tuple[str, str, int]:
    normalized_type = normalize_deepy_type(deepy_type)
    normalized_compaction = normalize_deepy_compaction_type(compaction_type)
    normalized_tokens = normalize_deepy_context_tokens(context_tokens)
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except (TypeError, ValueError):
        enhancer_no = 0
    if normalized_type == DEEPY_TYPE_PRIME:
        if enhancer_no != 5:
            raise ValueError("Deepy Prime requires the Qwen3.8 VL 27B model.")
        if normalized_compaction != DEEPY_COMPACTION_TYPE_SUMMARIZE or normalized_tokens < DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS:
            raise ValueError(f"Deepy Prime requires Summarize compaction and at least {DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS:,} context tokens.")
    return normalized_type, normalized_compaction, normalized_tokens


def normalize_deepy_custom_system_prompt(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def normalize_deepy_prime_guidance(value: Any) -> str:
    return normalize_deepy_custom_system_prompt(value) or DEEPY_PRIME_GUIDANCE_DEFAULT


def normalize_deepy_prime_mcp_servers(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Deepy Prime external MCP servers must be a JSON object.")
    normalized = {}
    for raw_name, raw_config in value.items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(raw_config, dict):
            raise ValueError("Each Deepy Prime external MCP server needs a non-empty name and a configuration object.")
        config = dict(raw_config)
        transport = str(config.get("transport", "stdio") or "stdio").strip().lower().replace("_", "-")
        if transport not in {"stdio", "sse", "streamable-http"}:
            raise ValueError(f"Deepy Prime MCP server '{name}' has an unsupported transport: {transport}.")
        if transport == "stdio" and not str(config.get("command", "") or "").strip():
            raise ValueError(f"Deepy Prime MCP stdio server '{name}' requires command.")
        if transport != "stdio" and not str(config.get("url", "") or "").strip():
            raise ValueError(f"Deepy Prime MCP server '{name}' requires url.")
        config["transport"] = transport
        normalized[name] = config
    return normalized


def normalize_deepy_auto_cancel_queue_tasks(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def normalize_deepy_file_system_access(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_DISABLED, "", "0", "false", "off", "no"}:
            return DEEPY_FILE_SYSTEM_ACCESS_DISABLED
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE, "readwrite", "write", "rw", "2"}:
            return DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_READ, "1", "true", "on", "yes"}:
            return DEEPY_FILE_SYSTEM_ACCESS_READ
        return DEEPY_FILE_SYSTEM_ACCESS_DISABLED
    return DEEPY_FILE_SYSTEM_ACCESS_READ if bool(value) else DEEPY_FILE_SYSTEM_ACCESS_DISABLED


def normalize_deepy_allow_read_file_system(value: Any) -> bool:
    return normalize_deepy_file_system_access(value) != DEEPY_FILE_SYSTEM_ACCESS_DISABLED


def normalize_deepy_file_system_paths(value: Any) -> list[str]:
    values = value.splitlines() if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
    paths = []
    seen = set()
    for item in values:
        path = str(item or "").strip()
        key = path.replace("\\", "/").casefold()
        if not path or key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def parse_deepy_file_system_paths(value: Any) -> list[tuple[str, str]]:
    parsed = []
    aliases = set()
    for entry in normalize_deepy_file_system_paths(value):
        if entry[0] in {'"', "'"}:
            quote = entry[0]
            end = entry.find(quote, 1)
            if end < 0:
                raise ValueError(f"Additional filesystem folder has an unterminated quote: {entry}")
            tail = entry[end + 1:]
            if tail and not tail[0].isspace():
                raise ValueError(f"Filesystem alias must be separated from its quoted path by a space: {entry}")
            path, remainder = entry[1:end].strip(), tail.strip()
            if remainder and any(character.isspace() for character in remainder):
                raise ValueError(f"Filesystem alias must be one word: {remainder}")
            alias = remainder
        else:
            parts = entry.rsplit(None, 1)
            path, alias = (parts[0], parts[1]) if len(parts) == 2 else (entry, "")
        if not path:
            raise ValueError("Additional filesystem folder path is empty.")
        if alias:
            if re.fullmatch(r"[A-Za-z0-9_-]+", alias) is None:
                raise ValueError(f"Filesystem alias may contain only letters, numbers, '_' and '-': {alias}")
            key = alias.casefold()
            if re.fullmatch(r"outputs\d*", key):
                raise ValueError(f"Filesystem alias is reserved for WanGP output folders: {alias}")
            if key in aliases:
                raise ValueError(f"Filesystem alias is used more than once: {alias}")
            aliases.add(key)
        parsed.append((path, alias))
    return parsed


def normalize_deepy_read_everywhere(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def normalize_deepy_mcp_auto_discover_paths(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def normalize_deepy_separate_requests_with_empty_line(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def estimate_deepy_kv_cache_mb(enhancer_enabled: Any, context_tokens: Any, kv_cache_quantization: Any = "") -> tuple[str | None, int | None]:
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
    head_dim = int(spec["head_dim"])
    vector_bytes = head_dim + 2 if resolve_deepy_kv_cache_quantization(kv_cache_quantization)[0] == "int8" else head_dim * int(spec["dtype_bytes"])
    total_bytes = 2 * int(spec["num_kv_cache_layers"]) * num_blocks * block_size * int(spec["num_key_value_heads"]) * vector_bytes
    return _DEEPY_QWEN_VARIANT_LABELS.get(enhancer_no, "Qwen3.5"), int(round(total_bytes / (1024 * 1024)))


def format_deepy_context_tokens_label(enhancer_enabled: Any, context_tokens: Any, kv_cache_quantization: Any = "") -> str:
    requested_quantization = normalize_deepy_kv_cache_quantization(kv_cache_quantization)
    quantization = resolve_deepy_kv_cache_quantization(requested_quantization)[0]
    variant_label, cache_mb = estimate_deepy_kv_cache_mb(enhancer_enabled, context_tokens, requested_quantization)
    if variant_label is None or cache_mb is None:
        return "Context Window Tokens (KV cache: N/A)"
    auto_label = " Auto" if requested_quantization == DEEPY_KV_CACHE_QUANTIZATION_AUTO else ""
    return f"Context Window Tokens (KV cache ~{cache_mb} MB {quantization.upper() if quantization else 'BF16'}{auto_label} on {variant_label})"


def normalize_deepy_runtime_config(server_config: dict[str, Any] | None) -> dict[str, Any]:
    runtime_config = dict(server_config or {})
    runtime_config[DEEPY_ENABLED_KEY] = normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0))
    runtime_config[DEEPY_TYPE_KEY] = normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT))
    runtime_config[DEEPY_VRAM_MODE_KEY] = normalize_deepy_vram_mode(runtime_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))
    runtime_config[DEEPY_TOOL_GEN_IMAGE_KEY] = normalize_deepy_tool_gen_image(runtime_config.get(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE))
    runtime_config[DEEPY_TOOL_EDIT_IMAGE_KEY] = normalize_deepy_tool_edit_image(runtime_config.get(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_KEY] = normalize_deepy_tool_gen_video(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY] = normalize_deepy_tool_gen_video_with_speech(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH))
    runtime_config[DEEPY_TOOL_GEN_SONG_KEY] = normalize_deepy_tool_gen_song(runtime_config.get(DEEPY_TOOL_GEN_SONG_KEY, DEEPY_DEFAULT_GEN_SONG))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY] = normalize_deepy_tool_gen_speech_from_description(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY] = normalize_deepy_tool_gen_speech_from_sample(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE))
    runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = normalize_deepy_context_tokens(runtime_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
    runtime_config[DEEPY_KV_CACHE_QUANTIZATION_KEY] = normalize_deepy_kv_cache_quantization(runtime_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))
    runtime_config[DEEPY_COMPACTION_TYPE_KEY] = normalize_deepy_compaction_type(runtime_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT))
    runtime_config[DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_custom_system_prompt(runtime_config.get(DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, ""))
    runtime_config[DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_prime_guidance(runtime_config.get(DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_GUIDANCE_DEFAULT))
    runtime_config[DEEPY_PRIME_MCP_SERVERS_KEY] = normalize_deepy_prime_mcp_servers(runtime_config.get(DEEPY_PRIME_MCP_SERVERS_KEY, {}))
    runtime_config[DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY] = normalize_deepy_mcp_auto_discover_paths(runtime_config.get(DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT))
    runtime_config[DEEPY_ALLOW_READ_FILE_SYSTEM_KEY] = normalize_deepy_file_system_access(runtime_config.get(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_ACCESS_DEFAULT))
    runtime_config[DEEPY_FILE_SYSTEM_PATHS_KEY] = normalize_deepy_file_system_paths(runtime_config.get(DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_FILE_SYSTEM_PATHS_DEFAULT))
    runtime_config[DEEPY_READ_EVERYWHERE_KEY] = normalize_deepy_read_everywhere(runtime_config.get(DEEPY_READ_EVERYWHERE_KEY, DEEPY_READ_EVERYWHERE_DEFAULT))
    runtime_config[DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY] = normalize_deepy_auto_cancel_queue_tasks(runtime_config.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT))
    runtime_config[DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY] = normalize_deepy_separate_requests_with_empty_line(runtime_config.get(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT))
    return runtime_config


def get_deepy_default_runtime_config() -> dict[str, Any]:
    return {
        DEEPY_ENABLED_KEY: 0,
        DEEPY_TYPE_KEY: DEEPY_TYPE_DEFAULT,
        DEEPY_VRAM_MODE_KEY: DEEPY_VRAM_MODE_UNLOAD,
        DEEPY_TOOL_EDIT_IMAGE_KEY: DEEPY_DEFAULT_EDIT_IMAGE,
        DEEPY_TOOL_GEN_IMAGE_KEY: DEEPY_DEFAULT_GEN_IMAGE,
        DEEPY_TOOL_GEN_VIDEO_KEY: DEEPY_DEFAULT_GEN_VIDEO,
        DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH,
        DEEPY_TOOL_GEN_SONG_KEY: DEEPY_DEFAULT_GEN_SONG,
        DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION,
        DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE,
        DEEPY_CONTEXT_TOKENS_KEY: DEEPY_CONTEXT_TOKENS_DEFAULT,
        DEEPY_KV_CACHE_QUANTIZATION_KEY: DEEPY_KV_CACHE_QUANTIZATION_DEFAULT,
        DEEPY_COMPACTION_TYPE_KEY: DEEPY_COMPACTION_TYPE_DEFAULT,
        DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY: "",
        DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY: DEEPY_PRIME_GUIDANCE_DEFAULT,
        DEEPY_PRIME_MCP_SERVERS_KEY: {},
        DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY: DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT,
        DEEPY_ALLOW_READ_FILE_SYSTEM_KEY: DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT,
        DEEPY_FILE_SYSTEM_PATHS_KEY: list(DEEPY_FILE_SYSTEM_PATHS_DEFAULT),
        DEEPY_READ_EVERYWHERE_KEY: DEEPY_READ_EVERYWHERE_DEFAULT,
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
        DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY: DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT,
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
    return len(deepy_requirement_error(server_config)) == 0


def deepy_requirement_error(server_config: dict[str, Any] | None) -> str:
    runtime_config = server_config or {}
    from shared.remote_llm.config import is_remote_engine, local_enhancer_id, resolve_role_engine

    deepy_engine = resolve_role_engine(runtime_config, "deepy")
    if is_remote_engine(deepy_engine):
        if normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT)) != DEEPY_TYPE_PRIME:
            return "External LLM engines require Deepy Prime because only Deepy Prime exposes WanGP's MCP tools."
        return ""
    try:
        enhancer_enabled = local_enhancer_id(deepy_engine, runtime_config.get("enhancer_enabled", 0))
    except Exception:
        enhancer_enabled = 0
    if enhancer_enabled not in DEEPY_QWEN_ENHANCER_IDS:
        return "Deepy requires Prompt Enhancer to be set to a Qwen3.5VL Abliterated or Qwen3.8VL Uncensored mode in the Extensions tab."
    try:
        validate_deepy_version_config(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT), runtime_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT), runtime_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT), enhancer_enabled)
    except ValueError as exc:
        return str(exc)
    return ""


def deepy_enabled(server_config: dict[str, Any] | None) -> bool:
    runtime_config = server_config or {}
    return normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0)) == 1


def deepy_available(server_config: dict[str, Any] | None) -> bool:
    return deepy_enabled(server_config) and deepy_requirement_met(server_config)


def deepy_requirement_message(server_config: dict[str, Any] | None) -> str:
    runtime_config = server_config or {}
    requirement_error = deepy_requirement_error(runtime_config)
    if len(requirement_error) == 0:
        from shared.remote_llm.config import ENGINE_LABELS, is_remote_engine, resolve_role_engine

        version = "Deepy Prime" if normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT)) == DEEPY_TYPE_PRIME else "Deepy Zero"
        engine = resolve_role_engine(runtime_config, "deepy")
        detail = f" using {ENGINE_LABELS.get(engine, engine.title())} externally" if is_remote_engine(engine) else " with the current Prompt Enhancer and context configuration"
        return (
            "<div style='color:#1b6d44; font-weight:600;'>"
            f"{version} is available{detail}."
            "</div>"
        )
    return (
        "<div style='color:#8a4a14; font-weight:600;'>"
        f"{requirement_error}"
        "</div>"
    )
