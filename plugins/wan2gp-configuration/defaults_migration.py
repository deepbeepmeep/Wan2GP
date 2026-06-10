import json
from decimal import Decimal, InvalidOperation

from shared.deepy.config import DEEPY_ENABLED_KEY


LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY = "_extensions_defaults_migrated"
EXTENSIONS_DEFAULTS_VERSION_KEY = "extensions_defaults_version"
EXTENSIONS_DEFAULTS_TARGET_VERSION = Decimal("1.12")
EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT = str(EXTENSIONS_DEFAULTS_TARGET_VERSION)

MMAUDIO_MODE_CHOICES = [("Standard", 1), ("NSFW", 2)]
SEEDVC_MODE_CHOICES = [("v1.0 Speech", 1), ("v1.0 Singing / F0 44k", 2), ("v2 Speech", 3)]
FLASHVSR_MODE_CHOICES = [
    ("FlashVSR v1.1 Tiny (Slightly Lower Quality, Faster VAE Decoding, Needs Less RAM)", 1),
    ("FlashVSR v1.1 Full (Best Quality, Slower VAE Decoding, Needs More RAM)", 2),
]
PROMPT_ENHANCER_CHOICES = [
    ("Florence 2 (image captioning) + LLama 3.2 3B (text generation)", 1),
    ("Florence 2 (image captioning) + Llama Joy 8B (uncensored, richer)", 2),
    ("Qwen3.5VL Abliterated 4B (recommended, captioning + uncensored text enhancement, vllm accelerated if available)", 3),
    ("Qwen3.5VL Abliterated 9B (captioning + uncensored high end text enhancement, vllm accelerated if available)", 4),
]

MMAUDIO_DEFAULT_MODE = MMAUDIO_MODE_CHOICES[0][1]
SEEDVC_DEFAULT_MODE = 2
FLASHVSR_DEFAULT_MODE = FLASHVSR_MODE_CHOICES[0][1]
PROMPT_ENHANCER_LOW_VRAM_DEFAULT_MODE = PROMPT_ENHANCER_CHOICES[0][1]
PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE = 3
PROMPT_ENHANCER_DEFAULT_MODE = PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE
PROMPT_ENHANCER_QWEN_MIN_VRAM_GB = 10
DEEPY_DEFAULT_ENABLED = 1
LEGACY_MEDIAFLOW_PLUGIN_IDS = {"wan2gp-process-full-video", "wan2gp-mediaflow"}
MEDIAFLOW_PLUGIN_ID = "wan2gp-media-flow"


def _to_int(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _is_off(value) -> bool:
    return _to_int(value, 0) == 0


def _cuda_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def get_prompt_enhancer_default_mode() -> int:
    return PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE if _cuda_vram_gb() >= PROMPT_ENHANCER_QWEN_MIN_VRAM_GB else PROMPT_ENHANCER_LOW_VRAM_DEFAULT_MODE


def _write_config(config, config_filename):
    if not config_filename:
        return
    with open(config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(config, indent=4))


def _set_missing_persistence(config, key):
    if key not in config or _to_int(config.get(key), 1) not in (1, 2):
        config[key] = 1


def _extension_defaults_version(config) -> Decimal:
    version = config.get(EXTENSIONS_DEFAULTS_VERSION_KEY, None)
    if version is not None:
        try:
            return Decimal(str(version))
        except (InvalidOperation, ValueError):
            return Decimal("1.0")
    return Decimal("1.1") if config.get(LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY, False) else Decimal("1.0")


def migrate_extension_defaults(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False

    version = _extension_defaults_version(server_config)
    prompt_enhancer_default_mode = get_prompt_enhancer_default_mode()
    changed = False

    if version < Decimal("1.1"):
        mmaudio_mode = server_config.get("mmaudio_mode", None)
        if mmaudio_mode is None:
            mmaudio_mode = 0 if _is_off(server_config.get("mmaudio_enabled", 0)) else MMAUDIO_DEFAULT_MODE
        if _is_off(mmaudio_mode):
            server_config["mmaudio_mode"] = MMAUDIO_DEFAULT_MODE
            changed = True
        else:
            server_config["mmaudio_mode"] = _to_int(mmaudio_mode, MMAUDIO_DEFAULT_MODE)
        _set_missing_persistence(server_config, "mmaudio_persistence")
        server_config["mmaudio_enabled"] = server_config["mmaudio_persistence"]

        if _is_off(server_config.get("seedvc_mode", 0)):
            server_config["seedvc_mode"] = SEEDVC_DEFAULT_MODE
            changed = True
        _set_missing_persistence(server_config, "seedvc_persistence")

        if _is_off(server_config.get("flashvsr_mode", 0)):
            server_config["flashvsr_mode"] = FLASHVSR_DEFAULT_MODE
            changed = True
        _set_missing_persistence(server_config, "flashvsr_persistence")
        _set_missing_persistence(server_config, "pid_persistence")

        if _is_off(server_config.get("enhancer_enabled", 0)):
            server_config["enhancer_enabled"] = prompt_enhancer_default_mode
            changed = True

        if _is_off(server_config.get(DEEPY_ENABLED_KEY, 0)):
            server_config[DEEPY_ENABLED_KEY] = DEEPY_DEFAULT_ENABLED
            changed = True

    if version < Decimal("1.11"):
        if server_config.get("seedvc_mode") != SEEDVC_DEFAULT_MODE:
            server_config["seedvc_mode"] = SEEDVC_DEFAULT_MODE
            changed = True
        _set_missing_persistence(server_config, "seedvc_persistence")
        _set_missing_persistence(server_config, "pid_persistence")

    if server_config.pop(LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY, None) is not None:
        changed = True
    if server_config.get(EXTENSIONS_DEFAULTS_VERSION_KEY) != EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT:
        server_config[EXTENSIONS_DEFAULTS_VERSION_KEY] = EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT
        changed = True
    changed = migrate_mediaflow_plugin_id(server_config) or changed

    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def migrate_mediaflow_plugin_id(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False
    enabled_plugins = server_config.get("enabled_plugins", [])
    if not isinstance(enabled_plugins, list):
        return False
    changed = False
    migrated = []
    seen = set()
    for plugin_id in enabled_plugins:
        plugin_id = str(plugin_id or "").strip()
        if plugin_id in LEGACY_MEDIAFLOW_PLUGIN_IDS:
            plugin_id = MEDIAFLOW_PLUGIN_ID
            changed = True
        if not plugin_id or plugin_id in seen:
            changed = True
            continue
        seen.add(plugin_id)
        migrated.append(plugin_id)
    if changed:
        server_config["enabled_plugins"] = migrated
        _write_config(server_config, server_config_filename)
    return changed


def enabled_choice_value(value, choices, default):
    allowed = {choice_value for _, choice_value in choices}
    value = _to_int(value, default)
    return value if value in allowed else default
