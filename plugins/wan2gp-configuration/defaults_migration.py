import json

from shared.deepy.config import DEEPY_ENABLED_KEY


EXTENSIONS_DEFAULTS_MIGRATED_KEY = "_extensions_defaults_migrated"

MMAUDIO_MODE_CHOICES = [("Standard", 1), ("NSFW", 2)]
SEEDVC_MODE_CHOICES = [("v1.0", 1)]
FLASHVSR_MODE_CHOICES = [
    ("FlashVSR v1.1 Tiny (Slightly Lower Quality, Faster VAE Decoding, Needs Less RAM)", 1),
    ("FlashVSR v1.1 Full (Best Quality, Slower VAE Decoding, Needs More RAM)", 2),
]
PROMPT_ENHANCER_CHOICES = [
    ("Florence 2 (image captioning) + LLama 3.2 3B (text generation)", 1),
    ("Florence 2 (image captioning) + Llama Joy 8B (uncensored, richer)", 2),
    ("Qwen3.5VL Abliterated 4B (captioning + uncensored text enhancement, vllm accelerated if available)", 3),
    ("Qwen3.5VL Abliterated 9B (captioning + uncensored high end text enhancement, vllm accelerated if available)", 4),
]

MMAUDIO_DEFAULT_MODE = MMAUDIO_MODE_CHOICES[0][1]
SEEDVC_DEFAULT_MODE = SEEDVC_MODE_CHOICES[0][1]
FLASHVSR_DEFAULT_MODE = FLASHVSR_MODE_CHOICES[0][1]
PROMPT_ENHANCER_DEFAULT_MODE = 3
DEEPY_DEFAULT_ENABLED = 1


def _to_int(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _is_off(value) -> bool:
    return _to_int(value, 0) == 0


def _write_config(config, config_filename):
    if not config_filename:
        return
    with open(config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(config, indent=4))


def _set_missing_persistence(config, key):
    if key not in config or _to_int(config.get(key), 1) not in (1, 2):
        config[key] = 1


def migrate_extension_defaults(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict) or server_config.get(EXTENSIONS_DEFAULTS_MIGRATED_KEY, False):
        return False

    changed = False

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

    if _is_off(server_config.get("enhancer_enabled", 0)):
        server_config["enhancer_enabled"] = PROMPT_ENHANCER_DEFAULT_MODE
        changed = True

    if _is_off(server_config.get(DEEPY_ENABLED_KEY, 0)):
        server_config[DEEPY_ENABLED_KEY] = DEEPY_DEFAULT_ENABLED
        changed = True

    server_config[EXTENSIONS_DEFAULTS_MIGRATED_KEY] = True
    _write_config(server_config, server_config_filename)
    return changed


def enabled_choice_value(value, choices, default):
    allowed = {choice_value for _, choice_value in choices}
    value = _to_int(value, default)
    return value if value in allowed else default
