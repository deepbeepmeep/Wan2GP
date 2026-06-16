import json
from decimal import Decimal, InvalidOperation

from postprocessing.mmaudio import MMAUDIO_DEFAULT_MODE
from postprocessing.mmaudio.audio_processor import MMAudioProcessor
from postprocessing.rife.temporal_upsampler import RifeTemporalUpsampler
from postprocessing.seedvc.audio_processor import SeedVCProcessor
from shared.deepy.config import DEEPY_ENABLED_KEY


LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY = "_extensions_defaults_migrated"
EXTENSIONS_DEFAULTS_VERSION_KEY = "extensions_defaults_version"
EXTENSIONS_DEFAULTS_TARGET_VERSION = Decimal("1.17")
EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT = str(EXTENSIONS_DEFAULTS_TARGET_VERSION)
INSTALLED_REMOTE_PLUGINS_KEY = "installed_remote_plugins"

SEEDVC_MODE_CHOICES = [("v1.0 Speech", 1), ("v1.0 Singing / F0 44k", 2), ("v2 Speech", 3)]
PROMPT_ENHANCER_CHOICES = [
    ("Florence 2 (image captioning) + LLama 3.2 3B (text generation)", 1),
    ("Florence 2 (image captioning) + Llama Joy 8B (uncensored, richer)", 2),
    ("Qwen3.5VL Abliterated 4B (recommended, captioning + uncensored text enhancement, vllm accelerated if available)", 3),
    ("Qwen3.5VL Abliterated 9B (captioning + uncensored high end text enhancement, vllm accelerated if available)", 4),
]

SEEDVC_DEFAULT_MODE = 2
PROMPT_ENHANCER_LOW_VRAM_DEFAULT_MODE = PROMPT_ENHANCER_CHOICES[0][1]
PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE = 3
PROMPT_ENHANCER_QWEN_MIN_VRAM_GB = 10
DEEPY_DEFAULT_ENABLED = 1
MEDIAFLOW_PLUGIN_ID = "media_flow"
PLUGIN_ID_MIGRATIONS = {
    "wan2gp-about": "about",
    "wan2gp-configuration": "configuration",
    "wan2gp-downloads": "downloads",
    "wan2gp-guides": "guides",
    "wan2gp-media-flow": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-mediaflow": MEDIAFLOW_PLUGIN_ID,
    "media-flow": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-models-manager": "models_manager",
    "models-manager": "models_manager",
    "wan2gp-motion-designer": "motion_designer",
    "motion-designer": "motion_designer",
    "wan2gp-plugin-manager": "plugin_manager",
    "plugin-manager": "plugin_manager",
    "wan2gp-process-full-video": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-sample": "sample",
    "wan2gp-video-mask-creator": "video_mask_creator",
    "video-mask-creator": "video_mask_creator",
}
PROTECTED_PLUGIN_IDS = {
    "about",
    "configuration",
    "downloads",
    "guides",
    "media_flow",
    "models_manager",
    "motion_designer",
    "plugin_manager",
    "sample",
    "video_mask_creator",
}


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


def _migrate_audio_processors_config(server_config, version: Decimal) -> bool:
    from postprocessing import audio_processors as audio_processor_api

    changed = False
    sections = server_config.get(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY, {})
    if not isinstance(sections, dict):
        sections = {}
        server_config[audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY] = sections
        changed = True
    else:
        server_config.setdefault(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY, sections)
    legacy_keys = ("mmaudio_mode", "mmaudio_persistence", "mmaudio_enabled", "seedvc_mode", "seedvc_persistence")
    if any(key in server_config for key in legacy_keys):
        mmaudio_mode = server_config.get("mmaudio_mode", None)
        if mmaudio_mode is None:
            mmaudio_mode = 0 if _is_off(server_config.get("mmaudio_enabled", 0)) else MMAUDIO_DEFAULT_MODE
        if version < Decimal("1.1") and _is_off(mmaudio_mode):
            mmaudio_mode = MMAUDIO_DEFAULT_MODE
        mmaudio_persistence = server_config.get("mmaudio_persistence", None)
        if mmaudio_persistence is None:
            mmaudio_persistence = 2 if _to_int(server_config.get("mmaudio_enabled", 0), 0) == 2 else 1
        sections["mmaudio"] = MMAudioProcessor.normalize_config_section({"mode": mmaudio_mode, "persistence": mmaudio_persistence})

        seedvc_mode = server_config.get("seedvc_mode", SEEDVC_DEFAULT_MODE if version < Decimal("1.11") else 0)
        if version < Decimal("1.11") and _is_off(seedvc_mode):
            seedvc_mode = SEEDVC_DEFAULT_MODE
        sections["seedvc"] = SeedVCProcessor.normalize_config_section({"mode": seedvc_mode, "persistence": server_config.get("seedvc_persistence", 1)})
        for key in legacy_keys:
            if key in server_config:
                del server_config[key]
                changed = True
        changed = True
    changed = audio_processor_api.migrate_audio_processor_config(server_config) or changed
    return changed


def _migrate_temporal_upsamplers_config(server_config) -> bool:
    from postprocessing import temporal_upsamplers as temporal_upsampler_api

    changed = False
    sections = server_config.get(temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(sections, dict):
        sections = {}
        server_config[temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY] = sections
        changed = True
    else:
        server_config.setdefault(temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY, sections)
    if "rife_version" in server_config:
        sections["rife"] = RifeTemporalUpsampler.normalize_config_section({"version": server_config["rife_version"]})
        del server_config["rife_version"]
        changed = True
    changed = temporal_upsampler_api.migrate_temporal_upsampler_config(server_config) or changed
    return changed


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

    if version < Decimal("1.13"):
        from postprocessing import spatial_upsamplers as upsampler_api

        changed = upsampler_api.migrate_upsampler_config(server_config, prefer_legacy=True, apply_pre_1_1_defaults=version < Decimal("1.1")) or changed

    changed = _migrate_audio_processors_config(server_config, version) or changed
    changed = _migrate_temporal_upsamplers_config(server_config) or changed

    if server_config.pop(LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY, None) is not None:
        changed = True
    if server_config.get(EXTENSIONS_DEFAULTS_VERSION_KEY) != EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT:
        server_config[EXTENSIONS_DEFAULTS_VERSION_KEY] = EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT
        changed = True
    changed = migrate_bundled_plugin_ids(server_config) or changed
    changed = migrate_installed_remote_plugins(server_config) or changed

    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def _migrate_plugin_id_list(server_config, key) -> bool:
    plugin_ids = server_config.get(key, [])
    if not isinstance(plugin_ids, list):
        return False
    changed = False
    migrated = []
    seen = set()
    for plugin_id in plugin_ids:
        plugin_id = str(plugin_id or "").strip()
        plugin_id = PLUGIN_ID_MIGRATIONS.get(plugin_id, plugin_id)
        if not plugin_id or plugin_id in seen:
            changed = True
            continue
        seen.add(plugin_id)
        migrated.append(plugin_id)
    if migrated != plugin_ids:
        server_config[key] = migrated
        changed = True
    return changed


def migrate_bundled_plugin_ids(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False
    changed = _migrate_plugin_id_list(server_config, "enabled_plugins")
    changed = _migrate_plugin_id_list(server_config, "pending_plugin_deletions") or changed
    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def migrate_mediaflow_plugin_id(server_config, server_config_filename="") -> bool:
    return migrate_bundled_plugin_ids(server_config, server_config_filename)


def migrate_installed_remote_plugins(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False
    changed = _migrate_plugin_id_list(server_config, INSTALLED_REMOTE_PLUGINS_KEY)
    if INSTALLED_REMOTE_PLUGINS_KEY not in server_config:
        server_config[INSTALLED_REMOTE_PLUGINS_KEY] = [plugin_id for plugin_id in server_config.get("enabled_plugins", []) if plugin_id not in PROTECTED_PLUGIN_IDS]
        changed = True
    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def enabled_choice_value(value, choices, default):
    allowed = {choice_value for _, choice_value in choices}
    value = _to_int(value, default)
    return value if value in allowed else default
