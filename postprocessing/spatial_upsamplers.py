"""Spatial upsampler plugin API.

Every spatial upsampler (built-in or extension) class is listed in
``spatial_upsampler_handlers`` and declares its capabilities through
``query_upsampler_def()``:

```python
{
    "name": "FlashVSR",                      # display name
    "upsampler_types": ("postprocessing",),  # "postprocessing" and/or "vae"
    "media": ("video", "image"),             # media kinds the upsampler can process
    "profile": "video",                      # memory profile kind: video, image, or audio
    "config_key": "flashvsr",                # optional subkey under wgp_config["spatial_upsamplers"]
    "pos": 20,                               # default dropdown order for this handler's methods
    "method_pos": {"flashvsr": 20},          # optional per-method order; independent of multiplier
    "methods": [("FlashVSR", "flashvsr")],   # interchangeable post-processing methods (label, method key)
    "vae_methods": [],                       # VAE methods (label, method key); model-pipeline integration
    "multipliers": {"flashvsr": (2.0, 4.0)}, # supported upsampling multipliers per method key
    "default_spatial_upsampling": "flashvsr2",
}
```

Handlers must also implement:
- ``is_upsampling(value)``: does this handler own this ``spatial_upsampling`` value?
- ``split_value(value)`` -> ``(method, scale)`` or ``None``
- ``build_value(method, scale)`` -> ``spatial_upsampling`` value or ``None``
- ``validate_upsampling(value, image_mode)`` -> error text ("" when valid)

``wgp.py`` calls ``register_spatial_upsamplers(...)`` once and then accesses
handlers through this API instead of keeping per-upsampler globals.

Post-processing ("postprocessing" type) handlers are interchangeable and must
additionally implement ``upscale(sample, value, **kwargs)`` and may implement
``load_upsampler(value, **kwargs)``, ``download(...)``, ``enabled()`` and
``release_vram()``. They are automatically offered for late post-processing of
existing media. VAE ("vae") handlers are plugged into model pipelines through
the generic VAE upsampler hooks below; model defs declare support.

Handlers may also expose Config-tab controls with ``create_config_ui(...)`` and
normalize their own nested section under ``wgp_config["spatial_upsamplers"]``.

Upsamplers that allocate their own mmgp offload object must register it in
``shared.utils.offload_registry`` so its resources can be tracked and released
centrally (WanGP unload tool).
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any

from shared.attention import attention_shared_state
from shared.utils import offload_registry

# Backward compatibility for external plugins written against the old module name.
sys.modules.setdefault("postprocessing.upsamplers", sys.modules[__name__])

UPSAMPLER_TYPE_POSTPROCESSING = "postprocessing"
UPSAMPLER_TYPE_VAE = "vae"
UPSAMPLER_PROFILE_VIDEO = "video"
UPSAMPLER_PROFILE_IMAGE = "image"
UPSAMPLER_PROFILE_AUDIO = "audio"
UPSAMPLER_CONFIG_KEY = "spatial_upsamplers"

spatial_upsampler_handlers = [
    "postprocessing.lanczos.wgp_bridge.LanczosUpsampler",
    "postprocessing.flashvsr.wgp_bridge.FlashVSRBridge",
    "postprocessing.pid.wgp_bridge.PiDBridge",
    "postprocessing.chain_of_zoom.wgp_bridge.ChainOfZoomBridge",
    "postprocessing.spatial_upsamplers.WanVaeUpsampler",
]
_upsampler_handlers: list[Any] = []
_registered_upsampler_handler_paths: set[str] = set()


@dataclass
class UpsamplerConfigBinding:
    handler: Any
    config_key: str
    controls: list[tuple[str, Any]]


def format_multiplier(scale: float) -> str:
    scale = float(scale)
    return str(int(scale)) if scale.is_integer() else f"{scale:g}"


def format_multiplier_label(scale: float) -> str:
    return f"x{format_multiplier(scale)}"


def format_method_label(label: str) -> str:
    return str(label or "").removesuffix(" Upsampler")


def format_method_scale_label(label: str, scale: float) -> str:
    return f"{format_method_label(label)} {format_multiplier_label(scale)}"


def register_upsampler(handler) -> None:
    if handler not in _upsampler_handlers:
        _upsampler_handlers.append(handler)


def _load_upsampler_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _config_key_from_handler_def(handler_def: dict[str, Any], fallback_name: str) -> str:
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return fallback_name.lower()


def default_config_sections(handler_modules: list[str] | None = None) -> dict[str, dict[str, Any]]:
    sections = {}
    for path in spatial_upsampler_handlers if handler_modules is None else handler_modules:
        handler_cls = _load_upsampler_class(str(path or "").strip())
        if not hasattr(handler_cls, "default_config"):
            continue
        config = dict(handler_cls.default_config())
        if not config:
            continue
        handler_def = handler_cls.query_upsampler_def()
        sections[_config_key_from_handler_def(handler_def, handler_cls.__name__)] = config
    return sections


def register_spatial_upsamplers(server_config, files_locator, handler_modules: list[str] | None = None) -> None:
    modules = spatial_upsampler_handlers if handler_modules is None else handler_modules
    for path in modules:
        path = str(path or "").strip()
        if not path or path in _registered_upsampler_handler_paths:
            continue
        register_upsampler(_load_upsampler_class(path)(server_config, files_locator))
        _registered_upsampler_handler_paths.add(path)


def upsampler_handlers(upsampler_type: str | None = None, enabled_only: bool = False) -> list[Any]:
    handlers = []
    for handler in _upsampler_handlers:
        if upsampler_type is not None and upsampler_type not in handler.query_upsampler_def().get("upsampler_types", ()):
            continue
        if enabled_only and not handler_enabled(handler):
            continue
        handlers.append(handler)
    return handlers


def handler_enabled(handler) -> bool:
    return not hasattr(handler, "enabled") or handler.enabled()


def query_upsampler_defs(upsampler_type: str | None = None, enabled_only: bool = False) -> list[dict[str, Any]]:
    return [handler.query_upsampler_def() for handler in upsampler_handlers(upsampler_type, enabled_only)]


def _method_choices(handler_def: dict[str, Any]) -> list[tuple[str, str]]:
    return handler_def.get("methods", []) + handler_def.get("vae_methods", [])


def _method_labels(handler_def: dict[str, Any]) -> dict[str, str]:
    return {key: label for label, key in _method_choices(handler_def)}


def _handler_method_label(handler, label: str, method: str) -> str:
    return handler.format_method_label(label, method) if hasattr(handler, "format_method_label") else label


def _handler_method_available(handler, method: str) -> bool:
    return not hasattr(handler, "method_available") or handler.method_available(method)


def _handler_pos(handler_def: dict[str, Any]) -> float:
    try:
        return float(handler_def.get("pos", 1000))
    except (TypeError, ValueError):
        return 1000


def _method_pos(handler_def: dict[str, Any], method: str) -> float:
    method_pos = handler_def.get("method_pos", {})
    if isinstance(method_pos, dict) and method in method_pos:
        try:
            return float(method_pos[method])
        except (TypeError, ValueError):
            pass
    return _handler_pos(handler_def)


def find_upsampler(spatial_upsampling) -> Any | None:
    if not str(spatial_upsampling or "").strip():
        return None
    return next((handler for handler in _upsampler_handlers if handler.is_upsampling(spatial_upsampling)), None)


def find_postprocessing_upsampler(spatial_upsampling) -> Any | None:
    handler = find_upsampler(spatial_upsampling)
    if handler is None:
        return None
    method = handler.split_value(spatial_upsampling)[0]
    return handler if method in [key for _, key in handler.query_upsampler_def().get("methods", [])] else None


def find_vae_upsampler(spatial_upsampling) -> Any | None:
    handler = find_upsampler(spatial_upsampling)
    if handler is None:
        return None
    method = handler.split_value(spatial_upsampling)[0]
    return handler if method in [key for _, key in handler.query_upsampler_def().get("vae_methods", [])] else None


def is_vae_upsampling(spatial_upsampling) -> bool:
    return find_vae_upsampler(spatial_upsampling) is not None


def upscale_postprocessing(handler, sample, spatial_upsampling, *, main_offloadobj=None, **kwargs):
    persistent = handler.persistent_models() if hasattr(handler, "persistent_models") else False
    name = handler.query_upsampler_def()["name"]
    with attention_shared_state():
        try:
            if main_offloadobj is not None:
                main_offloadobj.unload_all()
            if hasattr(handler, "load_upsampler"):
                handler.load_upsampler(spatial_upsampling, **kwargs)
            return handler.upscale(sample, spatial_upsampling, **kwargs)
        finally:
            if persistent:
                offload_registry.unload_vram([name])
            else:
                offload_registry.release_all([name])


def validate_postprocessing_spatial_upsampling(spatial_upsampling, image_mode: int) -> str:
    if is_vae_upsampling(spatial_upsampling):
        return "VAE Spatial Upsampling is only available during generation"
    edit_upsampler = find_postprocessing_upsampler(spatial_upsampling)
    if len(spatial_upsampling) > 0 and edit_upsampler is None:
        return f"No spatial upsampler registered for '{spatial_upsampling}'"
    if edit_upsampler is not None:
        error = edit_upsampler.validate_upsampling(spatial_upsampling, image_mode)
        if error:
            return error
    return ""


def _handler_supports_model_vae_method(handler, method: str, model_type, model_def, image_mode: int) -> bool:
    return hasattr(handler, "supports_model_vae_method") and handler.supports_model_vae_method(method, model_type, model_def, image_mode)


def query_model_vae_method_choices(model_type, model_def, image_mode: int) -> list[tuple[str, str]]:
    choices = []
    for handler in upsampler_handlers(UPSAMPLER_TYPE_VAE):
        handler_def = handler.query_upsampler_def()
        for label, method in handler_def.get("vae_methods", []):
            if _handler_supports_model_vae_method(handler, method, model_type, model_def, image_mode):
                choices.append((_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), _handler_method_label(handler, label, method), method))
    return [(label, method) for _, _, _, label, method in sorted(choices)]


def validate_model_vae_upsampling(spatial_upsampling, image_mode: int, model_type, model_def, medium: str) -> str:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None:
        return ""
    if hasattr(handler, "validate_model_vae_upsampling"):
        return handler.validate_model_vae_upsampling(spatial_upsampling, image_mode, model_type, model_def, medium)
    method = handler.split_value(spatial_upsampling)[0]
    return "" if _handler_supports_model_vae_method(handler, method, model_type, model_def, image_mode) else f"{format_upsampling_label(spatial_upsampling)} is not available for {medium}"


def model_load_vae_upsampling_value(spatial_upsampling, model_type, model_def, image_mode: int) -> str | None:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "model_load_upsampling_value"):
        return None
    return handler.model_load_upsampling_value(spatial_upsampling, model_type, model_def, image_mode)


def loaded_model_vae_upsampling_value(model) -> str | None:
    for handler in upsampler_handlers(UPSAMPLER_TYPE_VAE):
        if hasattr(handler, "loaded_model_vae_upsampling_value"):
            value = handler.loaded_model_vae_upsampling_value(model)
            if value is not None:
                return value
    return None


def model_load_kwargs_for_vae_upsampling(spatial_upsampling, model_type, model_def, image_mode: int) -> dict[str, Any]:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "model_load_kwargs_for_vae_upsampling"):
        return {}
    return handler.model_load_kwargs_for_vae_upsampling(spatial_upsampling, model_type, model_def, image_mode)


def post_model_process_vae_upsampling(sample, spatial_upsampling):
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "post_model_process_vae_upsampling"):
        return sample
    return handler.post_model_process_vae_upsampling(sample, spatial_upsampling)


def has_post_model_process_vae_upsampling(spatial_upsampling) -> bool:
    handler = find_vae_upsampler(spatial_upsampling)
    return handler is not None and hasattr(handler, "post_model_process_vae_upsampling")


def prepare_vae_upsampler(handler, spatial_upsampling, **kwargs):
    if handler is None or not hasattr(handler, "prepare_vae_upsampler"):
        return None
    return handler.prepare_vae_upsampler(spatial_upsampling, **kwargs)


def release_vae_upsampler(handler, session) -> None:
    if handler is not None and session is not None and hasattr(handler, "release_vram") and not (hasattr(handler, "persistent_models") and handler.persistent_models()):
        handler.release_vram()


def find_upsampler_by_method(method) -> Any | None:
    method = str(method or "").strip()
    if not method:
        return None
    for handler in _upsampler_handlers:
        handler_def = handler.query_upsampler_def()
        if method in [key for _, key in _method_choices(handler_def)]:
            return handler
    return None


def require_upsampler_by_method(method) -> Any:
    handler = find_upsampler_by_method(method)
    if handler is None:
        raise RuntimeError(f"No spatial upsampler registered for method '{method}'")
    return handler


def method_multipliers(method) -> tuple[float, ...]:
    handler = find_upsampler_by_method(method)
    if handler is None:
        return ()
    return tuple(handler.query_upsampler_def().get("multipliers", {}).get(str(method), ()))


def ratio_choices_for_method(method) -> list[tuple[str, float]]:
    return [(format_multiplier_label(scale), scale) for scale in method_multipliers(method)]


def _default_multiplier_from_def(handler_def: dict[str, Any], method: str) -> float | None:
    multipliers = tuple(handler_def.get("multipliers", {}).get(method, ()))
    if not multipliers:
        return None
    default_value = handler_def.get("default_spatial_upsampling", "")
    handler = find_upsampler(default_value)
    if handler is not None:
        split = handler.split_value(default_value)
        if split is not None and split[0] == method and split[1] in multipliers:
            return split[1]
    return multipliers[0]


def default_multiplier_for_method(method) -> float:
    handler = find_upsampler_by_method(method)
    if handler is None:
        return 2.0
    return _default_multiplier_from_def(handler.query_upsampler_def(), str(method or "").strip()) or 2.0


def normalize_multiplier_for_method(method, scale) -> float:
    multipliers = method_multipliers(method)
    if not multipliers:
        return default_multiplier_for_method("lanczos")
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = default_multiplier_for_method(method)
    return scale if scale in multipliers else default_multiplier_for_method(method)


def split_upsampling_value(value) -> tuple[str, float] | None:
    handler = find_upsampler(value)
    return None if handler is None else handler.split_value(value)


def build_upsampling_value(method, scale) -> str | None:
    handler = find_upsampler_by_method(method)
    return None if handler is None else handler.build_value(method, scale)


def format_upsampling_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    handler = find_upsampler(text)
    if handler is None:
        return text
    split = handler.split_value(text)
    if split is None:
        return text
    method, scale = split
    label = _method_labels(handler.query_upsampler_def()).get(method)
    if label:
        label = _handler_method_label(handler, label, method)
    return format_method_scale_label(label, scale) if label else text


def normalize_upsampling_state(method, scale) -> tuple[list[tuple[str, float]], float, str]:
    method = str(method or "").strip()
    ratio_choices = ratio_choices_for_method(method) or ratio_choices_for_method("lanczos")
    scale = normalize_multiplier_for_method(method or "lanczos", scale)
    return ratio_choices, scale, "" if not method else build_upsampling_value(method, scale) or ""


def normalize_upsampling_value_for_method(method, current_value) -> tuple[list[tuple[str, float]], float, str]:
    split = split_upsampling_value(current_value)
    return normalize_upsampling_state(method, 2.0 if split is None else split[1])


def _method_choice_sort_key(choice: tuple[str, str]) -> tuple[float, str, str]:
    label, method = choice
    handler = find_upsampler_by_method(method)
    position = 1000 if handler is None else _method_pos(handler.query_upsampler_def(), method)
    return position, str(label or "").casefold(), str(method or "")


def query_postprocessing_method_choices(image_outputs: bool = False, late_postprocessing: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    video_post_choices, image_post_choices = [], []
    for handler in upsampler_handlers(UPSAMPLER_TYPE_POSTPROCESSING):
        handler_def = handler.query_upsampler_def()
        method_keys = [key for _, key in handler_def.get("methods", [])]
        if "lanczos" not in method_keys and not (late_postprocessing or handler_enabled(handler)):
            continue
        media = handler_def.get("media", (UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_PROFILE_IMAGE))
        if (UPSAMPLER_PROFILE_IMAGE if image_outputs else UPSAMPLER_PROFILE_VIDEO) not in media:
            continue
        choices = [(_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), _handler_method_label(handler, label, method), method) for label, method in handler_def.get("methods", []) if _handler_method_available(handler, method)]
        if UPSAMPLER_PROFILE_VIDEO in media:
            video_post_choices += choices
        else:
            image_post_choices += choices
    return [(label, method) for _, _, _, label, method in sorted(video_post_choices)], [(label, method) for _, _, _, label, method in sorted(image_post_choices)]


def dropdown_state(spatial_upsampling, *, image_outputs: bool = False, late_postprocessing: bool = False, vae_choices: list[tuple[str, str]] | None = None, excluded_methods: set[str] | None = None, exclude_method_fn=None) -> dict[str, Any]:
    method, scale = split_upsampling_value(spatial_upsampling) or ("", 2.0)
    video_post_choices, image_post_choices = query_postprocessing_method_choices(image_outputs=image_outputs, late_postprocessing=late_postprocessing)
    excluded_methods = excluded_methods or set()
    image_post_choices = [choice for choice in image_post_choices if choice[1] not in excluded_methods and not (exclude_method_fn and exclude_method_fn(choice[1]))]
    method_choices = [("None", "")] + sorted(video_post_choices + image_post_choices + list(vae_choices or []), key=_method_choice_sort_key)
    if method not in {value for _, value in method_choices}:
        method = ""
    ratio_choices, scale, value = normalize_upsampling_state(method, scale)
    return {"method": method, "scale": scale, "value": value, "method_choices": method_choices, "ratio_choices": ratio_choices}


def create_generation_spatial_ui(gr, spatial_upsampling, *, image_outputs: bool = False, late_postprocessing: bool = False, vae_choices: list[tuple[str, str]] | None = None, excluded_methods: set[str] | None = None, exclude_method_fn=None, elem_classes=None, update_form: bool = False, field_help=None) -> dict[str, Any]:
    method, scale = split_upsampling_value(spatial_upsampling) or ("", 2.0)
    state = dropdown_state(build_upsampling_value(method, scale) or "", image_outputs=image_outputs, late_postprocessing=late_postprocessing, vae_choices=vae_choices, excluded_methods=excluded_methods, exclude_method_fn=exclude_method_fn)
    with gr.Row():
        method_component = gr.Dropdown(choices=state["method_choices"], value=state["method"], visible=True, scale=3, label="Spatial Upsampling", elem_classes=elem_classes)
        if field_help is not None:
            field_help.bind(method_component, "spatial_upsampling")
        ratio_component = gr.Dropdown(choices=state["ratio_choices"], value=state["scale"], visible=state["method"] != "", scale=1, label="Scale", elem_classes=elem_classes)
    value_component = gr.Textbox(value=state["value"], visible=False, elem_classes=elem_classes)

    def refresh_method(method, value):
        ratio_choices, scale, value = normalize_upsampling_value_for_method(method, value)
        return gr.update(choices=ratio_choices, value=scale, visible=bool(method)), value

    def refresh_ratio(method, scale):
        _, scale, value = normalize_upsampling_state(method, scale)
        return gr.update(value=scale, visible=bool(method)), value

    if not update_form:
        outputs = [ratio_component, value_component]
        gr.on(triggers=[method_component.input], fn=refresh_method, inputs=[method_component, value_component], outputs=outputs, show_progress="hidden")
        gr.on(triggers=[ratio_component.input], fn=refresh_ratio, inputs=[method_component, ratio_component], outputs=outputs, show_progress="hidden")
    return {"value": value_component, "method": method_component, "ratio": ratio_component}


def query_postprocessing_upsampling_choices(include_name: bool = True, enabled_only: bool = False, image_outputs: bool | None = None) -> list[tuple[str, str]]:
    """Flat (label, value) choices covering every method x multiplier of the post-processing upsamplers."""
    choices = []
    for handler in upsampler_handlers(UPSAMPLER_TYPE_POSTPROCESSING, enabled_only):
        handler_def = handler.query_upsampler_def()
        if image_outputs is not None:
            media = handler_def.get("media", ("video", "image"))
            if ("image" if image_outputs else "video") not in media:
                continue
        multipliers = handler_def.get("multipliers", {})
        for label, method in handler_def.get("methods", []):
            if not _handler_method_available(handler, method):
                continue
            for scale in multipliers.get(method, ()):
                value = handler.build_value(method, scale)
                if value is not None:
                    choices.append((_method_pos(handler_def, method), str(label or "").casefold(), float(scale), str(value or ""), format_method_scale_label(label, scale) if include_name else format_multiplier_label(scale), value))
    return [(label, value) for _, _, _, _, label, value in sorted(choices)]


def profile_type_for_handler(handler) -> str:
    handler_def = handler.query_upsampler_def()
    profile = str(handler_def.get("profile", "") or "").strip().lower()
    if profile in (UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_PROFILE_IMAGE, UPSAMPLER_PROFILE_AUDIO):
        return profile
    if getattr(handler, "uses_image_profile", False):
        return UPSAMPLER_PROFILE_IMAGE
    media = handler_def.get("media", ())
    if media == (UPSAMPLER_PROFILE_IMAGE,):
        return UPSAMPLER_PROFILE_IMAGE
    if media == (UPSAMPLER_PROFILE_AUDIO,):
        return UPSAMPLER_PROFILE_AUDIO
    return UPSAMPLER_PROFILE_VIDEO


def config_key_for_handler(handler) -> str:
    handler_def = handler.query_upsampler_def()
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return handler.__class__.__name__.lower()


def _nested_configs(server_config: dict[str, Any]) -> dict[str, Any]:
    configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(configs, dict):
        configs = {}
        server_config[UPSAMPLER_CONFIG_KEY] = configs
    else:
        server_config.setdefault(UPSAMPLER_CONFIG_KEY, configs)
    return configs


def _default_config(handler) -> dict[str, Any]:
    return dict(handler.default_config()) if hasattr(handler, "default_config") else {}


def _legacy_config(handler, server_config: dict[str, Any]) -> dict[str, Any]:
    return dict(handler.legacy_config(server_config)) if hasattr(handler, "legacy_config") else {}


def _legacy_config_keys(handler) -> tuple[str, ...]:
    return tuple(handler.legacy_config_keys()) if hasattr(handler, "legacy_config_keys") else ()


def _normalize_config_section(handler, config: dict[str, Any]) -> dict[str, Any]:
    return dict(handler.normalize_config_section(config)) if hasattr(handler, "normalize_config_section") else dict(config)


def read_config_section(server_config: dict[str, Any], handler, *, prefer_legacy: bool = False) -> dict[str, Any]:
    configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
    config_key = config_key_for_handler(handler)
    nested = configs.get(config_key, {}) if isinstance(configs, dict) else {}
    nested = nested if isinstance(nested, dict) else {}
    nested_exists = isinstance(configs, dict) and config_key in configs
    legacy = _legacy_config(handler, server_config) if prefer_legacy or not nested_exists else {}
    values = {**_default_config(handler), **dict(nested or {}), **legacy}
    return _normalize_config_section(handler, values)


def read_config_section_by_key(server_config: dict[str, Any], config_key: str) -> dict[str, Any]:
    handler = next((candidate for candidate in _upsampler_handlers if config_key_for_handler(candidate) == config_key), None)
    if handler is None:
        configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
        return dict(configs.get(config_key, {})) if isinstance(configs, dict) else {}
    return read_config_section(server_config, handler)


def write_config_section(server_config: dict[str, Any], handler, config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config_section(handler, {**_default_config(handler), **dict(config or {})})
    _nested_configs(server_config)[config_key_for_handler(handler)] = config
    return config


def migrate_upsampler_config(server_config: dict[str, Any], *, prefer_legacy: bool = False, apply_pre_1_1_defaults: bool = False) -> bool:
    legacy_keys = tuple(dict.fromkeys(key for handler in _upsampler_handlers for key in _legacy_config_keys(handler)))
    before = repr((server_config.get(UPSAMPLER_CONFIG_KEY, None), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    for handler in _upsampler_handlers:
        if _default_config(handler) or hasattr(handler, "legacy_config"):
            config = write_config_section(server_config, handler, read_config_section(server_config, handler, prefer_legacy=prefer_legacy))
            if apply_pre_1_1_defaults and hasattr(handler, "apply_pre_1_1_defaults") and handler.apply_pre_1_1_defaults(config):
                write_config_section(server_config, handler, config)
    for key in legacy_keys:
        server_config.pop(key, None)
    after = repr((server_config.get(UPSAMPLER_CONFIG_KEY, None), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    return before != after


def config_for_method(method, server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    handler = require_upsampler_by_method(method)
    return read_config_section(handler.server_config if server_config is None else server_config, handler)


def create_config_ui(gr, server_config: dict[str, Any], *, lock_config: bool = False) -> list[UpsamplerConfigBinding]:
    bindings = []
    for handler in _upsampler_handlers:
        if not hasattr(handler, "create_config_ui"):
            continue
        config = write_config_section(server_config, handler, read_config_section(server_config, handler))
        controls = handler.create_config_ui(gr, config, lock_config=lock_config)
        if controls:
            bindings.append(UpsamplerConfigBinding(handler, config_key_for_handler(handler), list(controls)))
    return bindings


def config_components(bindings: list[UpsamplerConfigBinding]) -> list[Any]:
    return [component for binding in bindings for _, component in binding.controls]


def collect_config_update(bindings: list[UpsamplerConfigBinding], values) -> dict[str, dict[str, Any]]:
    values = list(values or [])
    updates, index = {}, 0
    for binding in bindings:
        config = {}
        for field, _ in binding.controls:
            if index >= len(values):
                raise ValueError("Spatial upsampler config UI values do not match registered controls")
            config[field] = values[index]
            index += 1
        updates[binding.config_key] = _normalize_config_section(binding.handler, {**_default_config(binding.handler), **config})
    if index != len(values):
        raise ValueError("Spatial upsampler config UI values do not match registered controls")
    return updates


def validate_config_update_messages(bindings: list[UpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for binding in bindings:
        if binding.config_key in updates and hasattr(binding.handler, "validate_config_section"):
            message = binding.handler.validate_config_section(updates[binding.config_key])
            if isinstance(message, str) and message:
                messages.append(message)
            elif isinstance(message, list):
                messages += [text for text in message if text]
    return messages


def apply_config_update(server_config: dict[str, Any], bindings: list[UpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> None:
    for binding in bindings:
        if binding.config_key in updates:
            write_config_section(server_config, binding.handler, updates[binding.config_key])


def release_changed_config_upsamplers(old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> None:
    changed_keys = set(changed_keys)
    for handler in _upsampler_handlers:
        if not hasattr(handler, "release_vram"):
            continue
        old_section = read_config_section(old_config, handler)
        new_section = read_config_section(new_config, handler)
        if hasattr(handler, "config_requires_release"):
            should_release = handler.config_requires_release(old_section, new_section, changed_keys)
        else:
            should_release = old_section != new_section
        if should_release:
            handler.release_vram()


class SimpleScaleSuffixMixin:
    """Value helpers for upsamplers encoding values as '<method><multiplier>' (e.g. 'lanczos2', 'coz4')."""

    def _method_keys(self):
        handler_def = self.query_upsampler_def()
        return [key for _, key in _method_choices(handler_def)]

    def split_value(self, value):
        text = str(value or "").strip().lower()
        # longest prefix first so 'flashvsr2pass' wins over 'flashvsr'
        for method in sorted(self._method_keys(), key=len, reverse=True):
            if text.startswith(method):
                try:
                    # declared multipliers are UI capabilities; out-of-list scales are
                    # still parsed and rejected by validate_upsampling when unsupported
                    return method, float(text[len(method):] or 2.0)
                except ValueError:
                    return None
        return None

    def build_value(self, method, scale):
        method = str(method or "").strip().lower()
        if method not in self._method_keys():
            return None
        multipliers = self.query_upsampler_def().get("multipliers", {}).get(method, ())
        scale = float(scale or 0)
        if scale not in multipliers:
            scale = _default_multiplier_from_def(self.query_upsampler_def(), method) or 0
        return f"{method}{format_multiplier(scale)}"

    def is_upsampling(self, value) -> bool:
        return self.split_value(value) is not None


class WanVaeUpsampler(SimpleScaleSuffixMixin):
    """Capability declaration for the Wan VAE 2x upscaling decoder.

    VAE upsamplers are plugged directly into model pipelines: models declare
    support through their model def and the upsampler API decides whether a main
    model reload or an external VAE upsampler runtime is needed.
    """

    def __init__(self, server_config=None, files_locator=None):
        pass

    def query_upsampler_def(self) -> dict[str, Any]:
        return {
            "name": "VAE Upscaling",
            "upsampler_types": (UPSAMPLER_TYPE_VAE,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "vae",
            "pos": 30,
            "method_pos": {"vae": 30},
            "methods": [],
            "vae_methods": [("VAE Upscaling", "vae")],
            "multipliers": {"vae": (1.0, 2.0)},
            "default_spatial_upsampling": "vae2",
        }

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        return "" if split is not None and split[1] in self.query_upsampler_def()["multipliers"]["vae"] else "VAE Spatial Upsampling only supports x1.0 and x2.0"

    def supports_model_vae_method(self, method, model_type, model_def, image_mode: int) -> bool:
        return method == "vae" and image_mode in model_def.get("vae_upsampler", [])

    def validate_model_vae_upsampling(self, spatial_upsampling, image_mode: int, model_type, model_def, medium: str) -> str:
        error = self.validate_upsampling(spatial_upsampling, image_mode)
        if error:
            return error
        return "" if self.supports_model_vae_method("vae", model_type, model_def, image_mode) else f"VAE Spatial Upsampling is not available for {medium}"

    def model_load_upsampling_value(self, spatial_upsampling, model_type, model_def, image_mode: int) -> str | None:
        return spatial_upsampling if self.supports_model_vae_method("vae", model_type, model_def, image_mode) else None

    def loaded_model_vae_upsampling_value(self, model) -> str | None:
        return None if model is None or not hasattr(model, "vae") or not hasattr(model.vae, "upsampling_set") else model.vae.upsampling_set

    def model_load_kwargs_for_vae_upsampling(self, spatial_upsampling, model_type, model_def, image_mode: int) -> dict[str, Any]:
        return {"VAE_upsampling": spatial_upsampling}

    def post_model_process_vae_upsampling(self, sample, spatial_upsampling):
        split = self.split_value(spatial_upsampling)
        if split is not None and split[1] == 1.0:
            from PIL import Image
            from postprocessing.lanczos import resize_lanczos_spatial

            return resize_lanczos_spatial(sample, 0.5, method=Image.Resampling.BICUBIC)
        return sample
