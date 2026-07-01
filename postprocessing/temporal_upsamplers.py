"""Temporal upsampler plugin API.

Temporal upsamplers are interchangeable handlers for frame interpolation.
Handlers expose methods and supported multipliers through
``query_temporal_upsampler_def()`` and may expose config controls under
``wgp_config["temporal_upsamplers"][config_key]``.

Plugin authors can register processors from ``plugin_info.json`` with:

```json
{
  "temporal_upsampler_handlers": ["my_plugin.temporal.MyTemporalUpsampler"]
}
```
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from shared.attention import attention_shared_state
from shared.utils import offload_registry


TEMPORAL_UPSAMPLER_CONFIG_KEY = "temporal_upsamplers"

temporal_upsampler_handlers = [
    "postprocessing.rife.temporal_upsampler.RifeTemporalUpsampler",
]
_temporal_upsampler_handlers: list[Any] = []
_registered_temporal_upsampler_handler_paths: set[str] = set()


@dataclass
class TemporalUpsamplerConfigBinding:
    handler: Any
    config_key: str
    controls: list[tuple[str, Any]]


def format_multiplier(scale: float) -> str:
    scale = float(scale)
    return str(int(scale)) if scale.is_integer() else f"{scale:g}"


def format_multiplier_label(scale: float) -> str:
    return f"x{format_multiplier(scale)}"


def format_method_scale_label(label: str, scale: float) -> str:
    return f"{str(label or '').removesuffix(' Upsampler')} {format_multiplier_label(scale)}"


def register_temporal_upsampler(handler) -> None:
    if handler not in _temporal_upsampler_handlers:
        _temporal_upsampler_handlers.append(handler)


def _load_upsampler_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _method_choices(handler_def: dict[str, Any]) -> list[tuple[str, str]]:
    return [(label, method) for label, method in handler_def.get("methods", [])]


def _method_labels(handler_def: dict[str, Any]) -> dict[str, str]:
    return {method: label for label, method in _method_choices(handler_def)}


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
    for path in temporal_upsampler_handlers if handler_modules is None else handler_modules:
        handler_cls = _load_upsampler_class(str(path or "").strip())
        if not hasattr(handler_cls, "default_config"):
            continue
        config = dict(handler_cls.default_config())
        if not config:
            continue
        handler_def = handler_cls.query_temporal_upsampler_def()
        sections[_config_key_from_handler_def(handler_def, handler_cls.__name__)] = config
    return sections


def register_temporal_upsamplers(server_config, files_locator, handler_modules: list[str] | None = None) -> None:
    modules = temporal_upsampler_handlers if handler_modules is None else handler_modules
    for path in modules:
        path = str(path or "").strip()
        if not path or path in _registered_temporal_upsampler_handler_paths:
            continue
        register_temporal_upsampler(_load_upsampler_class(path)(server_config, files_locator))
        _registered_temporal_upsampler_handler_paths.add(path)


def registered_temporal_upsamplers(enabled_only: bool = False) -> list[Any]:
    return [handler for handler in _temporal_upsampler_handlers if not enabled_only or handler_enabled(handler)]


def handler_enabled(handler) -> bool:
    return not hasattr(handler, "enabled") or handler.enabled()


def query_temporal_upsampler_defs(enabled_only: bool = False) -> list[dict[str, Any]]:
    return [handler.query_temporal_upsampler_def() for handler in registered_temporal_upsamplers(enabled_only)]


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


def find_temporal_upsampler(temporal_upsampling) -> Any | None:
    if not str(temporal_upsampling or "").strip():
        return None
    return next((handler for handler in _temporal_upsampler_handlers if handler.is_upsampling(temporal_upsampling)), None)


def find_temporal_upsampler_by_method(method) -> Any | None:
    method = str(method or "").strip()
    if not method:
        return None
    for handler in _temporal_upsampler_handlers:
        if method in [key for _, key in _method_choices(handler.query_temporal_upsampler_def())]:
            return handler
    return None


def require_temporal_upsampler_by_method(method) -> Any:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        raise RuntimeError(f"No temporal upsampler registered for method '{method}'")
    return handler


def method_multipliers(method) -> tuple[float, ...]:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        return ()
    return tuple(handler.query_temporal_upsampler_def().get("multipliers", {}).get(str(method), ()))


def multiplier_choices_for_method(method) -> list[tuple[str, float]]:
    return [(format_multiplier_label(scale), scale) for scale in method_multipliers(method)]


def _default_multiplier_from_def(handler_def: dict[str, Any], method: str) -> float | None:
    multipliers = tuple(handler_def.get("multipliers", {}).get(method, ()))
    if not multipliers:
        return None
    default_value = handler_def.get("default_temporal_upsampling", "")
    handler = find_temporal_upsampler(default_value)
    if handler is not None:
        split = handler.split_value(default_value)
        if split is not None and split[0] == method and split[1] in multipliers:
            return split[1]
    return multipliers[0]


def default_multiplier_for_method(method) -> float:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        return 2.0
    return _default_multiplier_from_def(handler.query_temporal_upsampler_def(), str(method or "").strip()) or 2.0


def normalize_multiplier_for_method(method, scale) -> float:
    multipliers = method_multipliers(method)
    if not multipliers:
        return 2.0
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = default_multiplier_for_method(method)
    return scale if scale in multipliers else default_multiplier_for_method(method)


def split_temporal_upsampling_value(value) -> tuple[str, float] | None:
    handler = find_temporal_upsampler(value)
    return None if handler is None else handler.split_value(value)


def build_temporal_upsampling_value(method, scale) -> str | None:
    handler = find_temporal_upsampler_by_method(method)
    return None if handler is None else handler.build_value(method, scale)


def format_temporal_upsampling_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    handler = find_temporal_upsampler(text)
    if handler is None:
        return text
    split = handler.split_value(text)
    if split is None:
        return text
    method, scale = split
    label = _method_labels(handler.query_temporal_upsampler_def()).get(method)
    return format_method_scale_label(label, scale) if label else text


def normalize_temporal_upsampling_state(method, scale) -> tuple[list[tuple[str, float]], float | None, str]:
    method = str(method or "").strip()
    if not method:
        return [], None, ""
    multiplier_choices = multiplier_choices_for_method(method)
    scale = normalize_multiplier_for_method(method, scale)
    return multiplier_choices, scale, build_temporal_upsampling_value(method, scale) or ""


def normalize_temporal_upsampling_value_for_method(method, current_value) -> tuple[list[tuple[str, float]], float | None, str]:
    split = split_temporal_upsampling_value(current_value)
    return normalize_temporal_upsampling_state(method, 2.0 if split is None else split[1])


def query_method_choices(enabled_only: bool = False) -> list[tuple[str, str]]:
    choices = []
    for handler in registered_temporal_upsamplers(enabled_only):
        handler_def = handler.query_temporal_upsampler_def()
        choices += [(_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), label, method) for label, method in _method_choices(handler_def)]
    return [(label, method) for _, _, _, label, method in sorted(choices)]


def dropdown_state(temporal_upsampling, *, late_postprocessing: bool = False) -> dict[str, Any]:
    method, scale = split_temporal_upsampling_value(temporal_upsampling) or ("", 2.0)
    method_choices = [("None", "")] + query_method_choices(enabled_only=not late_postprocessing)
    if method not in {value for _, value in method_choices}:
        method = ""
    multiplier_choices, scale, value = normalize_temporal_upsampling_state(method, scale)
    return {"method": method, "scale": scale, "value": value, "method_choices": method_choices, "multiplier_choices": multiplier_choices}


def create_generation_temporal_ui(gr, temporal_upsampling, *, visible: bool = True, late_postprocessing: bool = False, elem_classes=None, update_form: bool = False, field_help=None) -> dict[str, Any]:
    state = dropdown_state(temporal_upsampling if visible else "", late_postprocessing=late_postprocessing)
    with gr.Row():
        method = gr.Dropdown(choices=state["method_choices"], value=state["method"], visible=visible, scale=3, label="Temporal Upsampling", elem_classes=elem_classes)
        if field_help is not None:
            field_help.bind(method, "temporal_upsampling")
        multiplier = gr.Dropdown(choices=state["multiplier_choices"], value=state["scale"], visible=visible and state["method"] != "", scale=1, label="Multiplier", elem_classes=elem_classes)
    value = gr.Textbox(value=state["value"], visible=False, elem_classes=elem_classes)

    def refresh_method(method, value):
        multiplier_choices, scale, value = normalize_temporal_upsampling_value_for_method(method, value)
        return gr.update(choices=multiplier_choices, value=scale, visible=bool(method)), value

    def refresh_multiplier(method, scale):
        _, scale, value = normalize_temporal_upsampling_state(method, scale)
        return gr.update(value=scale, visible=bool(method)), value

    if not update_form:
        outputs = [multiplier, value]
        gr.on(triggers=[method.input], fn=refresh_method, inputs=[method, value], outputs=outputs, show_progress="hidden")
        gr.on(triggers=[multiplier.input], fn=refresh_multiplier, inputs=[method, multiplier], outputs=outputs, show_progress="hidden")
    return {"value": value, "method": method, "multiplier": multiplier}


def validate_temporal_upsampling(temporal_upsampling, *, source_is_image: bool = False) -> str:
    temporal_upsampling = str(temporal_upsampling or "").strip()
    if not temporal_upsampling:
        return ""
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None:
        return f"No temporal upsampler registered for '{temporal_upsampling}'"
    if hasattr(handler, "validate_upsampling"):
        return handler.validate_upsampling(temporal_upsampling, source_is_image=source_is_image)
    return "Temporal Upsampling can not be used with an Image" if source_is_image else ""


def temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, *, main_offloadobj=None, **kwargs):
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None:
        if str(temporal_upsampling or "").strip():
            raise ValueError(f"No temporal upsampler registered for '{temporal_upsampling}'")
        return sample, previous_last_frame, fps
    name = handler.query_temporal_upsampler_def()["name"]
    persistent = handler.persistent_models() if hasattr(handler, "persistent_models") else False
    with attention_shared_state():
        try:
            if main_offloadobj is not None:
                main_offloadobj.unload_all()
            if hasattr(handler, "load_upsampler"):
                handler.load_upsampler(temporal_upsampling, **kwargs)
            return handler.temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, **kwargs)
        finally:
            if persistent:
                offload_registry.unload_vram([name])
            else:
                offload_registry.release_all([name])


def download_for_value(temporal_upsampling, process_files, **kwargs):
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None or not hasattr(handler, "download"):
        return False
    return handler.download(process_files, temporal_upsampling=temporal_upsampling, **kwargs)


def query_download_defs(enabled_only: bool = True) -> list[dict[str, Any]]:
    defs = []
    for handler in registered_temporal_upsamplers(enabled_only):
        if hasattr(handler, "query_download_defs"):
            defs.extend(handler.query_download_defs(enabled_only=enabled_only))
        elif hasattr(handler, "query_download_def"):
            download_def = handler.query_download_def(enabled_only=enabled_only)
            if download_def is not None:
                defs.append(download_def)
    return defs


def config_key_for_handler(handler) -> str:
    handler_def = handler.query_temporal_upsampler_def()
    return _config_key_from_handler_def(handler_def, handler.__class__.__name__)


def _nested_configs(server_config: dict[str, Any]) -> dict[str, Any]:
    configs = server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(configs, dict):
        configs = {}
        server_config[TEMPORAL_UPSAMPLER_CONFIG_KEY] = configs
    else:
        server_config.setdefault(TEMPORAL_UPSAMPLER_CONFIG_KEY, configs)
    return configs


def _default_config(handler) -> dict[str, Any]:
    return dict(handler.default_config()) if hasattr(handler, "default_config") else {}


def _normalize_config_section(handler, config: dict[str, Any]) -> dict[str, Any]:
    return dict(handler.normalize_config_section(config)) if hasattr(handler, "normalize_config_section") else dict(config)


def read_config_section(server_config: dict[str, Any], handler) -> dict[str, Any]:
    configs = server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    nested = configs.get(config_key_for_handler(handler), {}) if isinstance(configs, dict) else {}
    values = {**_default_config(handler), **(nested if isinstance(nested, dict) else {})}
    return _normalize_config_section(handler, values)


def write_config_section(server_config: dict[str, Any], handler, config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config_section(handler, {**_default_config(handler), **dict(config or {})})
    _nested_configs(server_config)[config_key_for_handler(handler)] = config
    return config


def migrate_temporal_upsampler_config(server_config: dict[str, Any]) -> bool:
    before = repr(server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, None))
    for handler in _temporal_upsampler_handlers:
        if _default_config(handler):
            write_config_section(server_config, handler, read_config_section(server_config, handler))
    return before != repr(server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, None))


def create_config_ui(gr, server_config: dict[str, Any], *, lock_config: bool = False) -> list[TemporalUpsamplerConfigBinding]:
    bindings = []
    for handler in _temporal_upsampler_handlers:
        if not hasattr(handler, "create_config_ui"):
            continue
        config = write_config_section(server_config, handler, read_config_section(server_config, handler))
        controls = handler.create_config_ui(gr, config, lock_config=lock_config)
        if controls:
            bindings.append(TemporalUpsamplerConfigBinding(handler, config_key_for_handler(handler), list(controls)))
    return bindings


def config_components(bindings: list[TemporalUpsamplerConfigBinding]) -> list[Any]:
    return [component for binding in bindings for _, component in binding.controls]


def collect_config_update(bindings: list[TemporalUpsamplerConfigBinding], values) -> dict[str, dict[str, Any]]:
    values = list(values or [])
    updates, index = {}, 0
    for binding in bindings:
        config = {}
        for field, _ in binding.controls:
            if index >= len(values):
                raise ValueError("Temporal upsampler config UI values do not match registered controls")
            config[field] = values[index]
            index += 1
        updates[binding.config_key] = _normalize_config_section(binding.handler, {**_default_config(binding.handler), **config})
    if index != len(values):
        raise ValueError("Temporal upsampler config UI values do not match registered controls")
    return updates


def validate_config_update_messages(bindings: list[TemporalUpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for binding in bindings:
        if binding.config_key in updates and hasattr(binding.handler, "validate_config_section"):
            message = binding.handler.validate_config_section(updates[binding.config_key])
            if isinstance(message, str) and message:
                messages.append(message)
            elif isinstance(message, list):
                messages += [text for text in message if text]
    return messages


def apply_config_update(server_config: dict[str, Any], bindings: list[TemporalUpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> None:
    for binding in bindings:
        if binding.config_key in updates:
            write_config_section(server_config, binding.handler, updates[binding.config_key])


def release_changed_config_temporal_upsamplers(old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> None:
    changed_keys = set(changed_keys)
    for handler in _temporal_upsampler_handlers:
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
    """Value helpers for temporal upsamplers encoding values as '<method><multiplier>'."""

    def _method_keys(self):
        return [key for _, key in _method_choices(self.query_temporal_upsampler_def())]

    def split_value(self, value):
        text = str(value or "").strip().lower()
        for method in sorted(self._method_keys(), key=len, reverse=True):
            if text.startswith(method):
                try:
                    return method, float(text[len(method):] or 2.0)
                except ValueError:
                    return None
        return None

    def build_value(self, method, scale):
        method = str(method or "").strip().lower()
        if method not in self._method_keys():
            return None
        multipliers = self.query_temporal_upsampler_def().get("multipliers", {}).get(method, ())
        scale = float(scale or 0)
        if scale not in multipliers:
            scale = _default_multiplier_from_def(self.query_temporal_upsampler_def(), method) or 0
        return f"{method}{format_multiplier(scale)}"

    def is_upsampling(self, value) -> bool:
        return self.split_value(value) is not None
