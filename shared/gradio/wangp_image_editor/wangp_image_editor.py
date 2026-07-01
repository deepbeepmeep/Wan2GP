from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from threading import RLock

from gradio.components.base import server
from gradio.components.image_editor import AcceptBlobs, EditorDataBlobs, ImageEditor
from PIL import Image

from shared.gradio import gradio_save_image_cache_patch


gradio_save_image_cache_patch.install()


_WANGP_VALUE_CACHE_LIMIT = 32
_wangp_state_lock = RLock()
_wangp_blob_changes_by_id = {}
_wangp_blob_storage_by_id = {}
_wangp_value_cache_by_id = OrderedDict()
_wangp_output_cache_by_id = OrderedDict()
_wangp_last_value_by_instance = {}


class WanGPImageEditor(ImageEditor):
    TEMPLATE_DIR = "templates/"
    FRONTEND_DIR = "frontend/"
    WANGP_FRONTEND_BUILD_ID = "20260701-connection-loss-export-log-34"
    _wangp_magic_mask_patch_enabled = True

    @classmethod
    def get_component_class_id(cls) -> str:
        return hashlib.sha256(f"{cls.__module__}.{cls.__name__}:{cls.WANGP_FRONTEND_BUILD_ID}".encode()).hexdigest()


def _wangp_debug(self, message: str) -> None:
    if gradio_save_image_cache_patch.WANGP_GRADIO_IMAGE_DEBUG:
        print(
            "[WanGP image-editor] "
            f"component={getattr(self, '_id', None)} elem_id={getattr(self, 'elem_id', None)} "
            f"label={getattr(self, 'label', None)!r} {message}",
            flush=True,
        )


@server
def _wangp_accept_blobs(self, data: AcceptBlobs):
    if not hasattr(self, "_wangp_blob_changes"):
        self._wangp_blob_changes = {}
    payload_id = data.data["id"]
    blob_type = data.data["type"]
    instance_id = data.data.get("instance_id")
    if blob_type == "wangp_meta":
        raw = data.files[0][1]
        changes = json.loads(raw.decode("utf-8"))
        changes["_instance_id"] = instance_id
        self._wangp_blob_changes[payload_id] = changes
        with _wangp_state_lock:
            _wangp_blob_changes_by_id[payload_id] = changes
        self._wangp_debug(
            "dirty_state "
            f"id={payload_id} instance={instance_id} image_changed={bool(changes.get('background'))} "
            f"mask_changed={bool(changes.get('layers'))} composite_changed={bool(changes.get('composite'))}"
        )
        return

    output = super(WanGPImageEditor, self).accept_blobs(data)
    index = int(data.data["index"]) if data.data["index"] and data.data["index"] != "null" else None
    file = data.files[0][1]
    with _wangp_state_lock:
        current = _wangp_blob_storage_by_id.get(payload_id, EditorDataBlobs(background=None, layers=[], composite=None))
        if blob_type == "layer" and index is not None:
            if index >= len(current.layers):
                current.layers.extend([None] * (index + 1 - len(current.layers)))
            current.layers[index] = file
        elif blob_type == "background":
            current.background = file
        elif blob_type == "composite":
            current.composite = file
        _wangp_blob_storage_by_id[payload_id] = current
    return output


def _wangp_convert_cached_image(self, file):
    return self.convert_and_format_image(file)


def _wangp_remember_value(value_id, value, instance_id=None):
    if value_id is not None and value is not None:
        _wangp_value_cache_by_id[value_id] = value
        _wangp_value_cache_by_id.move_to_end(value_id)
        while len(_wangp_value_cache_by_id) > _WANGP_VALUE_CACHE_LIMIT:
            _wangp_value_cache_by_id.popitem(last=False)
    if instance_id is not None and value is not None:
        _wangp_last_value_by_instance[instance_id] = value


def _wangp_rebind_instance_values(previous, value):
    if previous is None or value is None:
        return
    for instance_id, instance_value in list(_wangp_last_value_by_instance.items()):
        if _wangp_same_editor_value(instance_value, previous):
            _wangp_last_value_by_instance[instance_id] = value


def _wangp_remember_output(value_id, output):
    if value_id is None or output is None:
        return
    _wangp_output_cache_by_id[value_id] = output
    _wangp_output_cache_by_id.move_to_end(value_id)
    while len(_wangp_output_cache_by_id) > _WANGP_VALUE_CACHE_LIMIT:
        _wangp_output_cache_by_id.popitem(last=False)


def _wangp_cached_output(value_id):
    output = _wangp_output_cache_by_id.get(value_id)
    if output is not None:
        _wangp_output_cache_by_id.move_to_end(value_id)
    return output


def _wangp_preprocess_blobs(self, payload):
    with _wangp_state_lock:
        cached_value = _wangp_value_cache_by_id.get(payload.id)
        if cached_value is not None:
            _wangp_value_cache_by_id.move_to_end(payload.id)
    if cached_value is not None:
        value = _wangp_copy_editor_value(cached_value)
        self._wangp_last_preprocessed_value = value
        self._wangp_last_preprocessed_value_id = payload.id
        self._wangp_debug(
            "read_value "
            f"id={payload.id} source=value_cache image_changed=False mask_changed=False "
            f"composite_changed=False reuse_image=True reuse_mask=True layers={len(value['layers'])}"
        )
        return value

    changes = getattr(self, "_wangp_blob_changes", {}).pop(payload.id, None)
    with _wangp_state_lock:
        global_changes = _wangp_blob_changes_by_id.pop(payload.id, None)
    if changes is None:
        changes = global_changes or {}

    base_id = changes.get("base_id")
    with _wangp_state_lock:
        previous = _wangp_value_cache_by_id.get(base_id) if base_id is not None else None
        if previous is not None:
            _wangp_value_cache_by_id.move_to_end(base_id)
    previous_source = "base" if previous is not None else "instance"
    instance_id = changes.get("_instance_id")
    if previous is None:
        with _wangp_state_lock:
            previous = _wangp_last_value_by_instance.get(instance_id) if instance_id is not None else None
    if previous is None:
        previous = getattr(self, "_wangp_last_preprocessed_value", None)
        previous_source = "component"
    if previous is None:
        changes = {"background": True, "layers": True, "composite": False}
        previous_source = "none"

    background_changed = bool(changes.get("background", True))
    layers_changed = bool(changes.get("layers", True))
    composite_changed = bool(changes.get("composite", False))

    cached = self.blob_storage.get(payload.id)
    if cached is None:
        with _wangp_state_lock:
            cached = _wangp_blob_storage_by_id.get(payload.id)
    if cached is None and (background_changed or layers_changed or composite_changed):
        raise RuntimeError(f"WanGPImageEditor missing in-memory blob payload for changed value id {payload.id}")
    if background_changed and cached.background is None:
        raise RuntimeError(f"WanGPImageEditor missing changed background blob for id {payload.id}")
    if composite_changed and cached.composite is None:
        raise RuntimeError(f"WanGPImageEditor missing changed composite blob for id {payload.id}")

    background = self._wangp_convert_cached_image(cached.background) if background_changed else previous["background"]
    layers = [self._wangp_convert_cached_image(layer) for layer in cached.layers] if layers_changed else previous["layers"]
    composite = self._wangp_convert_cached_image(cached.composite) if composite_changed else previous["composite"] if previous is not None else None

    value = _wangp_normalize_editor_value({
        "background": background,
        "layers": [layer for layer in layers if layer is not None] if layers else [],
        "composite": composite,
    })
    self._wangp_last_preprocessed_value = value
    self._wangp_last_preprocessed_value_id = payload.id
    with _wangp_state_lock:
        _wangp_remember_value(payload.id, value, instance_id)
        _wangp_blob_storage_by_id.pop(payload.id, None)
    self.blob_storage.pop(payload.id, None)
    self._wangp_debug(
        "read_value "
        f"id={payload.id} instance={instance_id} previous={previous_source} "
        f"image_changed={background_changed} mask_changed={layers_changed} "
        f"composite_changed={composite_changed} reuse_image={not background_changed} "
        f"reuse_mask={not layers_changed} layers={len(value['layers'])}"
    )
    return value


def _wangp_preprocess(self, payload):
    if payload is not None and payload.id is not None:
        return self._wangp_preprocess_blobs(payload)
    output = super(WanGPImageEditor, self).preprocess(payload)
    if output is None:
        self._wangp_last_preprocessed_value = None
        self._wangp_last_preprocessed_value_id = None
    else:
        self._wangp_last_preprocessed_value = output
        self._wangp_last_preprocessed_value_id = None
    return output


def _wangp_image_size(image):
    size = getattr(image, "size", None)
    return tuple(size) if isinstance(size, tuple) and len(size) == 2 else None


def _wangp_normalize_mask_layer(layer):
    if not isinstance(layer, Image.Image) or layer.mode != "RGBA":
        return layer
    red, green, blue, alpha = layer.split()
    red_extrema = red.getextrema()
    green_extrema = green.getextrema()
    blue_extrema = blue.getextrema()
    alpha_extrema = alpha.getextrema()
    if red_extrema != (255, 255) or green_extrema != (255, 255) or blue_extrema != (255, 255):
        return layer
    if alpha_extrema[0] == 255:
        return layer
    return Image.merge("RGBA", (alpha, alpha, alpha, alpha))


def _wangp_normalize_editor_value(value):
    if value is None:
        return None
    value["layers"] = [_wangp_normalize_mask_layer(layer) for layer in value["layers"]]
    return value


def _wangp_compatible_layers(background, previous):
    if not previous or previous["background"] is None or not previous["layers"]:
        return []
    if _wangp_image_size(background) != _wangp_image_size(previous["background"]):
        return []
    return list(previous["layers"])


def _wangp_empty_mask(background):
    size = _wangp_image_size(background)
    return Image.new("RGBA", size, (255, 255, 255, 0)) if size is not None else None


def _wangp_value_to_editor_value(value, previous=None):
    if value is None:
        return None
    if isinstance(value, dict):
        return _wangp_normalize_editor_value({
            "background": value["background"],
            "layers": list(value["layers"]),
            "composite": None,
        })
    layers = _wangp_compatible_layers(value, previous)
    if not layers:
        empty_mask = _wangp_empty_mask(value)
        layers = [empty_mask] if empty_mask is not None else []
    return _wangp_normalize_editor_value({"background": value, "layers": layers, "composite": None})


def _wangp_copy_editor_value(value):
    if value is None:
        return None
    return {
        "background": value["background"],
        "layers": list(value["layers"]),
        "composite": value["composite"],
    }


def _wangp_same_editor_value(left, right):
    if left is not right and (not isinstance(left, dict) or not isinstance(right, dict)):
        return False
    if left is right:
        return True
    if left["background"] is not right["background"] or left["composite"] is not right["composite"]:
        return False
    return len(left["layers"]) == len(right["layers"]) and all(left_layer is right_layer for left_layer, right_layer in zip(left["layers"], right["layers"]))


def _wangp_image_debug_part(image):
    if image is None:
        return "None"
    mode = getattr(image, "mode", None)
    size = getattr(image, "size", None)
    alpha = f", alpha={image.getchannel('A').getextrema()}" if isinstance(image, Image.Image) and "A" in image.getbands() else ""
    return f"{type(image).__name__}(id={id(image)}, mode={mode}, size={size}{alpha})"


def _wangp_postprocess(self, value):
    previous = getattr(self, "_wangp_last_preprocessed_value", None)
    previous_id = getattr(self, "_wangp_last_preprocessed_value_id", None)
    with _wangp_state_lock:
        output = _wangp_cached_output(previous_id) if _wangp_same_editor_value(value, previous) else None
    if output is not None:
        self._wangp_debug(f"postprocess_value id={previous_id} source=output_cache")
        return output
    cached_value = _wangp_value_to_editor_value(value, previous)
    value_id = f"wangp-{uuid.uuid4().hex}" if cached_value is not None else None
    value_kind = type(value).__name__ if value is not None else "None"
    if cached_value is None:
        self._wangp_debug(f"postprocess_value kind={value_kind} empty=True background=None layers=0 composite=None")
    else:
        self._wangp_debug(
            "postprocess_value "
            f"id={value_id} kind={value_kind} empty=False background={_wangp_image_debug_part(cached_value['background'])} "
            f"layers={len(cached_value['layers'])} "
            f"layer0={_wangp_image_debug_part(cached_value['layers'][0]) if cached_value['layers'] else 'None'} "
            f"composite={_wangp_image_debug_part(cached_value['composite'])}"
        )
    output = super(WanGPImageEditor, self).postprocess(_wangp_copy_editor_value(cached_value))
    self._wangp_last_preprocessed_value = cached_value if output is not None else None
    if output is not None and value_id is not None:
        with _wangp_state_lock:
            _wangp_remember_value(value_id, cached_value)
            _wangp_remember_output(value_id, output)
            _wangp_rebind_instance_values(previous, cached_value)
        output.id = value_id
        self._wangp_last_preprocessed_value_id = value_id
    elif output is None:
        self._wangp_last_preprocessed_value_id = None
    return output


WanGPImageEditor._wangp_debug = _wangp_debug
_wangp_accept_blobs.__name__ = "accept_blobs"
_wangp_accept_blobs.__qualname__ = "WanGPImageEditor.accept_blobs"
WanGPImageEditor.accept_blobs = _wangp_accept_blobs
WanGPImageEditor._wangp_convert_cached_image = _wangp_convert_cached_image
WanGPImageEditor._wangp_preprocess_blobs = _wangp_preprocess_blobs
WanGPImageEditor.preprocess = _wangp_preprocess
WanGPImageEditor.postprocess = _wangp_postprocess
