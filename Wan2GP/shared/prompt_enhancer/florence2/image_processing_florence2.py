from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers.image_processing_base import ImageProcessingMixin


def _as_list(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if torch.is_tensor(image):
        return image.detach().cpu().numpy()
    if isinstance(image, Image.Image):
        return np.array(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _infer_input_format(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        return "channels_first"
    return "channels_last"


def _to_channels_last(arr: np.ndarray, input_format: str) -> np.ndarray:
    if input_format == "channels_first":
        return np.transpose(arr, (1, 2, 0))
    return arr


def _to_channels_first(arr: np.ndarray, input_format: str) -> np.ndarray:
    if input_format == "channels_last":
        return np.transpose(arr, (2, 0, 1))
    return arr


def _compute_resize_size(image_size: Tuple[int, int], size: Dict[str, int]) -> Tuple[int, int]:
    height, width = image_size
    if "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    if "shortest_edge" in size:
        target = int(size["shortest_edge"])
        if height <= width:
            new_h = target
            new_w = int(round(width * target / max(height, 1)))
        else:
            new_w = target
            new_h = int(round(height * target / max(width, 1)))
        return new_h, new_w
    raise ValueError(f"Unsupported size dict: {size}")


def _resolve_resample(resample: Optional[int]) -> int:
    if resample is None:
        return Image.BICUBIC
    try:
        return Image.Resampling(resample)
    except Exception:
        return resample


def _center_crop_pil(image: Image.Image, crop_size: Dict[str, int]) -> Image.Image:
    target_h = int(crop_size["height"])
    target_w = int(crop_size["width"])
    width, height = image.size
    if width < target_w or height < target_h:
        padded_w = max(width, target_w)
        padded_h = max(height, target_h)
        padded = Image.new(image.mode, (padded_w, padded_h), (0, 0, 0))
        padded.paste(image, ((padded_w - width) // 2, (padded_h - height) // 2))
        image = padded
        width, height = image.size
    left = int(round((width - target_w) / 2.0))
    top = int(round((height - target_h) / 2.0))
    return image.crop((left, top, left + target_w, top + target_h))


def _normalize_return_tensors(value: Optional[Union[str, Any]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.lower()
    name = getattr(value, "name", None)
    if name:
        return name.lower()
    return str(value).lower()


class Florence2ImageProcessorLite(ImageProcessingMixin):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_seq_length: int,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[int] = None,
        do_center_crop: bool = False,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.image_seq_length = int(image_seq_length)
        self.do_resize = bool(do_resize)
        self.size = size or {"height": 224, "width": 224}
        self.resample = resample
        self.do_center_crop = bool(do_center_crop)
        self.crop_size = crop_size or {"height": 224, "width": 224}
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = float(rescale_factor)
        self.do_normalize = bool(do_normalize)
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def from_preprocessor_config(cls, model_dir: Union[str, Path]) -> "Florence2ImageProcessorLite":
        config_path = Path(model_dir) / "preprocessor_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing Florence2 preprocessor_config.json in {model_dir}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            image_seq_length=data.get("image_seq_length", 0),
            do_resize=data.get("do_resize", True),
            size=data.get("size") or data.get("crop_size") or {"height": 224, "width": 224},
            resample=data.get("resample"),
            do_center_crop=data.get("do_center_crop", False),
            crop_size=data.get("crop_size") or data.get("size") or {"height": 224, "width": 224},
            do_rescale=data.get("do_rescale", True),
            rescale_factor=data.get("rescale_factor", 1 / 255),
            do_normalize=data.get("do_normalize", True),
            image_mean=data.get("image_mean"),
            image_std=data.get("image_std"),
            do_convert_rgb=data.get("do_convert_rgb"),
        )

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor, List[Any]],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[int] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Iterable[float]] = None,
        image_std: Optional[Iterable[float]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, Any]] = "pt",
        data_format: Optional[str] = "channels_first",
        input_data_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        resample = self.resample if resample is None else resample
        do_center_crop = self.do_center_crop if do_center_crop is None else do_center_crop
        crop_size = self.crop_size if crop_size is None else crop_size
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = list(self.image_mean if image_mean is None else image_mean)
        image_std = list(self.image_std if image_std is None else image_std)
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb

        resample = _resolve_resample(resample)
        want_torch = _normalize_return_tensors(return_tensors) in ("pt", "pytorch", "tensortype.pytorch")

        processed: List[np.ndarray] = []
        for image in _as_list(images):
            if isinstance(image, Image.Image):
                img = image
                if do_convert_rgb:
                    img = ImageOps.exif_transpose(img).convert("RGB")
            else:
                arr = _to_numpy(image)
                input_fmt = input_data_format or _infer_input_format(arr)
                arr = _to_channels_last(arr, input_fmt)
                img = Image.fromarray(arr.astype(np.uint8))
                if do_convert_rgb:
                    img = img.convert("RGB")

            if do_resize:
                out_h, out_w = _compute_resize_size((img.size[1], img.size[0]), size)
                img = img.resize((out_w, out_h), resample=resample)

            if do_center_crop:
                img = _center_crop_pil(img, crop_size)

            arr = np.array(img).astype(np.float32)
            if do_rescale:
                arr = arr * float(rescale_factor)
            if do_normalize:
                mean = np.array(image_mean, dtype=np.float32)
                std = np.array(image_std, dtype=np.float32)
                arr = (arr - mean) / std

            if data_format in ("channels_first", "first"):
                arr = _to_channels_first(arr, "channels_last")
            processed.append(arr)

        batch = np.stack(processed, axis=0)
        if want_torch:
            batch = torch.from_numpy(batch).float()
        return {"pixel_values": batch}
