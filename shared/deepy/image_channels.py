"""Image color metadata and channel operations for Deepy media tools."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageColor


CHANNELS = "RGBA"


def image_color_details(image: Image.Image) -> dict:
    bands = list(image.getbands())
    has_alpha = "A" in bands or "transparency" in image.info
    alpha_extrema = None
    if has_alpha:
        alpha = image.getchannel("A") if "A" in bands else image.convert("RGBA").getchannel("A")
        alpha_extrema = [int(value) for value in alpha.getextrema()]
    return {
        "color_mode": image.mode,
        "bands": bands,
        "has_alpha": has_alpha,
        "has_transparent_pixels": bool(alpha_extrema and alpha_extrema[0] < 255),
        "alpha_extrema": alpha_extrema,
    }


def _load_image(path: str) -> Image.Image:
    with Image.open(Path(path)) as image:
        return image.copy()


def _flatten(image: Image.Image, background: str | None) -> Image.Image:
    if not background:
        raise ValueError("background is required when converting an image with alpha to RGB.")
    try:
        color = ImageColor.getrgb(background)
    except ValueError as exc:
        raise ValueError("background must be a color name or hex color.") from exc
    backdrop = Image.new("RGBA", image.size, (*color[:3], 255))
    return Image.alpha_composite(backdrop, image.convert("RGBA")).convert("RGB")


def render_image_channels(
    operation: str,
    image_path: str | None = None,
    mode: str | None = None,
    channels: list[str] | None = None,
    channel_paths: dict[str, str] | None = None,
    background: str | None = None,
) -> list[tuple[str, Image.Image]]:
    """Return named PNG-ready images; callers choose paths and publish them."""
    operation = str(operation or "").strip().lower()
    if operation not in {"convert", "extract", "combine"}:
        raise ValueError("operation must be convert, extract, or combine.")
    if operation != "combine" and not image_path:
        raise ValueError("media_id is required for convert and extract.")
    if mode is not None and mode not in {"RGB", "RGBA"}:
        raise ValueError("mode must be RGB or RGBA.")
    if operation != "extract" and channels is not None:
        raise ValueError("channels is only used by extract.")
    if operation != "combine" and channel_paths:
        raise ValueError("channel_sources is only used by combine.")

    if operation == "convert":
        image = _load_image(image_path)
        target = mode or "RGBA"
        if target == "RGB" and image_color_details(image)["has_alpha"]:
            image = _flatten(image, background)
        else:
            image = image.convert(target)
        return [(f"converted_{target.lower()}", image)]

    if operation == "extract":
        image = _load_image(image_path)
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError("extract requires an RGB or RGBA source image.")
        if channels is not None and not isinstance(channels, list):
            raise ValueError("channels must be a list of R, G, B, or A.")
        selected = list(image.getbands()) if channels is None else channels
        if not selected or len(set(selected)) != len(selected) or any(channel not in image.getbands() for channel in selected):
            raise ValueError("channels must be distinct channels present in the source image.")
        return [(f"channel_{channel}", image.getchannel(channel)) for channel in selected]

    paths = channel_paths or {}
    if not isinstance(paths, dict) or any(channel not in CHANNELS for channel in paths):
        raise ValueError("channel_sources must map only R, G, B, or A to image IDs.")
    if not image_path and not all(channel in paths for channel in "RGB"):
        raise ValueError("combine needs media_id or R, G, and B channel sources.")
    base = _load_image(image_path) if image_path else None
    if base is not None and base.mode not in {"RGB", "RGBA"}:
        raise ValueError("combine requires an RGB or RGBA base image.")
    masks = {}
    for channel, path in paths.items():
        mask = _load_image(path)
        if mask.mode != "L":
            raise ValueError(f"channel_sources.{channel} must be an 8-bit grayscale image.")
        masks[channel] = mask
    size = base.size if base is not None else next(iter(masks.values())).size
    if any(mask.size != size for mask in masks.values()):
        raise ValueError("All channel sources and the base image must have the same dimensions.")
    target = mode or ("RGBA" if "A" in masks or base is not None and base.mode == "RGBA" else "RGB")
    if target == "RGB" and "A" in masks:
        raise ValueError("An A channel requires RGBA output.")
    if base is not None and base.mode == "RGBA" and target == "RGB":
        base = _flatten(base, background)
    elif base is not None:
        base = base.convert(target)
    base_bands = dict(zip(base.getbands(), base.split())) if base is not None else {}
    merged = Image.merge(target, tuple(masks.get(channel) or base_bands.get(channel) or Image.new("L", size, 255) for channel in target))
    return [(f"combined_{target.lower()}", merged)]
