"""
Qwen image compositing utilities.

Handles resolution capping and mask-based image compositing
for Qwen inpainting/annotation workflows.

Extracted from qwen_handler.py.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image  # type: ignore

from source.core.log import model_logger

# Maximum pixel dimension for Qwen image editing tasks
QWEN_MAX_DIMENSION = 1200


def cap_qwen_resolution(
    resolution_str: str,
    max_dimension: int = QWEN_MAX_DIMENSION,
    task_id: str = "unknown",
) -> Optional[str]:
    """
    Cap resolution to max_dimension px on the longest side while maintaining aspect ratio.

    Returns the (possibly capped) resolution string, or None if the input is invalid.
    """
    if not resolution_str or "x" not in resolution_str:
        return None
    try:
        width, height = map(int, resolution_str.split("x"))
    except ValueError:
        model_logger.warning(
            f"[QWEN_COMPOSITOR] Task {task_id}: Invalid resolution format: {resolution_str}",
            task_id=task_id,
        )
        return None

    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        width = int(width * ratio)
        height = int(height * ratio)
        capped = f"{width}x{height}"
        model_logger.debug(
            f"[QWEN_COMPOSITOR] Task {task_id}: Resolution capped from {resolution_str} to {capped}",
            task_id=task_id,
        )
        return capped
    return resolution_str


def create_qwen_masked_composite(
    image_url: str,
    mask_url: str,
    output_dir: Path,
    task_id: str = "unknown",
) -> str:
    """
    Create composite image with green overlay for Qwen inpainting/annotation.

    Downloads the source image and binary mask, resizes both to fit within
    QWEN_MAX_DIMENSION, applies a green overlay on masked regions, and saves
    the result as JPEG.

    Returns the path to the saved composite image.

    Raises ValueError if compositing fails.
    """
    try:
        model_logger.debug_anomaly("QWEN_COMPOSITOR", f"Task {task_id}: Downloading image from {image_url}", task_id=task_id)
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        image = Image.open(BytesIO(img_response.content)).convert("RGB")

        max_dimension = QWEN_MAX_DIMENSION
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            model_logger.debug_anomaly("QWEN_COMPOSITOR", f"Task {task_id}: Resized image from {width}x{height} to {new_width}x{new_height}", task_id=task_id)
            width, height = new_width, new_height

        model_logger.debug_anomaly("QWEN_COMPOSITOR", f"Task {task_id}: Downloading mask from {mask_url}", task_id=task_id)
        mask_response = requests.get(mask_url, timeout=30)
        mask_response.raise_for_status()
        mask = Image.open(BytesIO(mask_response.content)).convert("L")

        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.LANCZOS)
            model_logger.debug_anomaly("QWEN_COMPOSITOR", f"Task {task_id}: Resized mask to match image: {width}x{height}", task_id=task_id)

        mask = mask.point(lambda x: 0 if x < 128 else 255)
        green_overlay = Image.new("RGB", (width, height), (0, 255, 0))
        composite = Image.composite(green_overlay, image, mask)

        output_dir.mkdir(parents=True, exist_ok=True)
        composite_filename = f"inpaint_composite_{task_id}.jpg"
        composite_path = output_dir / composite_filename
        composite.save(composite_path, "JPEG", quality=95)

        model_logger.debug(
            f"[QWEN_COMPOSITOR] Task {task_id}: Created green mask composite: {composite_path}",
            task_id=task_id,
        )
        return str(composite_path)
    except (OSError, ValueError, RuntimeError) as e:
        model_logger.error(
            f"[QWEN_COMPOSITOR] Task {task_id}: Failed to create masked composite: {e}",
            task_id=task_id,
        )
        raise ValueError(f"Composite image creation failed: {e}") from e
