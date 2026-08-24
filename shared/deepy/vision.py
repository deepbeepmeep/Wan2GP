from __future__ import annotations

import math
from typing import Any

from PIL import Image

from shared.prompt_enhancer.qwen35_vl import _prepare_multimodal_vllm_prompt
from shared.utils.video_decode import decode_video_frame_indices_ffmpeg


VISION_MAX_IMAGES = 5
VISION_REMOTE_MAX_IMAGES = 10
VISION_MAX_VISUAL_TOKENS_PER_IMAGE = 1024
VISION_ANSWER_MAX_NEW_TOKENS = 1024
VISION_REMOTE_MAX_IMAGE_EDGE = 1024
VISION_VIDEO_MAX_IMAGES = 80
VISION_VIDEO_REMOTE_MAX_IMAGES = 160
VISION_VIDEO_MAX_IMAGE_EDGE = 256
VISION_VIDEO_MID_RES_MAX_IMAGE_EDGE = 512
VISION_VIDEO_MID_RES_SAMPLE_DIVISOR = 4
VISION_VIDEO_MAX_SAMPLES_PER_SECOND = 2
VISION_VIDEO_DECODE_BATCH_SIZE = 8
VISION_QA_SYSTEM_PROMPT = "Answer the user's question about the labeled visual inputs accurately and concisely. Inputs may be images or ordered frames from one or more videos. If the answer is uncertain, say so."


def resize_inspection_image(image: Any, max_edge: int) -> Image.Image:
    resized = image.convert("RGB")
    resized.thumbnail((int(max_edge), int(max_edge)), Image.Resampling.LANCZOS)
    return resized


def decode_inspection_video_frames(path: str, frame_indices: list[int], max_edge: int | None = None) -> list[Image.Image]:
    images = []
    for offset in range(0, len(frame_indices), VISION_VIDEO_DECODE_BATCH_SIZE):
        current_indices = frame_indices[offset:offset + VISION_VIDEO_DECODE_BATCH_SIZE]
        frames = decode_video_frame_indices_ffmpeg(path, current_indices, bridge="numpy")
        if len(frames) != len(current_indices):
            raise RuntimeError(f"Video decoder returned {len(frames)} of {len(current_indices)} requested frames.")
        for frame in frames:
            image = Image.fromarray(frame).convert("RGB")
            images.append(resize_inspection_image(image, max_edge) if max_edge is not None else image)
    return images


def video_inspection_sample_count(*, remote: bool, mid_res_sampling: bool) -> int:
    base_count = VISION_VIDEO_REMOTE_MAX_IMAGES if remote else VISION_VIDEO_MAX_IMAGES
    return base_count // VISION_VIDEO_MID_RES_SAMPLE_DIVISOR if mid_res_sampling else base_count


def _inspection_image_size(processor: Any, max_pixels_per_image: int | None = None) -> tuple[dict[str, int], int, int]:
    image_processor = processor.image_processor
    merge_size = int(image_processor.merge_size)
    token_edge = int(image_processor.patch_size) * merge_size
    token_budget_pixels = VISION_MAX_VISUAL_TOKENS_PER_IMAGE * token_edge * token_edge
    max_pixels = token_budget_pixels if max_pixels_per_image is None else min(token_budget_pixels, int(max_pixels_per_image))
    min_pixels = min(int(image_processor.size.get("shortest_edge", max_pixels)), max_pixels)
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}, merge_size, min(VISION_MAX_VISUAL_TOKENS_PER_IMAGE, math.ceil(max_pixels / (token_edge * token_edge)))


def build_image_question_prompt(caption_model: Any, processor: Any, image: Any, question: str, system_prompt: str | None = None, image_labels: list[str] | None = None, *, max_images: int = VISION_MAX_IMAGES, max_pixels_per_image: int | None = None):
    question = str(question or "").strip()
    if len(question) == 0:
        raise ValueError("Vision question is empty.")
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if not 1 <= len(images) <= int(max_images):
        raise ValueError(f"Vision inspection requires between 1 and {int(max_images)} images.")
    if image_labels is not None and len(image_labels) != len(images):
        raise ValueError("Vision input labels must match the image count.")
    messages = []
    system_prompt = str(system_prompt or VISION_QA_SYSTEM_PROMPT).strip()
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    content = []
    for index, current_image in enumerate(images):
        if image_labels is not None:
            content.append({"type": "text", "text": str(image_labels[index]).strip()})
        content.append({"type": "image", "image": current_image})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_size, merge_size, max_visual_tokens = _inspection_image_size(processor, max_pixels_per_image=max_pixels_per_image)
    model_inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
        return_mm_token_type_ids=True,
        images_kwargs={"size": image_size},
    )
    image_grid_thw = model_inputs.get("image_grid_thw")
    image_grids = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
    if image_grids is None or len(image_grids) != len(images):
        raise RuntimeError("Vision processor returned an unexpected image grid count.")
    if any(int(grid[0]) * int(grid[1]) * int(grid[2]) // (merge_size * merge_size) > max_visual_tokens for grid in image_grids):
        raise RuntimeError("Vision processor exceeded the per-image visual token limit.")
    return _prepare_multimodal_vllm_prompt(caption_model, model_inputs)


__all__ = [
    "VISION_ANSWER_MAX_NEW_TOKENS", "VISION_MAX_IMAGES", "VISION_MAX_VISUAL_TOKENS_PER_IMAGE", "VISION_QA_SYSTEM_PROMPT",
    "VISION_REMOTE_MAX_IMAGES", "VISION_REMOTE_MAX_IMAGE_EDGE", "VISION_VIDEO_MAX_IMAGE_EDGE", "VISION_VIDEO_MAX_IMAGES",
    "VISION_VIDEO_MAX_SAMPLES_PER_SECOND", "VISION_VIDEO_MID_RES_MAX_IMAGE_EDGE", "VISION_VIDEO_MID_RES_SAMPLE_DIVISOR", "VISION_VIDEO_REMOTE_MAX_IMAGES", "build_image_question_prompt",
    "decode_inspection_video_frames", "resize_inspection_image", "video_inspection_sample_count",
]
