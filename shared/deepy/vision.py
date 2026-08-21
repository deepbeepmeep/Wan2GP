from __future__ import annotations

from typing import Any

from shared.prompt_enhancer.qwen35_vl import _prepare_multimodal_vllm_prompt


VISION_MAX_IMAGES = 5
VISION_MAX_VISUAL_TOKENS_PER_IMAGE = 1024
VISION_ANSWER_MAX_NEW_TOKENS = 1024
VISION_QA_SYSTEM_PROMPT = "Answer the user's question about the provided image or images accurately and concisely. If the answer is uncertain, say so."


def _inspection_image_size(processor: Any) -> tuple[dict[str, int], int]:
    image_processor = processor.image_processor
    merge_size = int(image_processor.merge_size)
    token_edge = int(image_processor.patch_size) * merge_size
    max_pixels = VISION_MAX_VISUAL_TOKENS_PER_IMAGE * token_edge * token_edge
    min_pixels = min(int(image_processor.size.get("shortest_edge", max_pixels)), max_pixels)
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}, merge_size


def build_image_question_prompt(caption_model: Any, processor: Any, image: Any, question: str, system_prompt: str | None = None):
    question = str(question or "").strip()
    if len(question) == 0:
        raise ValueError("Vision question is empty.")
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if not 1 <= len(images) <= VISION_MAX_IMAGES:
        raise ValueError(f"Vision inspection requires between 1 and {VISION_MAX_IMAGES} images.")
    messages = []
    system_prompt = str(system_prompt or VISION_QA_SYSTEM_PROMPT).strip()
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "image", "image": current_image} for current_image in images] + [{"type": "text", "text": question}],
        }
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_size, merge_size = _inspection_image_size(processor)
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
    if any(int(grid[0]) * int(grid[1]) * int(grid[2]) // (merge_size * merge_size) > VISION_MAX_VISUAL_TOKENS_PER_IMAGE for grid in image_grids):
        raise RuntimeError("Vision processor exceeded the per-image visual token limit.")
    return _prepare_multimodal_vllm_prompt(caption_model, model_inputs)


__all__ = ["VISION_ANSWER_MAX_NEW_TOKENS", "VISION_MAX_IMAGES", "VISION_MAX_VISUAL_TOKENS_PER_IMAGE", "VISION_QA_SYSTEM_PROMPT", "build_image_question_prompt"]
