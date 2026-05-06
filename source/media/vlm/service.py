"""Backend-neutral service layer for VLM prompt generation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from source.media.vlm.image_prep import create_framed_vlm_image
from source.media.vlm.model import download_qwen_vlm_if_needed
from source.runtime.wgp_bridge import (
    create_qwen_prompt_expander,
    ensure_wan2gp_on_path as _ensure_runtime_bridge_path,
)

DEFAULT_QWEN_VLM_MODEL = "Qwen2.5-VL-7B-Instruct"
DEFAULT_TRANSITION_BASE_PROMPT = (
    "the objects/people inside the scene move excitingly and things transform or shift with the camera"
)
TRANSITION_FALLBACK_PROMPT = "cinematic transition"
SINGLE_IMAGE_FALLBACK_PROMPT = "cinematic video"
ensure_wan2gp_on_path = _ensure_runtime_bridge_path


def initialize_qwen_prompt_extender(
    *,
    device: str = "cuda",
    context: str | None = None,
    expected_items: int | None = None,
):
    """Initialize the shared Qwen VLM prompt extender through the runtime boundary."""
    del context, expected_items

    wan_root = Path(globals()["ensure_wan2gp_on_path"]())
    model_path = wan_root / "ckpts" / DEFAULT_QWEN_VLM_MODEL
    download_qwen_vlm_if_needed(model_path)
    return create_qwen_prompt_expander(
        model_name=str(model_path),
        device=device,
        is_vl=True,
    )


def cleanup_qwen_prompt_extender(extender, *, context: str | None = None) -> None:
    """Best-effort cleanup for VLM prompt extenders."""
    del context
    if extender is None:
        return

    for attr_name in ("model", "processor"):
        if hasattr(extender, attr_name):
            try:
                delattr(extender, attr_name)
            except (AttributeError, TypeError):
                pass

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def _base_prompt_or_default(base_prompt: Optional[str], default: str = DEFAULT_TRANSITION_BASE_PROMPT) -> str:
    return base_prompt if base_prompt and base_prompt.strip() else default


def _fallback_prompt(base_prompt: Optional[str], fallback: str) -> str:
    return base_prompt if base_prompt and base_prompt.strip() else fallback


def _transition_query(base_prompt: Optional[str]) -> str:
    base_prompt_text = _base_prompt_or_default(base_prompt)
    return f"""You are viewing two images side by side: the LEFT image shows the STARTING frame, and the RIGHT image shows the ENDING frame of a video sequence.

Your goal is to create a THREE-SENTENCE prompt that describes the MOTION and CHANGES in this transition based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what MOVES, what CHANGES, and HOW things transition between these frames.

SENTENCE 1 (PRIMARY MOTION): Describe the main action, camera movement, and major scene transitions.
SENTENCE 2 (MOVING ELEMENTS): Describe how the characters, objects, and environment move or change.
SENTENCE 3 (MOTION DETAILS): Describe secondary movements, environmental dynamics, particles, lighting shifts, and small-scale motions.

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""


def _single_image_query(base_prompt: Optional[str]) -> str:
    base_prompt_text = _base_prompt_or_default(base_prompt)
    return f"""You are viewing a single image that will be the starting frame of a video sequence.

Your goal is to create a THREE-SENTENCE prompt that describes the image and suggests NATURAL MOTION based on the user's description: '{base_prompt_text}'

SENTENCE 1 (SCENE & CAMERA): Describe the scene and suggest camera movement.
SENTENCE 2 (SUBJECT MOTION): Describe the main subjects and how they could naturally move or animate.
SENTENCE 3 (ENVIRONMENTAL DYNAMICS): Describe ambient motion, lighting changes, particles, and subtle movements.

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""


def _join_quad_query(base_prompt: Optional[str]) -> str:
    base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a video sequence"
    return f"""Look at these 4 frames from a video, left to right in time order.

The MIDDLE TWO frames show where we need to generate a transition.

Context from user: {base_prompt_text}

Write a short video prompt describing what happens between the middle frames. Include:
1. The motion or action
2. The visual style
3. Key details in the scene

Keep it under 50 words total. Just write the prompt, nothing else.

Your prompt:"""


def _transition_system_prompt() -> str:
    return (
        "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES "
        "following this structure: 1) PRIMARY MOTION, 2) MOVING ELEMENTS, 3) MOTION DETAILS. "
        "Focus exclusively on what moves and changes, not static descriptions."
    )


def _single_image_system_prompt() -> str:
    return (
        "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES "
        "following this structure: 1) SCENE & CAMERA, 2) SUBJECT MOTION, "
        "3) ENVIRONMENTAL DYNAMICS. Focus on natural motion that could emerge from this single image."
    )


def _join_quad_system_prompt() -> str:
    return "Write a short video description prompt. Under 50 words. No explanations, just the prompt."


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _stitch_images_horizontally(images: Sequence[Image.Image]) -> Image.Image:
    combined_width = sum(img.width for img in images)
    combined_height = max(img.height for img in images)
    combined = Image.new("RGB", (combined_width, combined_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined


def generate_transition_pair_prompts(
    image_pairs: Sequence[tuple[str, str]],
    base_prompts: Sequence[Optional[str]],
    *,
    device: str = "cuda",
) -> list[str]:
    """Generate VLM prompts for transition image pairs with legacy fallback behavior."""
    if len(image_pairs) != len(base_prompts):
        raise ValueError(f"image_pairs and base_prompts must have same length ({len(image_pairs)} != {len(base_prompts)})")
    if not image_pairs:
        return []

    extender = None
    try:
        extender = initialize_qwen_prompt_extender(
            device=device,
            context="transition-pairs",
            expected_items=len(image_pairs),
        )
        results: list[str] = []
        for (start_path, end_path), base_prompt in zip(image_pairs, base_prompts):
            try:
                start_img = _load_rgb(start_path)
                end_img = _load_rgb(end_path)
                combined_img = create_framed_vlm_image(start_img, end_img)
                result = extender.extend_with_img(
                    prompt=_transition_query(base_prompt),
                    system_prompt=_transition_system_prompt(),
                    image=combined_img,
                )
                results.append(result.prompt.strip())
            except (RuntimeError, ValueError, OSError, AttributeError):
                results.append(_fallback_prompt(base_prompt, TRANSITION_FALLBACK_PROMPT))
        return results
    except (RuntimeError, ValueError, OSError, ImportError):
        return [_fallback_prompt(base_prompt, TRANSITION_FALLBACK_PROMPT) for base_prompt in base_prompts]
    finally:
        cleanup_qwen_prompt_extender(extender, context="transition-pairs")


def generate_single_image_prompts(
    image_paths: Sequence[str],
    base_prompts: Sequence[Optional[str]],
    *,
    device: str = "cuda",
) -> list[str]:
    """Generate VLM prompts for single start images with legacy fallback behavior."""
    if len(image_paths) != len(base_prompts):
        raise ValueError(f"image_paths and base_prompts must have same length ({len(image_paths)} != {len(base_prompts)})")
    if not image_paths:
        return []

    extender = None
    try:
        extender = initialize_qwen_prompt_extender(
            device=device,
            context="single-images",
            expected_items=len(image_paths),
        )
        results: list[str] = []
        for image_path, base_prompt in zip(image_paths, base_prompts):
            try:
                img = _load_rgb(image_path)
                result = extender.extend_with_img(
                    prompt=_single_image_query(base_prompt),
                    system_prompt=_single_image_system_prompt(),
                    image=img,
                )
                results.append(result.prompt.strip())
            except (RuntimeError, ValueError, OSError, AttributeError):
                results.append(_fallback_prompt(base_prompt, SINGLE_IMAGE_FALLBACK_PROMPT))
        return results
    except (RuntimeError, ValueError, OSError, ImportError):
        return [_fallback_prompt(base_prompt, SINGLE_IMAGE_FALLBACK_PROMPT) for base_prompt in base_prompts]
    finally:
        cleanup_qwen_prompt_extender(extender, context="single-images")


def generate_join_quad_prompts(
    image_quads: Sequence[tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    base_prompt: Optional[str],
    *,
    device: str = "cuda",
) -> list[Optional[str]]:
    """Generate VLM prompts for join-clips four-frame quads, preserving None fallbacks."""
    results: list[Optional[str]] = [None] * len(image_quads)
    valid_items = [(idx, quad) for idx, quad in enumerate(image_quads) if all(path is not None for path in quad)]
    if not valid_items:
        return results

    extender = None
    try:
        extender = initialize_qwen_prompt_extender(
            device=device,
            context="join-quads",
            expected_items=len(valid_items),
        )
        for idx, quad in valid_items:
            try:
                images = [_load_rgb(path) for path in quad if path is not None]
                combined_img = _stitch_images_horizontally(images)
                result = extender.extend_with_img(
                    prompt=_join_quad_query(base_prompt),
                    system_prompt=_join_quad_system_prompt(),
                    image=combined_img,
                )
                results[idx] = result.prompt.strip()
            except (RuntimeError, ValueError, OSError, AttributeError):
                results[idx] = None
        return results
    except (RuntimeError, ValueError, OSError, ImportError):
        return results
    finally:
        cleanup_qwen_prompt_extender(extender, context="join-quads")


__all__ = [
    "DEFAULT_QWEN_VLM_MODEL",
    "SINGLE_IMAGE_FALLBACK_PROMPT",
    "TRANSITION_FALLBACK_PROMPT",
    "cleanup_qwen_prompt_extender",
    "create_qwen_prompt_expander",
    "download_qwen_vlm_if_needed",
    "ensure_wan2gp_on_path",
    "generate_join_quad_prompts",
    "generate_single_image_prompts",
    "generate_transition_pair_prompts",
    "initialize_qwen_prompt_extender",
]
