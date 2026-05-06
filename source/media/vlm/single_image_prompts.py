"""Single-image prompt generation using the shared VLM service."""

from typing import List, Optional

from source.media.vlm.service import generate_single_image_prompts


def generate_single_image_prompt(
    image_path: str,
    base_prompt: Optional[str] = None,
    device: str = "cuda",
) -> str:
    """
    Generate a descriptive prompt based on a single image.

    Existing callers expect a single prompt and fallback behavior is provided by
    the shared service.
    """
    prompts = generate_single_image_prompts(
        image_paths=[image_path],
        base_prompts=[base_prompt],
        device=device,
    )
    if prompts:
        return prompts[0]
    return base_prompt if base_prompt and base_prompt.strip() else "cinematic video"


def generate_single_image_prompts_batch(
    image_paths: List[str],
    base_prompts: List[Optional[str]],
    device: str = "cuda",
) -> List[str]:
    """
    Batch generate prompts for multiple single images.

    The backend-neutral service owns VLM initialization, inference, cleanup, and
    fallback semantics.
    """
    return generate_single_image_prompts(
        image_paths=image_paths,
        base_prompts=base_prompts,
        device=device,
    )
