"""Transition prompt generation using the shared VLM service."""

from typing import List, Optional, Tuple

from source.core.log import headless_logger, model_logger
from source.media.vlm.service import generate_transition_pair_prompts


def generate_transition_prompt(
    start_image_path: str,
    end_image_path: str,
    base_prompt: Optional[str] = None,
    device: str = "cuda",
) -> str:
    """
    Generate a descriptive prompt for a transition between two images.

    Existing callers expect a single prompt and fallback to the supplied base
    prompt, or a generic transition prompt when no base prompt exists.
    """
    try:
        prompts = generate_transition_prompts_batch(
            image_pairs=[(start_image_path, end_image_path)],
            base_prompts=[base_prompt],
            device=device,
            task_id=None,
            upload_debug_images=False,
        )
        if prompts:
            return prompts[0]
    except (RuntimeError, ValueError, OSError, ImportError) as e:
        model_logger.error(f"[VLM_TRANSITION] ERROR: Failed to generate transition prompt: {e}", exc_info=True)

    if base_prompt and base_prompt.strip():
        model_logger.debug_anomaly("VLM_TRANSITION", f"Falling back to base prompt: {base_prompt}")
        return base_prompt

    model_logger.debug_anomaly("VLM_TRANSITION", "Falling back to generic prompt")
    return "cinematic transition"


def generate_transition_prompts_batch(
    image_pairs: List[Tuple[str, str]],
    base_prompts: List[Optional[str]],
    device: str = "cuda",
    task_id: Optional[str] = None,
    upload_debug_images: bool = True,
) -> List[str]:
    """
    Batch generate transition prompts for multiple image pairs.

    `task_id` and `upload_debug_images` are retained for API compatibility with
    travel callers; the backend-neutral service owns the VLM lifecycle.
    """
    del task_id, upload_debug_images
    return generate_transition_pair_prompts(
        image_pairs=image_pairs,
        base_prompts=base_prompts,
        device=device,
    )


def test_vlm_transition():
    """Test function for VLM transition prompt generation."""
    headless_logger.essential("\n" + "=" * 80)
    headless_logger.essential("Testing VLM Transition Prompt Generation")
    headless_logger.essential("=" * 80 + "\n")

    headless_logger.essential("To test, call:")
    headless_logger.essential("  generate_transition_prompt('path/to/start.jpg', 'path/to/end.jpg')")
    headless_logger.essential("\nExample usage in travel orchestrator:")
    headless_logger.essential("  if orchestrator_payload.get('enhance_prompt', False):")
    headless_logger.essential("      prompt = generate_transition_prompt(start_img, end_img, base_prompt)")


if __name__ == "__main__":
    test_vlm_transition()
