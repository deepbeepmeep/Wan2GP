"""Single-image prompt generation using VLM."""

from pathlib import Path
from typing import Optional, List
from PIL import Image
import torch

from source.core.constants import BYTES_PER_GB
from source.core.log import headless_logger, model_logger
from source.media.vlm.model import download_qwen_vlm_if_needed
from source.runtime.wgp_bridge import create_qwen_prompt_expander


def generate_single_image_prompt(
    image_path: str,
    base_prompt: Optional[str] = None,
    device: str = "cuda",
) -> str:
    """
    Use QwenVLM to generate a descriptive prompt based on a single image.

    This is used for single-image video generation where there's no transition
    between images - instead, we describe the image and suggest natural motion.

    Args:
        image_path: Path to the image
        base_prompt: Optional base prompt to incorporate
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Generated prompt describing the image and suggesting motion
    """
    try:
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"

        model_logger.debug_anomaly("VLM_SINGLE", f"Generating prompt for single image: {Path(image_path).name}")

        img = Image.open(image_path).convert("RGB")

        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        model_logger.debug_anomaly("VLM_SINGLE", f"Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        model_logger.debug_anomaly("VLM_SINGLE", f"Initializing Qwen2.5-VL-7B-Instruct from local path: {local_model_path}")
        extender = create_qwen_prompt_expander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True
        )

        base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

        query = f"""You are viewing a single image that will be the starting frame of a video sequence.

Your goal is to create a THREE-SENTENCE prompt that describes the image and suggests NATURAL MOTION based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what's in the image and how things could naturally move, animate, or change. Everything should suggest dynamic motion from this starting point.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (SCENE & CAMERA): Describe the scene and suggest camera movement (pan, zoom, tilt, tracking shot, etc.). What would a cinematographer do?

SENTENCE 2 (SUBJECT MOTION): Describe the main subjects and how they could naturally move or animate. People breathe, blink, shift weight; animals move; plants sway; water flows.

SENTENCE 3 (ENVIRONMENTAL DYNAMICS): Describe ambient motion - wind effects, lighting changes, particles, atmospheric effects, subtle movements that bring the scene to life.

Examples of MOTION-FOCUSED single-image descriptions:

- "The camera slowly pushes forward through the misty forest as the early morning light filters through the canopy. The tall pine trees sway gently in the breeze while a deer in the clearing lifts its head alertly and flicks its ears. Dust motes drift lazily through the golden light beams and fallen leaves rustle and tumble across the forest floor."

- "The camera tracks slowly around the woman as she stands at the window gazing out at the city. She shifts her weight slightly and turns her head, her hair catching the warm light from the sunset outside. The curtains billow softly in a gentle breeze while city lights begin twinkling on in the darkening skyline."

- "The camera zooms gradually into the vintage car parked on the empty desert road as heat waves shimmer off the asphalt. Chrome details on the car glint and sparkle as the harsh sun shifts position overhead. Tumbleweeds roll slowly across the cracked pavement while sand particles drift on the hot wind."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) SCENE & CAMERA, 2) SUBJECT MOTION, 3) ENVIRONMENTAL DYNAMICS. Focus on natural motion that could emerge from this single image."

        model_logger.debug_anomaly("VLM_SINGLE", f"Running inference...")
        result = extender.extend_with_img(
            prompt=query,
            system_prompt=system_prompt,
            image=img
        )

        vlm_prompt = result.prompt.strip()
        model_logger.debug_block(
            "VLM_RESULT",
            {
                "prompts": 1,
                "avg_length": len(vlm_prompt),
            },
        )

        # Cleanup
        try:
            del extender.model
            del extender.processor
            del extender
        except AttributeError as e:
            headless_logger.warning(f"[VLM_SINGLE] Could not delete VLM objects during cleanup: {e}")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return vlm_prompt

    except (RuntimeError, ValueError, OSError, ImportError) as e:
        model_logger.error(f"[VLM_SINGLE] ERROR: Failed to generate single image prompt: {e}", exc_info=True)

        if base_prompt and base_prompt.strip():
            model_logger.debug_anomaly("VLM_SINGLE", f"Falling back to base prompt: {base_prompt}")
            return base_prompt
        else:
            model_logger.debug_anomaly("VLM_SINGLE", f"Falling back to generic prompt")
            return "cinematic video"


def generate_single_image_prompts_batch(
    image_paths: List[str],
    base_prompts: List[Optional[str]],
    device: str = "cuda",
) -> List[str]:
    """
    Batch generate prompts for multiple single images.

    This is more efficient than calling generate_single_image_prompt() multiple times,
    because it loads the VLM model once and reuses it for all images.

    Args:
        image_paths: List of image paths
        base_prompts: List of base prompts (one per image, can be None)
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        List of generated prompts (one per image)
    """
    if len(image_paths) != len(base_prompts):
        raise ValueError(f"image_paths and base_prompts must have same length ({len(image_paths)} != {len(base_prompts)})")

    if not image_paths:
        return []

    try:
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"

        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / BYTES_PER_GB
            model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"GPU memory BEFORE: {gpu_mem_before:.2f} GB")

        model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Initializing Qwen2.5-VL-7B-Instruct for {len(image_paths)} single images...")

        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Using local model from: {local_model_path}")
        extender = create_qwen_prompt_expander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True
        )

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) SCENE & CAMERA, 2) SUBJECT MOTION, 3) ENVIRONMENTAL DYNAMICS. Focus on natural motion that could emerge from this single image."

        results = []
        for i, (image_path, base_prompt) in enumerate(zip(image_paths, base_prompts)):
            try:
                model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")

                img = Image.open(image_path).convert("RGB")
                model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Image {i}: dimensions={img.size}")

                base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

                query = f"""You are viewing a single image that will be the starting frame of a video sequence.

Your goal is to create a THREE-SENTENCE prompt that describes the image and suggests NATURAL MOTION based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what's in the image and how things could naturally move, animate, or change. Everything should suggest dynamic motion from this starting point.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (SCENE & CAMERA): Describe the scene and suggest camera movement (pan, zoom, tilt, tracking shot, etc.). What would a cinematographer do?

SENTENCE 2 (SUBJECT MOTION): Describe the main subjects and how they could naturally move or animate. People breathe, blink, shift weight; animals move; plants sway; water flows.

SENTENCE 3 (ENVIRONMENTAL DYNAMICS): Describe ambient motion - wind effects, lighting changes, particles, atmospheric effects, subtle movements that bring the scene to life.

Examples of MOTION-FOCUSED single-image descriptions:

- "The camera slowly pushes forward through the misty forest as the early morning light filters through the canopy. The tall pine trees sway gently in the breeze while a deer in the clearing lifts its head alertly and flicks its ears. Dust motes drift lazily through the golden light beams and fallen leaves rustle and tumble across the forest floor."

- "The camera tracks slowly around the woman as she stands at the window gazing out at the city. She shifts her weight slightly and turns her head, her hair catching the warm light from the sunset outside. The curtains billow softly in a gentle breeze while city lights begin twinkling on in the darkening skyline."

- "The camera zooms gradually into the vintage car parked on the empty desert road as heat waves shimmer off the asphalt. Chrome details on the car glint and sparkle as the harsh sun shifts position overhead. Tumbleweeds roll slowly across the cracked pavement while sand particles drift on the hot wind."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

                result = extender.extend_with_img(
                    prompt=query,
                    system_prompt=system_prompt,
                    image=img
                )

                vlm_prompt = result.prompt.strip()
                model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Generated: {vlm_prompt}")

                results.append(vlm_prompt)

            except (RuntimeError, ValueError, OSError) as e:
                model_logger.error(f"[VLM_SINGLE_BATCH] ERROR processing image {i+1}: {e}")
                if base_prompt and base_prompt.strip():
                    results.append(base_prompt)
                else:
                    results.append("cinematic video")

        model_logger.debug_anomaly("VLM_SINGLE_BATCH", f"Completed {len(results)}/{len(image_paths)} prompts")
        avg_length = sum(len(prompt) for prompt in results) / len(results) if results else 0
        model_logger.debug_block(
            "VLM_RESULT",
            {
                "prompts": len(results),
                "avg_length": round(avg_length, 1),
            },
        )

        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / BYTES_PER_GB
            headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", f"GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")

        headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", f"Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", "Successfully deleted VLM objects")
        except AttributeError as e:
            headless_logger.warning(f"[VLM_SINGLE_CLEANUP] Error during deletion: {e}")

        import gc
        collected = gc.collect()
        headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", f"Garbage collected {collected} objects")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / BYTES_PER_GB
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", f"GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", f"GPU memory freed: {gpu_freed:.2f} GB")

        headless_logger.debug_anomaly("VLM_SINGLE_CLEANUP", "VLM cleanup complete")

        return results

    except (RuntimeError, ValueError, OSError, ImportError) as e:
        model_logger.error(f"[VLM_SINGLE_BATCH] CRITICAL ERROR: {e}", exc_info=True)
        return [bp if bp and bp.strip() else "cinematic video" for bp in base_prompts]
