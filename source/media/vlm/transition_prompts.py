"""Transition prompt generation using VLM for image-pair transitions."""

from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import torch

from source.core.constants import BYTES_PER_GB
from source.core.log import headless_logger, model_logger
from source.media.vlm.image_prep import create_framed_vlm_image, create_labeled_debug_image
from source.media.vlm.model import download_qwen_vlm_if_needed
from source.runtime.wgp_bridge import create_qwen_prompt_expander


def generate_transition_prompt(
    start_image_path: str,
    end_image_path: str,
    base_prompt: Optional[str] = None,
    device: str = "cuda",
) -> str:
    """
    Use QwenVLM to generate a descriptive prompt for the transition between two images.

    The model is automatically offloaded to CPU after inference to free VRAM.

    Args:
        start_image_path: Path to the starting image
        end_image_path: Path to the ending image
        base_prompt: Optional base prompt to append after VLM-generated description
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Generated prompt describing the transition, with base_prompt appended if provided
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
        model_logger.debug(f"[VLM_TRANSITION] Falling back to base prompt: {base_prompt}")
        return base_prompt

    model_logger.debug(f"[VLM_TRANSITION] Falling back to generic prompt")
    return "cinematic transition"


def generate_transition_prompts_batch(
    image_pairs: List[Tuple[str, str]],
    base_prompts: List[Optional[str]],
    device: str = "cuda",
    task_id: Optional[str] = None,
    upload_debug_images: bool = True
) -> List[str]:
    """
    Batch generate transition prompts for multiple image pairs.

    This is much more efficient than calling generate_transition_prompt() multiple times,
    because it loads the VLM model once and reuses it for all pairs.

    Args:
        image_pairs: List of (start_image_path, end_image_path) tuples
        base_prompts: List of base prompts to append (one per pair, can be None)
        device: Device to run the model on ('cuda' or 'cpu')
        task_id: Optional task ID for organizing debug image uploads
        upload_debug_images: Whether to upload debug combined images to storage (default: True)

    Returns:
        List of generated prompts (one per image pair)
    """
    if len(image_pairs) != len(base_prompts):
        raise ValueError(f"image_pairs and base_prompts must have same length ({len(image_pairs)} != {len(base_prompts)})")

    if not image_pairs:
        return []

    try:
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"

        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / BYTES_PER_GB
            model_logger.debug(f"[VLM_BATCH] GPU memory BEFORE: {gpu_mem_before:.2f} GB")

        model_logger.debug(f"[VLM_BATCH] Initializing Qwen2.5-VL-7B-Instruct for {len(image_pairs)} transitions...")

        local_model_path = wan_dir / "ckpts" / "Qwen2.5-VL-7B-Instruct"

        model_logger.debug(f"[VLM_BATCH] Checking for model at {local_model_path}...")
        download_qwen_vlm_if_needed(local_model_path)

        model_logger.debug(f"[VLM_BATCH] Using local model from: {local_model_path}")
        extender = create_qwen_prompt_expander(
            model_name=str(local_model_path),
            device=device,
            is_vl=True
        )

        model_logger.debug(f"[VLM_BATCH] Model loaded (initially on CPU)")

        system_prompt = "You are a video direction assistant. You MUST respond with EXACTLY THREE SENTENCES following this structure: 1) PRIMARY MOTION, 2) MOVING ELEMENTS, 3) MOTION DETAILS. Focus exclusively on what moves and changes, not static descriptions."

        results = []
        prev_end_hash = None
        for i, ((start_path, end_path), base_prompt) in enumerate(zip(image_pairs, base_prompts)):
            try:
                model_logger.debug(f"[VLM_BATCH] Processing pair {i+1}/{len(image_pairs)}: {Path(start_path).name} -> {Path(end_path).name}")

                # File debug logging
                start_exists = Path(start_path).exists() if start_path else False
                end_exists = Path(end_path).exists() if end_path else False
                model_logger.debug(f"[VLM_FILE_DEBUG] Pair {i}: start={start_path} (exists={start_exists})")
                model_logger.debug(f"[VLM_FILE_DEBUG] Pair {i}: end={end_path} (exists={end_exists})")

                # Compute file hashes for identity verification
                import hashlib
                def get_file_hash(filepath):
                    """Get first 8 chars of MD5 hash for quick file identity check."""
                    try:
                        with open(filepath, 'rb') as f:
                            return hashlib.md5(f.read()).hexdigest()[:8]
                    except OSError:
                        return 'ERROR'

                start_hash = None
                end_hash = None
                if start_exists:
                    start_size = Path(start_path).stat().st_size
                    start_hash = get_file_hash(start_path)
                    model_logger.debug(f"[VLM_FILE_DEBUG] Pair {i}: start file size={start_size} bytes, hash={start_hash}")
                if end_exists:
                    end_size = Path(end_path).stat().st_size
                    end_hash = get_file_hash(end_path)
                    model_logger.debug(f"[VLM_FILE_DEBUG] Pair {i}: end file size={end_size} bytes, hash={end_hash}")

                # Boundary check
                if i > 0 and prev_end_hash and start_hash:
                    if prev_end_hash == start_hash:
                        model_logger.debug(f"[VLM_BOUNDARY_CHECK] Pair {i} start matches Pair {i-1} end (hash={start_hash}) - boundary correct")
                    else:
                        model_logger.warning(f"[VLM_BOUNDARY_CHECK] MISMATCH! Pair {i} start (hash={start_hash}) != Pair {i-1} end (hash={prev_end_hash})")
                        model_logger.warning(f"[VLM_BOUNDARY_CHECK] This indicates images may be out of order or corrupted!")

                prev_end_hash = end_hash

                # Load and combine images
                start_img = Image.open(start_path).convert("RGB")
                end_img = Image.open(end_path).convert("RGB")

                model_logger.debug(f"[VLM_IMAGE_VERIFY] Pair {i}: start_img dimensions={start_img.size}, end_img dimensions={end_img.size}")

                import numpy as np
                def get_image_stats(img):
                    """Get average RGB and warmth indicator for image."""
                    arr = np.array(img)
                    avg_r, avg_g, avg_b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
                    warmth = (avg_r - avg_b) / 255 * 100
                    brightness = (avg_r + avg_g + avg_b) / 3
                    return f"RGB=({avg_r:.0f},{avg_g:.0f},{avg_b:.0f}) brightness={brightness:.0f} warmth={warmth:+.1f}%"

                model_logger.debug(f"[VLM_IMAGE_CONTENT] Pair {i} START: {get_image_stats(start_img)}")
                model_logger.debug(f"[VLM_IMAGE_CONTENT] Pair {i} END: {get_image_stats(end_img)}")

                combined_img = create_framed_vlm_image(start_img, end_img)

                # Save debug images
                debug_path = None
                start_debug_path = None
                end_debug_path = None
                try:
                    debug_dir = Path(start_path).parent / "vlm_debug"
                    debug_dir.mkdir(exist_ok=True)

                    labeled_debug_img = create_labeled_debug_image(start_img, end_img, pair_index=i)
                    debug_path = debug_dir / f"vlm_combined_pair{i}.jpg"
                    labeled_debug_img.save(str(debug_path), quality=95)
                    model_logger.debug(f"[VLM_DEBUG_SAVE] Saved labeled debug image for pair {i} to: {debug_path}")

                    start_debug_path = debug_dir / f"vlm_pair{i}_LEFT_start.jpg"
                    end_debug_path = debug_dir / f"vlm_pair{i}_RIGHT_end.jpg"
                    start_img.save(str(start_debug_path), quality=95)
                    end_img.save(str(end_debug_path), quality=95)
                    model_logger.debug(f"[VLM_DEBUG_SAVE] Saved individual images: {start_debug_path.name}, {end_debug_path.name}")
                except OSError as e_save:
                    model_logger.debug(f"[VLM_DEBUG_SAVE] Could not save debug image: {e_save}")

                # Upload debug images
                if upload_debug_images and task_id and debug_path and debug_path.exists():
                    try:
                        from source.utils.output_paths import upload_intermediate_file_to_storage

                        upload_filename = f"vlm_debug_pair{i}_combined.jpg"
                        upload_url = upload_intermediate_file_to_storage(
                            debug_path,
                            task_id,
                            upload_filename,
                        )
                        if upload_url:
                            model_logger.debug(f"[VLM_DEBUG_UPLOAD] Pair {i} COMBINED (what VLM sees): {upload_url}")
                        else:
                            model_logger.warning(f"[VLM_DEBUG_UPLOAD] Failed to upload combined image for pair {i}")

                        if start_debug_path and start_debug_path.exists():
                            start_url = upload_intermediate_file_to_storage(
                                start_debug_path, task_id, f"vlm_debug_pair{i}_LEFT.jpg"
                            )
                            if start_url:
                                model_logger.debug(f"[VLM_DEBUG_UPLOAD] Pair {i} LEFT (start image): {start_url}")

                        if end_debug_path and end_debug_path.exists():
                            end_url = upload_intermediate_file_to_storage(
                                end_debug_path, task_id, f"vlm_debug_pair{i}_RIGHT.jpg"
                            )
                            if end_url:
                                model_logger.debug(f"[VLM_DEBUG_UPLOAD] Pair {i} RIGHT (end image): {end_url}")

                    except (OSError, RuntimeError, ValueError) as e_upload:
                        model_logger.warning(f"[VLM_DEBUG_UPLOAD] Failed to upload debug images for pair {i}: {e_upload}")

                base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "the objects/people inside the scene move excitingly and things transform or shift with the camera"

                query = f"""You are viewing two images side by side: the LEFT image (with GREEN border) shows the STARTING frame, and the RIGHT image (with RED border) shows the ENDING frame of a video sequence.

Your goal is to create a THREE-SENTENCE prompt that describes the MOTION and CHANGES in this transition based on the user's description: '{base_prompt_text}'

FOCUS ON MOTION: Describe what MOVES, what CHANGES, and HOW things transition between these frames. Everything should be described in terms of motion and transformation, not static states.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE:

SENTENCE 1 (PRIMARY MOTION): Describe the main action, camera movement, and major scene transitions. What is the dominant movement happening?

SENTENCE 2 (MOVING ELEMENTS): Describe how the characters, objects, and environment are moving or changing. Focus on what's in motion and how it moves through space.

SENTENCE 3 (MOTION DETAILS): Describe the subtle motion details - secondary movements, environmental dynamics, particles, lighting shifts, and small-scale motions.

Examples of MOTION-FOCUSED descriptions:

- "The sun rises rapidly above the jagged peaks as the camera tilts upward from the dark valley floor. The silhouette pine trees sway gently against the shifting violet and gold sky as the entire landscape brightens. Wisps of morning mist evaporate and drift upward from the river surface while distant birds circle and glide through the upper left corner."

- "A woman sprints from the kitchen into the bright exterior sunlight as the camera pans right to track her accelerating path. Her vintage floral dress flows and ripples in the wind while colorful playground equipment blurs past in the background. Her hair whips back dynamically and dust particles kick up and swirl around her sneakers as she impacts the gravel."

- "The camera zooms aggressively inward into a macro shot of an eye as the brown horse reflection grows larger and more detailed. The iris textures shift under the changing warm lighting while the biological details come into sharper focus. The pupil constricts and contracts in reaction to the light while the tiny reflected horse tosses its mane and shifts position."

Now create your THREE-SENTENCE MOTION-FOCUSED description based on: '{base_prompt_text}'"""

                model_logger.debug(f"[VLM_QUERY_DEBUG] Pair {i}: base_prompt_text='{base_prompt_text[:100]}...'")

                result = extender.extend_with_img(
                    prompt=query,
                    system_prompt=system_prompt,
                    image=combined_img
                )

                vlm_prompt = result.prompt.strip()
                model_logger.debug(f"[VLM_BATCH] Generated: {vlm_prompt}")

                results.append(vlm_prompt)

            except (RuntimeError, ValueError, OSError) as e:
                model_logger.error(f"[VLM_BATCH] ERROR processing pair {i+1}: {e}")
                if base_prompt and base_prompt.strip():
                    results.append(base_prompt)
                else:
                    results.append("cinematic transition")

        model_logger.debug(f"[VLM_BATCH] Completed {len(results)}/{len(image_pairs)} prompts")

        if torch.cuda.is_available():
            gpu_mem_before_cleanup = torch.cuda.memory_allocated() / BYTES_PER_GB
            headless_logger.debug(f"[VLM_CLEANUP] GPU memory BEFORE cleanup: {gpu_mem_before_cleanup:.2f} GB")

        headless_logger.debug(f"[VLM_CLEANUP] Cleaning up VLM model and processor...")
        try:
            del extender.model
            del extender.processor
            del extender
            headless_logger.essential(f"[VLM_CLEANUP] Successfully deleted VLM objects")
        except AttributeError as e:
            headless_logger.warning(f"[VLM_CLEANUP] Error during deletion: {e}")

        import gc
        collected = gc.collect()
        headless_logger.debug(f"[VLM_CLEANUP] Garbage collected {collected} objects")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / BYTES_PER_GB
            gpu_freed = gpu_mem_before_cleanup - gpu_mem_after
            headless_logger.debug(f"[VLM_CLEANUP] GPU memory AFTER cleanup: {gpu_mem_after:.2f} GB")
            headless_logger.debug(f"[VLM_CLEANUP] GPU memory freed: {gpu_freed:.2f} GB")

        headless_logger.essential(f"[VLM_CLEANUP] VLM cleanup complete")

        return results

    except (RuntimeError, ValueError, OSError, ImportError) as e:
        model_logger.error(f"[VLM_BATCH] CRITICAL ERROR: {e}", exc_info=True)
        return [bp if bp and bp.strip() else "cinematic transition" for bp in base_prompts]


def test_vlm_transition():
    """Test function for VLM transition prompt generation."""
    headless_logger.essential("\n" + "="*80)
    headless_logger.essential("Testing VLM Transition Prompt Generation")
    headless_logger.essential("="*80 + "\n")

    headless_logger.essential("To test, call:")
    headless_logger.essential("  generate_transition_prompt('path/to/start.jpg', 'path/to/end.jpg')")
    headless_logger.essential("\nExample usage in travel orchestrator:")
    headless_logger.essential("  if orchestrator_payload.get('enhance_prompt', False):")
    headless_logger.essential("      prompt = generate_transition_prompt(start_img, end_img, base_prompt)")


if __name__ == "__main__":
    test_vlm_transition()
