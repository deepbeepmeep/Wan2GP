"""
VLM-enhanced prompt generation for join clips.

Extracts boundary frames from clips and uses Qwen VLM to generate
context-aware transition prompts.
"""

from pathlib import Path
from typing import Tuple, List, Optional

import cv2

from source.media.video import extract_frames_from_video
from source.media.vlm.service import generate_join_quad_prompts as _service_generate_join_quad_prompts
from source.core.log import orchestrator_logger
from source.utils.download_utils import download_video_if_url

__all__ = [
    "_extract_boundary_frames_for_vlm",
    "_generate_join_transition_prompt",
    "_generate_vlm_prompts_for_joins",
    "extract_boundary_frames_for_vlm",
    "generate_join_transition_prompt",
    "generate_vlm_prompts_for_joins",
]

def _extract_boundary_frames_for_vlm(
    clip_list: List[dict],
    temp_dir: Path,
    orchestrator_task_id: str) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Extract boundary frames from clips for VLM prompt generation.

    For each join (clip[i] -> clip[i+1]), extracts 4 frames:
    - First frame from clip[i] (scene context)
    - Last frame from clip[i] (boundary)
    - First frame from clip[i+1] (boundary)
    - Last frame from clip[i+1] (scene context)

    Args:
        clip_list: List of clip dicts with 'url' keys
        temp_dir: Directory to save temporary frame images
        orchestrator_task_id: Task ID for logging

    Returns:
        List of (start_first, start_boundary, end_boundary, end_last) tuples for each join
    """
    image_quads = []
    num_joins = len(clip_list) - 1

    # Cache downloaded videos and their extracted frames to avoid re-downloading
    clip_data_cache = {}

    for idx in range(num_joins):
        clip_start = clip_list[idx]
        clip_end = clip_list[idx + 1]

        start_url = clip_start.get("url")
        end_url = clip_end.get("url")

        if not start_url or not end_url:
            orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: Missing URL, skipping")
            image_quads.append((None, None, None, None))
            continue

        try:
            # === Extract frames from clip_start ===
            if start_url in clip_data_cache:
                start_frames, start_first_path, start_last_path = clip_data_cache[start_url]
            else:
                orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: Downloading clip_start: {start_url[:80]}...")
                local_start_path = download_video_if_url(
                    start_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx}_start"
                )

                start_frames = extract_frames_from_video(local_start_path)
                if not start_frames:
                    orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: Failed to extract frames from start clip")
                    image_quads.append((None, None, None, None))
                    continue

                start_first_path = temp_dir / f"vlm_clip{idx}_first.jpg"
                start_last_path = temp_dir / f"vlm_clip{idx}_last.jpg"
                cv2.imwrite(str(start_first_path), cv2.cvtColor(start_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(start_last_path), cv2.cvtColor(start_frames[-1], cv2.COLOR_RGB2BGR))

                clip_data_cache[start_url] = (start_frames, str(start_first_path), str(start_last_path))

            start_boundary_idx = len(start_frames) - 1
            start_boundary_path = temp_dir / f"vlm_clip{idx}_boundary.jpg"
            cv2.imwrite(str(start_boundary_path), cv2.cvtColor(start_frames[start_boundary_idx], cv2.COLOR_RGB2BGR))

            # === Extract frames from clip_end ===
            if end_url in clip_data_cache:
                end_frames, end_first_path, end_last_path = clip_data_cache[end_url]
            else:
                orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: Downloading clip_end: {end_url[:80]}...")
                local_end_path = download_video_if_url(
                    end_url,
                    download_target_dir=temp_dir,
                    task_id_for_logging=orchestrator_task_id,
                    descriptive_name=f"clip_{idx+1}_end"
                )

                end_frames = extract_frames_from_video(local_end_path)
                if not end_frames:
                    orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: Failed to extract frames from end clip")
                    image_quads.append((None, None, None, None))
                    continue

                end_first_path = temp_dir / f"vlm_clip{idx+1}_first.jpg"
                end_last_path = temp_dir / f"vlm_clip{idx+1}_last.jpg"
                cv2.imwrite(str(end_first_path), cv2.cvtColor(end_frames[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(end_last_path), cv2.cvtColor(end_frames[-1], cv2.COLOR_RGB2BGR))

                clip_data_cache[end_url] = (end_frames, str(end_first_path), str(end_last_path))

            end_boundary_idx = 0
            end_boundary_path = temp_dir / f"vlm_clip{idx+1}_boundary.jpg"
            cv2.imwrite(str(end_boundary_path), cv2.cvtColor(end_frames[end_boundary_idx], cv2.COLOR_RGB2BGR))

            orchestrator_logger.debug_block(
                "VLM_EXTRACT",
                {
                    "join_idx": idx,
                    "start_frames": len(start_frames),
                    "end_frames": len(end_frames),
                },
                task_id=orchestrator_task_id,
            )

            image_quads.append((
                str(clip_data_cache[start_url][1]),
                str(start_boundary_path),
                str(end_boundary_path),
                str(clip_data_cache[end_url][2])
            ))
            orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: 4 frames ready (first, boundary, boundary, last)")

        except (OSError, ValueError, RuntimeError) as e:
            orchestrator_logger.debug_anomaly("VLM_EXTRACT", f"Join {idx}: ERROR extracting frames: {e}")
            image_quads.append((None, None, None, None))

    return image_quads

def _generate_join_transition_prompt(
    start_first_path: str,
    start_boundary_path: str,
    end_boundary_path: str,
    end_last_path: str,
    base_prompt: str,
    extender) -> str:
    """
    Generate a single transition prompt for join_clips using 4 boundary images.

    Args:
        start_first_path: Path to first frame of starting clip
        start_boundary_path: Path to last frame of starting clip
        end_boundary_path: Path to first frame of ending clip
        end_last_path: Path to last frame of ending clip
        base_prompt: Base prompt for context
        extender: QwenPromptExpander instance

    Returns:
        Generated prompt
    """
    from PIL import Image

    start_first = Image.open(start_first_path).convert("RGB")
    start_boundary = Image.open(start_boundary_path).convert("RGB")
    end_boundary = Image.open(end_boundary_path).convert("RGB")
    end_last = Image.open(end_last_path).convert("RGB")

    images = [start_first, start_boundary, end_boundary, end_last]
    combined_width = sum(img.width for img in images)
    combined_height = max(img.height for img in images)
    combined_img = Image.new('RGB', (combined_width, combined_height))

    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    base_prompt_text = base_prompt if base_prompt and base_prompt.strip() else "a video sequence"

    query = f"""Look at these 4 frames from a video, left to right in time order.

The MIDDLE TWO frames show where we need to generate a transition.

Context from user: {base_prompt_text}

Write a short video prompt describing what happens between the middle frames. Include:
1. The motion or action
2. The visual style
3. Key details in the scene

Keep it under 50 words total. Just write the prompt, nothing else.

Example: "A cat jumps from table to chair. Realistic home video style. Sunny kitchen with wooden furniture."

Example: "Camera pans right across city skyline. Cinematic drone footage at sunset. Skyscrapers with orange light reflections."

Your prompt:"""

    system_prompt = "Write a short video description prompt. Under 50 words. No explanations, just the prompt."

    result = extender.extend_with_img(
        prompt=query,
        system_prompt=system_prompt,
        image=combined_img
    )

    return result.prompt.strip()

def _generate_vlm_prompts_for_joins(
    image_quads: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    base_prompt: str,
    vlm_device: str) -> List[Optional[str]]:
    """
    Generate VLM-enhanced prompts for ALL joins using 4-image quads.

    Args:
        image_quads: List of (start_first, start_boundary, end_boundary, end_last) tuples
        base_prompt: Base prompt to use as VLM context
        vlm_device: Device for VLM inference ('cuda' or 'cpu')

    Returns:
        List of enhanced prompts (None for joins with missing image quads)
    """
    return _service_generate_join_quad_prompts(
        image_quads=image_quads,
        base_prompt=base_prompt,
        device=vlm_device,
    )


# Public aliases for cross-module use.
extract_boundary_frames_for_vlm = _extract_boundary_frames_for_vlm
generate_join_transition_prompt = _generate_join_transition_prompt
generate_vlm_prompts_for_joins = _generate_vlm_prompts_for_joins
