"""Travel-specific video operations: RIFE interpolation, VACE ref prep, guide video creation.

Typed request adaptation for these helpers lives in
``source.task_handlers.travel.guidance.guide_video_ops``.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from source.core.log import generation_logger
from source.runtime.wgp_bridge import run_rife_temporal_interpolation
from source.utils.download_utils import download_image_if_url
from source.utils.frame_utils import (
    apply_strength_to_image,
    create_color_frame,
    get_easing_function,
    get_sequential_target_path,
    image_to_frame,
)
from source.utils.output_paths import wait_for_file_stable
from source.media.video.video_transforms import adjust_frame_brightness
from source.core.params.structure_guidance import StructureGuidanceConfig
from source.media.video.frame_extraction import extract_frames_from_video
from source.media.video.video_info import get_video_frame_count_and_fps
from source.media.video.ffmpeg_ops import create_video_from_frames_list

__all__ = [
    "rife_interpolate_images_to_video",
    "prepare_vace_ref_for_segment",
    "create_guide_video_for_travel_segment",
]


def rife_interpolate_images_to_video(
    image1: Image.Image,
    image2: Image.Image,
    num_frames: int,
    resolution_wh: tuple[int, int],
    output_path: str | Path,
    fps: int = 16,
) -> bool:
    """
    Interpolates between two PIL images using RIFE to generate a video.
    """
    try:
        generation_logger.debug("Imported RIFE modules for interpolation.")

        width_out, height_out = resolution_wh
        generation_logger.debug(f"Parsed resolution: {width_out}x{height_out}")

        def pil_to_tensor_rgb_norm(pil_im: Image.Image):
            pil_resized = pil_im.resize((width_out, height_out), Image.Resampling.LANCZOS)
            np_rgb = np.asarray(pil_resized).astype(np.float32) / 127.5 - 1.0  # [0,255]->[-1,1]
            tensor = torch.from_numpy(np_rgb).permute(2, 0, 1)  # C H W
            return tensor

        t_start = pil_to_tensor_rgb_norm(image1)
        t_end   = pil_to_tensor_rgb_norm(image2)

        sample_in = torch.stack([t_start, t_end], dim=1).unsqueeze(0)  # 1 x 3 x 2 x H x W

        device_for_rife = "cuda" if torch.cuda.is_available() else "cpu"
        sample_in = sample_in.to(device_for_rife)
        generation_logger.debug(f"Input tensor for RIFE prepared on device: {device_for_rife}, shape: {sample_in.shape}")

        exp_val = 3  # x8 (2^3 + 1 = 9 frames output by this RIFE implementation for 2 inputs)
        flownet_ckpt = os.path.join("ckpts", "flownet.pkl")
        generation_logger.debug(f"Checking for RIFE model: {flownet_ckpt}")
        if not os.path.exists(flownet_ckpt):
            generation_logger.error(f"RIFE Error: flownet.pkl not found at {flownet_ckpt}")
            return False
        generation_logger.debug(f"RIFE model found: {flownet_ckpt}. Exp_val: {exp_val}")

        sample_in_for_rife = sample_in[0]

        sample_out_from_rife = run_rife_temporal_interpolation(
            flownet_ckpt,
            sample_in_for_rife,
            exp_val,
            device=device_for_rife,
        )

        if sample_out_from_rife is None:
            generation_logger.error("RIFE process returned None.")
            return False

        generation_logger.debug(f"RIFE output tensor shape: {sample_out_from_rife.shape}")

        sample_out_no_batch = sample_out_from_rife.to("cpu")
        total_frames_generated = sample_out_no_batch.shape[1]
        generation_logger.debug(f"RIFE produced {total_frames_generated} frames.")

        if total_frames_generated < num_frames:
            generation_logger.warning(f"RIFE produced {total_frames_generated} frames, expected {num_frames}. Padding last frame.")
            pad_frames = num_frames - total_frames_generated
        else:
            pad_frames = 0

        frames_list_np = []
        for idx in range(min(num_frames, total_frames_generated)):
            frame_tensor = sample_out_no_batch[:, idx]
            frame_np = ((frame_tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frames_list_np.append(frame_bgr)

        if pad_frames > 0 and frames_list_np:
            last_frame_to_pad = frames_list_np[-1].copy()
            frames_list_np.extend([last_frame_to_pad for _ in range(pad_frames)])

        if not frames_list_np:
            generation_logger.error(f"No frames available to write for RIFE video (num_rife_frames: {num_frames}).")
            return False

        output_path_obj = Path(output_path)
        video_written = create_video_from_frames_list(frames_list_np, output_path_obj, fps, resolution_wh)
        generation_logger.debug(f"RIFE video saved to: {video_written.resolve()}")
        return True

    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"RIFE interpolation failed with exception: {e}", exc_info=True)
        return False

def prepare_vace_ref_for_segment(
    ref_instruction: dict,
    segment_processing_dir: Path,
    target_resolution_wh: tuple[int, int] | None,
    image_download_dir: Path | str | None = None,
    task_id_for_logging: str | None = "generic_headless_task"
) -> Path | None:
    '''
    Prepares a VACE reference image for a segment based on the given instruction.
    Downloads the image if 'original_path' is a URL and image_download_dir is provided.
    Applies strength adjustment and resizes, saving the result to segment_processing_dir.
    Returns the path to the processed image if successful, or None otherwise.
    '''
    generation_logger.debug(f"Task {task_id_for_logging} (prepare_vace_ref): VACE Ref instruction: {ref_instruction}, download_dir: {image_download_dir}")

    original_image_path_str = ref_instruction.get("original_path")
    strength_to_apply = ref_instruction.get("strength_to_apply")

    if not original_image_path_str:
        generation_logger.debug(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: No original_path in VACE ref instruction. Skipping.")
        return None

    local_original_image_path_str = download_image_if_url(original_image_path_str, image_download_dir, task_id_for_logging)
    local_original_image_path = Path(local_original_image_path_str)

    if not local_original_image_path.exists():
        generation_logger.warning(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: VACE ref original image not found (after potential download): {local_original_image_path} (original input: {original_image_path_str})")
        return None

    vace_ref_type = ref_instruction.get("type", "generic")
    segment_idx_for_naming = ref_instruction.get("segment_idx_for_naming", "unknown_idx")
    processed_vace_base_name = f"vace_ref_s{segment_idx_for_naming}_{vace_ref_type}_str{strength_to_apply:.2f}"
    original_suffix = local_original_image_path.suffix if local_original_image_path.suffix else ".png"

    output_path_for_processed_vace = get_sequential_target_path(segment_processing_dir, processed_vace_base_name, original_suffix)

    effective_target_resolution_wh = None
    if target_resolution_wh:
        effective_target_resolution_wh = ((target_resolution_wh[0] // 16) * 16, (target_resolution_wh[1] // 16) * 16)
        if effective_target_resolution_wh != target_resolution_wh:
            generation_logger.debug(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Adjusted VACE ref target resolution from {target_resolution_wh} to {effective_target_resolution_wh}")

    final_processed_path = apply_strength_to_image(
        image_path_input=local_original_image_path,
        strength=strength_to_apply,
        output_path=output_path_for_processed_vace,
        target_resolution_wh=effective_target_resolution_wh,
        task_id_for_logging=task_id_for_logging,
        image_download_dir=None
    )

    if final_processed_path and final_processed_path.exists():
        generation_logger.debug(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Prepared VACE ref: {final_processed_path}")
        return final_processed_path
    else:
        generation_logger.error(f"Task {task_id_for_logging}, Segment {segment_processing_dir.name}: Failed to apply strength/save VACE ref from {local_original_image_path}. Skipping.", exc_info=True)
        return None

def create_guide_video_for_travel_segment(
    segment_idx_for_logging: int,
    end_anchor_image_index: int,
    is_first_segment_from_scratch: bool,
    total_frames_for_segment: int,
    parsed_res_wh: tuple[int, int],
    fps_helpers: int,
    input_images_resolved_for_guide: list[str],
    path_to_previous_segment_video_output_for_guide: str | None,
    output_target_dir: Path,
    guide_video_base_name: str,
    segment_image_download_dir: Path | None,
    task_id_for_logging: str,
    orchestrator_details: dict,
    segment_params: dict,
    single_image_journey: bool = False,
    predefined_output_path: Path | None = None,
    # Structure guidance - prefer unified config, fall back to individual params
    structure_config: 'StructureGuidanceConfig | None' = None,
    structure_video_path: str | None = None,
    structure_video_treatment: str = "adjust",
    structure_type: str = "flow",
    structure_video_motion_strength: float = 1.0,
    structure_canny_intensity: float = 1.0,
    structure_depth_contrast: float = 1.0,
    structure_guidance_video_url: str | None = None,
    structure_guidance_frame_offset: int = 0,
    # Uni3C end frame exclusion - black out end frame so i2v handles it alone
    exclude_end_for_controlnet: bool = False,
) -> Path | None:
    """Creates the guide video for a travel segment with all fading and adjustments.

    Args:
        structure_config: Unified StructureGuidanceConfig object (preferred).
            If provided, overrides individual structure_* parameters.
        structure_video_path: Legacy - use structure_config instead
        ... (other params remain for backward compat)
    """
    try:
        # If unified config provided, extract values from it
        if structure_config is not None:
            generation_logger.debug_anomaly("GUIDE_VIDEO", f"Using unified StructureGuidanceConfig: {structure_config}")
            structure_video_path = structure_config.videos[0].path if structure_config.videos else None
            structure_video_treatment = structure_config.videos[0].treatment if structure_config.videos else "adjust"
            structure_type = structure_config.legacy_structure_type
            structure_video_motion_strength = structure_config.strength
            structure_canny_intensity = structure_config.canny_intensity
            structure_depth_contrast = structure_config.depth_contrast
            structure_guidance_video_url = structure_config.guidance_video_url
            structure_guidance_frame_offset = structure_config._frame_offset
            exclude_end_for_controlnet = structure_config.is_uni3c

        # Initialize guidance tracker for structure video feature
        from source.media.structure import GuidanceTracker, apply_structure_motion_with_tracking
        guidance_tracker = GuidanceTracker(total_frames_for_segment)
        # Use predefined path if provided (for UUID-based naming), otherwise generate unique path
        if predefined_output_path:
            actual_guide_video_path = predefined_output_path
        else:
            actual_guide_video_path = get_sequential_target_path(output_target_dir, guide_video_base_name, ".mp4")

        # Extract debug mode from orchestrator payload or segment params
        debug_mode = segment_params.get("debug_mode_enabled", orchestrator_details.get("debug_mode_enabled", False))

        gray_frame_bgr = create_color_frame(parsed_res_wh, (128, 128, 128))

        # Hardcoded fade parameters (duration_factor=0.0 means no fading)
        fi_low, fi_high, fi_curve, fi_factor = 0.0, 1.0, "ease_in_out", 0.0
        fo_low, fo_high, fo_curve, fo_factor = 0.0, 1.0, "ease_in_out", 0.0

        strength_adj = segment_params.get("subsequent_starting_strength_adjustment", 0.0)
        desat_factor = segment_params.get("desaturate_subsequent_starting_frames", 0.0)
        bright_adj = segment_params.get("adjust_brightness_subsequent_starting_frames", 0.0)
        frame_overlap_from_previous = segment_params.get("frame_overlap_from_previous", 0)

        if total_frames_for_segment <= 0:
            generation_logger.debug(f"Task {task_id_for_logging}: Guide video has 0 frames. Skipping creation.")
            return None

        frames_for_guide_list = [create_color_frame(parsed_res_wh, (128,128,128)).copy() for _ in range(total_frames_for_segment)]

        # Check for consolidated keyframe positions (frame consolidation optimization)
        consolidated_keyframe_positions = segment_params.get("consolidated_keyframe_positions")
        if consolidated_keyframe_positions and not single_image_journey:
            generation_logger.debug(f"Task {task_id_for_logging}: CONSOLIDATED SEGMENT - placing keyframes at positions {consolidated_keyframe_positions}")

            # For consolidated segments, we need to determine which input images to use
            # If this is the first segment, use images sequentially starting from 0
            # If this is a subsequent segment, only place the final target image (end anchor)
            if is_first_segment_from_scratch:
                # First segment: place images starting from index 0
                for img_idx, frame_pos in enumerate(consolidated_keyframe_positions):
                    if img_idx < len(input_images_resolved_for_guide) and frame_pos <= total_frames_for_segment - 1:
                        img_path = input_images_resolved_for_guide[img_idx]
                        keyframe_np = image_to_frame(img_path, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
                        if keyframe_np is not None:
                            frames_for_guide_list[frame_pos] = keyframe_np.copy()
                            # Mark keyframe as guided
                            guidance_tracker.mark_single_frame(frame_pos)
                            generation_logger.debug(f"Task {task_id_for_logging}: Placed image {img_idx} at frame {frame_pos}")
            else:
                # Subsequent consolidated segment: handle overlap + place end anchor
                # First, extract overlap frames from previous video if needed
                if frame_overlap_from_previous > 0 and path_to_previous_segment_video_output_for_guide:
                    generation_logger.debug(f"Task {task_id_for_logging}: CONSOLIDATED SEGMENT - extracting {frame_overlap_from_previous} overlap frames")

                    # Extract the overlap frames from the previous video
                    try:
                        all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide)
                        if all_prev_frames and len(all_prev_frames) >= frame_overlap_from_previous:
                            overlap_frames = all_prev_frames[-frame_overlap_from_previous:]
                            # Place overlap frames at the beginning of this video
                            for overlap_idx, overlap_frame in enumerate(overlap_frames):
                                if overlap_idx < total_frames_for_segment:
                                    frames_for_guide_list[overlap_idx] = overlap_frame.copy()
                                    # Mark as guided for structure video tracking
                                    guidance_tracker.mark_single_frame(overlap_idx)
                                    generation_logger.debug(f"Task {task_id_for_logging}: Placed overlap frame {overlap_idx} from previous video")
                    except (OSError, ValueError, RuntimeError) as e:
                        generation_logger.warning(f"Task {task_id_for_logging}: Could not extract overlap frames: {e}")

                # Then place ALL keyframe images at their positions (including intermediate ones)
                # For subsequent consolidated segments, we need to place:
                # 1. Intermediate keyframes at their positions
                # 2. The end anchor at the final position

                # Calculate which input images correspond to each keyframe position
                # The consolidated end anchor tells us which image should be the final one
                if len(consolidated_keyframe_positions) > 0:
                    # For subsequent segments, we need to place all keyframes that belong to this segment
                    # The end anchor index tells us which image should be at the final position
                    final_frame_pos = consolidated_keyframe_positions[-1]

                    # Calculate which input images should go at each keyframe position
                    # This depends on which segment we are and how keyframes were distributed
                    start_image_idx = end_anchor_image_index - len(consolidated_keyframe_positions) + 1

                    for i, frame_pos in enumerate(consolidated_keyframe_positions):
                        image_idx = start_image_idx + i
                        if image_idx >= 0 and image_idx < len(input_images_resolved_for_guide) and frame_pos <= total_frames_for_segment - 1:
                            img_path = input_images_resolved_for_guide[image_idx]
                            keyframe_np = image_to_frame(img_path, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
                            if keyframe_np is not None:
                                frames_for_guide_list[frame_pos] = keyframe_np.copy()
                                # Mark keyframe as guided
                                guidance_tracker.mark_single_frame(frame_pos)
                                generation_logger.debug(f"Task {task_id_for_logging}: Placed image {image_idx} at consolidated keyframe position {frame_pos}")

                    generation_logger.debug(f"Task {task_id_for_logging}: Placed {len(consolidated_keyframe_positions)} keyframes for consolidated segment (end anchor: image {end_anchor_image_index})")

            # Apply structure guidance to unguidanced frames before creating video
            if structure_video_path or structure_guidance_video_url:
                frames_for_guide_list = apply_structure_motion_with_tracking(
                    frames_for_guide_list=frames_for_guide_list,
                    guidance_tracker=guidance_tracker,
                    structure_video_path=structure_video_path,
                    structure_video_treatment=structure_video_treatment,
                    structure_type=structure_type,
                    parsed_res_wh=parsed_res_wh,
                    fps_helpers=fps_helpers,
                    motion_strength=structure_video_motion_strength,
                    canny_intensity=structure_canny_intensity,
                    depth_contrast=structure_depth_contrast,
                    structure_guidance_video_url=structure_guidance_video_url,
                    segment_processing_dir=output_target_dir,
                    structure_guidance_frame_offset=structure_guidance_frame_offset,
                )

                # Emit one guidance-track summary (post-structure) — the pre-state is
                # implicit from the STRUCTURE_MOTION card just above.
                generation_logger.debug_anomaly("GUIDANCE_TRACK", "post-structure:\n" + guidance_tracker.debug_summary())

            return create_video_from_frames_list(frames_for_guide_list, predefined_output_path or output_target_dir / guide_video_base_name, fps_helpers, parsed_res_wh)

        end_anchor_frame_np = None

        if not single_image_journey:
            end_anchor_img_path_str: str
            if end_anchor_image_index < len(input_images_resolved_for_guide):
                end_anchor_img_path_str = input_images_resolved_for_guide[end_anchor_image_index]
            else:
                 raise ValueError(f"Seg {segment_idx_for_logging}: End anchor index {end_anchor_image_index} out of bounds for input images list ({len(input_images_resolved_for_guide)} images available).")

            end_anchor_frame_np = image_to_frame(end_anchor_img_path_str, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
            if end_anchor_frame_np is None: raise ValueError(f"Failed to load end anchor image: {end_anchor_img_path_str}")

        num_end_anchor_duplicates = 1
        start_anchor_frame_np = None

        if is_first_segment_from_scratch:
            start_anchor_img_path_str = input_images_resolved_for_guide[0]
            start_anchor_frame_np = image_to_frame(start_anchor_img_path_str, parsed_res_wh, task_id_for_logging=task_id_for_logging, image_download_dir=segment_image_download_dir, debug_mode=debug_mode)
            if start_anchor_frame_np is None: raise ValueError(f"Failed to load start anchor: {start_anchor_img_path_str}")
            if frames_for_guide_list:
                frames_for_guide_list[0] = start_anchor_frame_np.copy()
                # Mark first frame as guided
                guidance_tracker.mark_single_frame(0)

            if single_image_journey:
                pass  # Single-image journey: first frame is anchor, rest are gray/masked — silent
            else:
                # This is the original logic for fading between start and end.
                pot_max_idx_start_fade = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_frames_start_fade = max(0, pot_max_idx_start_fade)
                num_start_fade_steps = int(avail_frames_start_fade * fo_factor)
                if num_start_fade_steps > 0:
                    easing_fn_out = get_easing_function(fo_curve)
                    for k_fo in range(num_start_fade_steps):
                        idx_in_guide = 1 + k_fo
                        if idx_in_guide >= total_frames_for_segment: break
                        alpha_lin = 1.0 - ((k_fo + 1) / float(num_start_fade_steps))
                        e_alpha = fo_low + (fo_high - fo_low) * easing_fn_out(alpha_lin)
                        frames_for_guide_list[idx_in_guide] = cv2.addWeighted(frames_for_guide_list[idx_in_guide].astype(np.float32), 1.0 - e_alpha, start_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx_in_guide)

                min_idx_end_fade = 1
                max_idx_end_fade = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_frames_end_fade = max(0, max_idx_end_fade - min_idx_end_fade + 1)
                num_end_fade_steps = int(avail_frames_end_fade * fi_factor)
                if num_end_fade_steps > 0:
                    actual_end_fade_start_idx = max(min_idx_end_fade, max_idx_end_fade - num_end_fade_steps + 1)
                    easing_fn_in = get_easing_function(fi_curve)
                    for k_fi in range(num_end_fade_steps):
                        idx_in_guide = actual_end_fade_start_idx + k_fi
                        if idx_in_guide >= total_frames_for_segment: break
                        alpha_lin = (k_fi + 1) / float(num_end_fade_steps)
                        e_alpha = fi_low + (fi_high - fi_low) * easing_fn_in(alpha_lin)
                        base_f = frames_for_guide_list[idx_in_guide]
                        frames_for_guide_list[idx_in_guide] = cv2.addWeighted(base_f.astype(np.float32), 1.0 - e_alpha, end_anchor_frame_np.astype(np.float32), e_alpha, 0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx_in_guide)
                elif fi_factor > 0 and avail_frames_end_fade > 0:
                    for k_fill in range(min_idx_end_fade, max_idx_end_fade + 1):
                        if k_fill < total_frames_for_segment:
                            frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()
                            # Mark filled frames as guided
                            guidance_tracker.mark_single_frame(k_fill)

        elif path_to_previous_segment_video_output_for_guide: # Continued or Subsequent
            generation_logger.debug(f"GuideBuilder (Seg {segment_idx_for_logging}): Subsequent segment logic started.")
            generation_logger.debug(f"GuideBuilder: Prev video path: {path_to_previous_segment_video_output_for_guide}")
            generation_logger.debug(f"GuideBuilder: Overlap from prev setting: {frame_overlap_from_previous}")
            generation_logger.debug(f"GuideBuilder: Prev video exists: {Path(path_to_previous_segment_video_output_for_guide).exists()}")

            if not Path(path_to_previous_segment_video_output_for_guide).exists():
                raise ValueError(f"Previous video path does not exist: {path_to_previous_segment_video_output_for_guide}")

            # Wait for file to be stable before reading (important for recently encoded videos)
            generation_logger.debug(f"GuideBuilder: Waiting for previous video file to stabilize...")
            file_stable = wait_for_file_stable(path_to_previous_segment_video_output_for_guide, checks=3, interval=1.0)
            if not file_stable:
                generation_logger.warning(f"GuideBuilder: File stability check failed, proceeding anyway")

            # Get the expected frame count for the previous segment from orchestrator data
            # IMPORTANT: Account for context frames that were added to segments after the first
            expected_prev_segment_frames = None
            if segment_idx_for_logging > 0 and orchestrator_details:
                segment_frames_expanded = orchestrator_details.get("segment_frames_expanded", [])
                frame_overlap_expanded = orchestrator_details.get("frame_overlap_expanded", [])
                if segment_idx_for_logging - 1 < len(segment_frames_expanded):
                    base_frames = segment_frames_expanded[segment_idx_for_logging - 1]
                    # For segments after the first, they generate extra context frames
                    # Previous segment index is segment_idx_for_logging - 1
                    prev_seg_idx = segment_idx_for_logging - 1
                    if prev_seg_idx > 0 and prev_seg_idx - 1 < len(frame_overlap_expanded):
                        # Previous segment had context frames added
                        prev_seg_context = frame_overlap_expanded[prev_seg_idx - 1]
                        expected_prev_segment_frames = base_frames + prev_seg_context
                        generation_logger.debug(f"GuideBuilder: Previous segment (idx {prev_seg_idx}) expected to have {expected_prev_segment_frames} frames ({base_frames} base + {prev_seg_context} context)")
                    else:
                        # Previous segment was segment 0, no context added
                        expected_prev_segment_frames = base_frames
                        generation_logger.debug(f"GuideBuilder: Previous segment (idx {prev_seg_idx}) expected to have {expected_prev_segment_frames} frames (no context)")


            # If we have the expected frame count, use it directly
            if expected_prev_segment_frames and expected_prev_segment_frames > 0:
                generation_logger.debug(f"GuideBuilder: Using known frame count {expected_prev_segment_frames} from orchestrator data")
                prev_vid_total_frames = expected_prev_segment_frames

                # Calculate overlap frames to extract
                actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                start_extraction_idx = max(0, prev_vid_total_frames - actual_overlap_to_use)
                generation_logger.debug(f"GuideBuilder: Extracting {actual_overlap_to_use} frames starting from index {start_extraction_idx}")

                # Extract the frames directly
                overlap_frames_raw = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, start_extraction_idx, actual_overlap_to_use)

                # Verify we got the expected number of frames
                if len(overlap_frames_raw) != actual_overlap_to_use:
                    generation_logger.warning(f"GuideBuilder: Expected {actual_overlap_to_use} frames but got {len(overlap_frames_raw)}. Falling back to manual extraction.")
                    # Fall back to extracting all frames
                    all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide)
                    prev_vid_total_frames = len(all_prev_frames)
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    overlap_frames_raw = all_prev_frames[-actual_overlap_to_use:] if actual_overlap_to_use > 0 else []
            else:
                # Fallback: No orchestrator data, use OpenCV or manual extraction
                generation_logger.debug(f"GuideBuilder: No orchestrator frame count data available, falling back to frame detection")
                prev_vid_total_frames, prev_vid_fps = get_video_frame_count_and_fps(path_to_previous_segment_video_output_for_guide)
                generation_logger.debug(f"GuideBuilder: Frame count from cv2: {prev_vid_total_frames}, fps: {prev_vid_fps}")

                if not prev_vid_total_frames:  # Handles None or 0
                    generation_logger.debug(f"GuideBuilder: Fallback triggered due to zero/None frame count. Manually reading frames.")
                    # Fallback: read all frames to determine length
                    all_prev_frames = extract_frames_from_video(path_to_previous_segment_video_output_for_guide)
                    prev_vid_total_frames = len(all_prev_frames)
                    generation_logger.debug(f"GuideBuilder: Manual frame count from fallback: {prev_vid_total_frames}")
                    if prev_vid_total_frames == 0:
                        raise ValueError("Previous segment video appears to have zero frames \u2013 cannot build guide overlap.")
                    # Decide how many overlap frames we can reuse
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    overlap_frames_raw = all_prev_frames[-actual_overlap_to_use:]
                    generation_logger.debug(f"GuideBuilder: Using fallback - extracting last {actual_overlap_to_use} frames from {prev_vid_total_frames} total frames")
                else:
                    generation_logger.debug(f"GuideBuilder: Using cv2 frame count.")
                    actual_overlap_to_use = min(frame_overlap_from_previous, prev_vid_total_frames)
                    start_extraction_idx = max(0, prev_vid_total_frames - actual_overlap_to_use)
                    generation_logger.debug(f"GuideBuilder: Extracting {actual_overlap_to_use} frames starting from index {start_extraction_idx}")
                    overlap_frames_raw = extract_frames_from_video(path_to_previous_segment_video_output_for_guide, start_extraction_idx, actual_overlap_to_use)

            # Log the final overlap calculation
            generation_logger.debug(f"GuideBuilder: Calculated actual_overlap_to_use: {actual_overlap_to_use if 'actual_overlap_to_use' in locals() else 'Not calculated'}")
            # Check video resolution to understand if it matches our target
            if overlap_frames_raw and len(overlap_frames_raw) > 0:
                first_frame_shape = overlap_frames_raw[0].shape
                prev_height, prev_width = first_frame_shape[0], first_frame_shape[1]
                generation_logger.debug(f"GuideBuilder: Previous video resolution from extracted frames: {prev_width}x{prev_height} (target: {parsed_res_wh[0]}x{parsed_res_wh[1]})")
                if prev_width != parsed_res_wh[0] or prev_height != parsed_res_wh[1]:
                    generation_logger.debug(f"GuideBuilder: Resolution mismatch detected! Previous video will be resized during guide creation.")

            frames_read_for_overlap = 0
            for k, frame_fp in enumerate(overlap_frames_raw):
                if k >= total_frames_for_segment: break
                original_shape = frame_fp.shape
                if frame_fp.shape[1]!=parsed_res_wh[0] or frame_fp.shape[0]!=parsed_res_wh[1]:
                    frame_fp = cv2.resize(frame_fp, parsed_res_wh, interpolation=cv2.INTER_AREA)
                    generation_logger.debug(f"GuideBuilder: Resized frame {k} from {original_shape} to {frame_fp.shape}")
                frames_for_guide_list[k] = frame_fp.copy()
                # Mark overlap frames as guided
                guidance_tracker.mark_single_frame(k)
                frames_read_for_overlap += 1

            generation_logger.debug(f"GuideBuilder: Frames copied into guide list: {frames_read_for_overlap}")

            # Log details about what frames were actually placed in the guide
            if frames_read_for_overlap > 0:
                generation_logger.debug(f"GuideBuilder: Guide frames 0-{frames_read_for_overlap-1} now contain frames from previous video")
                generation_logger.debug(f"GuideBuilder: Guide frames {frames_read_for_overlap}-{total_frames_for_segment-1} are still gray frames (will be modified by fade logic)")

            if frames_read_for_overlap > 0:
                if fo_factor > 0.0:
                    num_init_fade_steps = min(int(frames_read_for_overlap * fo_factor), frames_read_for_overlap)
                    easing_fn_fo_ol = get_easing_function(fo_curve)
                    for k_fo_ol in range(num_init_fade_steps):
                        alpha_l = 1.0 - ((k_fo_ol + 1) / float(num_init_fade_steps))
                        eff_s = fo_low + (fo_high - fo_low) * easing_fn_fo_ol(alpha_l)
                        eff_s = np.clip(eff_s + strength_adj, 0, 1)
                        base_f=frames_for_guide_list[k_fo_ol]
                        frames_for_guide_list[k_fo_ol] = cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                        if desat_factor > 0:
                            g=cv2.cvtColor(frames_for_guide_list[k_fo_ol],cv2.COLOR_BGR2GRAY)
                            gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
                            frames_for_guide_list[k_fo_ol]=cv2.addWeighted(frames_for_guide_list[k_fo_ol],1-desat_factor,gb,desat_factor,0)
                        if bright_adj!=0:
                            frames_for_guide_list[k_fo_ol]=adjust_frame_brightness(frames_for_guide_list[k_fo_ol],bright_adj)
                else:
                    eff_s=np.clip(fo_high+strength_adj,0,1)
                    if abs(eff_s-1.0)>1e-5 or desat_factor>0 or bright_adj!=0:
                        for k_s_ol in range(frames_read_for_overlap):
                            base_f=frames_for_guide_list[k_s_ol];frames_for_guide_list[k_s_ol]=cv2.addWeighted(gray_frame_bgr.astype(np.float32),1-eff_s,base_f.astype(np.float32),eff_s,0).astype(np.uint8)
                            if desat_factor>0: g=cv2.cvtColor(frames_for_guide_list[k_s_ol],cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR);frames_for_guide_list[k_s_ol]=cv2.addWeighted(frames_for_guide_list[k_s_ol],1-desat_factor,gb,desat_factor,0)
                            if bright_adj!=0: frames_for_guide_list[k_s_ol]=adjust_frame_brightness(frames_for_guide_list[k_s_ol],bright_adj)

            if not single_image_journey:
                min_idx_efs = frames_read_for_overlap; max_idx_efs = total_frames_for_segment - num_end_anchor_duplicates - 1
                avail_f_efs = max(0, max_idx_efs - min_idx_efs + 1); num_efs_steps = int(avail_f_efs * fi_factor)
                if num_efs_steps > 0:
                    actual_efs_start_idx = max(min_idx_efs, max_idx_efs - num_efs_steps + 1)
                    easing_fn_in_s = get_easing_function(fi_curve)
                    for k_fi_s in range(num_efs_steps):
                        idx = actual_efs_start_idx+k_fi_s
                        if idx >= total_frames_for_segment: break
                        if idx < min_idx_efs: continue
                        alpha_l=(k_fi_s+1)/float(num_efs_steps);e_alpha=fi_low+(fi_high-fi_low)*easing_fn_in_s(alpha_l);e_alpha=np.clip(e_alpha,0,1)
                        base_f=frames_for_guide_list[idx];frames_for_guide_list[idx]=cv2.addWeighted(base_f.astype(np.float32),1-e_alpha,end_anchor_frame_np.astype(np.float32),e_alpha,0).astype(np.uint8)
                        # Mark fade frames as guided
                        guidance_tracker.mark_single_frame(idx)
                elif fi_factor > 0 and avail_f_efs > 0:
                    for k_fill in range(min_idx_efs, max_idx_efs + 1):
                        if k_fill < total_frames_for_segment:
                            frames_for_guide_list[k_fill] = end_anchor_frame_np.copy()
                            # Mark filled frames as guided
                            guidance_tracker.mark_single_frame(k_fill)

        if not single_image_journey and total_frames_for_segment > 0 and end_anchor_frame_np is not None:
            for k_dup in range(min(num_end_anchor_duplicates, total_frames_for_segment)):
                idx_s = total_frames_for_segment - 1 - k_dup
                if idx_s >= 0:
                    frames_for_guide_list[idx_s] = end_anchor_frame_np.copy()
                    # Mark end anchor duplicates as guided
                    guidance_tracker.mark_single_frame(idx_s)
                else: break

        if is_first_segment_from_scratch and total_frames_for_segment > 0 and start_anchor_frame_np is not None:
            frames_for_guide_list[0] = start_anchor_frame_np.copy()
            # Already marked above, but ensure it's marked
            guidance_tracker.mark_single_frame(0)

        # Apply structure guidance to unguidanced frames before creating video
        if structure_video_path or structure_guidance_video_url:
            frames_for_guide_list = apply_structure_motion_with_tracking(
                frames_for_guide_list=frames_for_guide_list,
                guidance_tracker=guidance_tracker,
                structure_video_path=structure_video_path,
                structure_video_treatment=structure_video_treatment,
                structure_type=structure_type,
                parsed_res_wh=parsed_res_wh,
                fps_helpers=fps_helpers,
                motion_strength=structure_video_motion_strength,
                canny_intensity=structure_canny_intensity,
                depth_contrast=structure_depth_contrast,
                structure_guidance_video_url=structure_guidance_video_url,
                segment_processing_dir=output_target_dir,
                structure_guidance_frame_offset=structure_guidance_frame_offset,
            )

            generation_logger.debug_anomaly("GUIDANCE_TRACK", "post-structure:\n" + guidance_tracker.debug_summary())

        # Uni3C end frame exclusion: black out end frame so uni3c_zero_empty_frames=True
        # will zero the latent, letting i2v's native image_end handle the end frame alone
        if exclude_end_for_controlnet and frames_for_guide_list and len(frames_for_guide_list) > 0:
            black_frame = np.zeros((parsed_res_wh[1], parsed_res_wh[0], 3), dtype=np.uint8)
            frames_for_guide_list[-1] = black_frame

        if frames_for_guide_list:
            return create_video_from_frames_list(frames_for_guide_list, actual_guide_video_path, fps_helpers, parsed_res_wh)

        return None

    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"ERROR creating guide video for segment {segment_idx_for_logging}: {e}", exc_info=True)
        return None


# Legacy module surface now points at the canonical travel API facade.
from source.task_handlers.travel.api import (  # noqa: E402
    create_guide_video_for_travel_segment as create_guide_video_for_travel_segment,
    prepare_vace_ref_for_segment as prepare_vace_ref_for_segment,
    rife_interpolate_images_to_video as rife_interpolate_images_to_video,
)
