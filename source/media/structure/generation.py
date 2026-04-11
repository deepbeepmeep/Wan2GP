"""Structure guidance video generation orchestrators."""

import numpy as np
from pathlib import Path
from typing import Tuple
import torch

from source.core.constants import BYTES_PER_MB
from source.core.log import generation_logger
from source.media.structure.loading import load_structure_video_frames
from source.media.structure.preprocessors import process_structure_frames
from source.runtime.wgp_bridge import get_wan2gp_save_video_callable

# Re-export extracted functions so existing `from generation import X` still works
from source.media.structure.frame_ops import (  # noqa: F401
    create_neutral_frame,
    load_structure_video_frames_with_range,
)
from source.media.structure.compositing import (  # noqa: F401
    validate_structure_video_configs,
    create_composite_guidance_video,
)

__all__ = [
    "create_structure_guidance_video",
    "create_trimmed_structure_video",
    "create_neutral_frame",
    "load_structure_video_frames_with_range",
    "validate_structure_video_configs",
    "create_composite_guidance_video",
]


def create_structure_guidance_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    structure_type: str = "flow",
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    treatment: str = "adjust",
) -> Path:
    """
    Create a video of preprocessed structure visualizations from the structure video.

    This is the NEW orchestrator-level function that:
    1. Loads and preprocesses frames from the structure video
    2. Applies the chosen preprocessor (flow, canny, depth, or raw)
    3. Encodes them as an H.264 video

    The resulting video contains structure visualizations that segments can use as
    VACE guide videos for structural conditioning, or raw frames for Uni3C.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate (total unguidanced frames)
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        structure_type: Type of preprocessing:
            - "flow": Optical flow visualization (VACE)
            - "canny": Edge detection (VACE)
            - "depth": Depth map estimation (VACE)
            - "raw": No preprocessing - raw frames only (Uni3C)
            Default: "flow"
        motion_strength: Flow strength multiplier (only used for flow, also maps to uni3c_strength for raw)
        canny_intensity: Edge intensity multiplier (only used for canny)
        depth_contrast: Depth contrast adjustment (only used for depth)
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)

    Returns:
        Path to the created video file

    Raises:
        ValueError: If structure video cannot be loaded or processed
    """
    generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Creating {structure_type} visualization video...")
    generation_logger.debug(f"  Source: {structure_video_path}")
    generation_logger.debug(f"  Type: {structure_type}")
    generation_logger.debug(f"  Frames: {max_frames_needed}")
    generation_logger.debug(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    generation_logger.debug(f"  Treatment: {treatment}")

    # Log active strength parameter
    if structure_type == "flow" and abs(motion_strength - 1.0) > 1e-6:
        generation_logger.debug(f"  Motion strength: {motion_strength}")
    elif structure_type == "canny" and abs(canny_intensity - 1e-6) > 1e-6:
        generation_logger.debug(f"  Canny intensity: {canny_intensity}")
    elif structure_type == "raw":
        generation_logger.debug(f"  Raw frames: No preprocessing applied (Uni3C mode)")
        if abs(motion_strength - 1.0) > 1e-6:
            generation_logger.debug(f"  Uni3C strength: {motion_strength} (from motion_strength)")
    elif structure_type == "depth" and abs(depth_contrast - 1.0) > 1e-6:
        generation_logger.debug(f"  Depth contrast: {depth_contrast}")

    try:
        # Step 1: Load structure video frames with treatment mode
        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Loading structure video frames...")
        structure_frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,  # Apply center cropping
        )

        # Step 2: Process frames with chosen preprocessor
        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            structure_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
        )

        if not processed_frames:
            raise ValueError(f"No {structure_type} visualizations extracted from structure video")

        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Processed {len(processed_frames)} frames")

        # Step 3: Encode as video
        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Encoding video to {output_path}")

        save_video = get_wan2gp_save_video_callable()

        # Convert to numpy array format expected by save_video
        # save_video expects [T, H, W, C] in range [0, 255]
        video_tensor = np.stack(processed_frames, axis=0)  # [T, H, W, C]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save video using WGP's video utilities
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',  # Use libx264 with 8-bit encoding
            normalize=False,  # Already in [0, 255] uint8 range
            value_range=(0, 255)  # Specify value range for uint8 data
        )

        # Verify output exists
        if not output_path.exists():
            raise ValueError(f"Failed to create video at {output_path}")

        file_size_mb = output_path.stat().st_size / BYTES_PER_MB
        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Created video: {output_path.name} ({file_size_mb:.2f} MB)")

        # Clean up GPU memory
        generation_logger.debug_anomaly("STRUCTURE_GUIDANCE_VIDEO", f"Cleaning up GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"[ERROR] Failed to create structure guidance video: {e}", exc_info=True)
        raise


def create_trimmed_structure_video(
    structure_video_path: str,
    max_frames_needed: int,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    treatment: str = "adjust",
) -> Path:
    """
    Create a trimmed/adjusted version of the structure video without applying any style transfer.
    This preserves the original video content but clips/stretches it to match the generation length.

    Args:
        structure_video_path: Path to the source structure video
        max_frames_needed: Number of frames to generate
        target_resolution: (width, height) for output frames
        target_fps: FPS for the output video
        output_path: Where to save the output video
        treatment: "adjust" (stretch/compress entire video) or "clip" (temporal sample)

    Returns:
        Path to the created video file
    """
    generation_logger.debug_anomaly("TRIMMED_STRUCTURE_VIDEO", f"Creating trimmed structure video...")
    generation_logger.debug(f"  Source: {structure_video_path}")
    generation_logger.debug(f"  Frames: {max_frames_needed}")
    generation_logger.debug(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    generation_logger.debug(f"  Treatment: {treatment}")

    try:
        # Step 1: Load structure video frames with treatment mode
        # This handles the trimming/adjusting logic
        frames = load_structure_video_frames(
            structure_video_path,
            target_frame_count=max_frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
        )

        # Step 2: Encode as video directly (no style transfer)
        generation_logger.debug_anomaly("TRIMMED_STRUCTURE_VIDEO", f"Encoding video to {output_path}")

        save_video = get_wan2gp_save_video_callable()

        # Convert list of numpy arrays [H, W, C] to tensor [T, H, W, C]
        video_tensor = np.stack(frames, axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save video using WGP's video utilities
        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )

        return output_path

    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"[TRIMMED_STRUCTURE_VIDEO] Error: {e}")
        raise
