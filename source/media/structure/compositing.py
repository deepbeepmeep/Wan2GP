"""Multi-structure video compositing: validation and composite timeline assembly."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch

from source.core.constants import BYTES_PER_MB
from source.core.log import generation_logger
from source.core.params.structure_guidance import normalize_structure_treatment
from source.media.structure.frame_ops import create_neutral_frame, load_structure_video_frames_with_range
from source.media.structure.preprocessors import process_structure_frames
from source.runtime.wgp_bridge import get_wan2gp_save_video_callable

__all__ = [
    "validate_structure_video_configs",
    "create_composite_guidance_video",
]


def download_structure_source(
    source_path: str,
    *,
    config_idx: int,
    download_dir: Path,
) -> str:
    """Materialize a structure source locally when it is an HTTP URL."""
    if not source_path.startswith(("http://", "https://")):
        return source_path

    download_dir.mkdir(parents=True, exist_ok=True)

    import requests

    local_filename = f"structure_src_{config_idx}_{Path(source_path).name}"
    local_path = download_dir / local_filename

    if not local_path.exists():
        generation_logger.debug(f"  Downloading: {source_path}")
        response = requests.get(source_path, timeout=120)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        generation_logger.debug(
            f"  Downloaded: {local_path.name} ({len(response.content) / BYTES_PER_MB:.2f} MB)"
        )
    else:
        generation_logger.debug(f"  Using cached: {local_path.name}")

    return str(local_path)


def validate_structure_video_configs(
    configs: List[dict],
    total_frames: int,
) -> List[dict]:
    """
    Validate and normalize structure video configurations.

    Args:
        configs: List of structure video config dicts
        total_frames: Total frames in the output timeline

    Returns:
        Sorted and validated configs

    Raises:
        ValueError: If configs are invalid or overlap
    """
    if not configs:
        return []

    # Sort by start_frame
    sorted_configs = sorted(configs, key=lambda c: c.get("start_frame", 0))

    # First pass: validate required fields and clip/filter configs
    valid_configs = []
    for i, raw_config in enumerate(sorted_configs):
        config = dict(raw_config)

        # Check required fields
        if "path" not in config:
            raise ValueError(f"Structure video config {i} missing 'path'")
        if "start_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'start_frame'")
        if "end_frame" not in config:
            raise ValueError(f"Structure video config {i} missing 'end_frame'")

        start = config["start_frame"]
        end = config["end_frame"]

        # Validate range
        if start < 0:
            raise ValueError(f"Config {i}: start_frame {start} < 0")

        # Skip configs that start beyond video length
        if start >= total_frames:
            generation_logger.warning(f"[COMPOSITE] Config {i}: start_frame {start} >= total_frames {total_frames}, skipping this config")
            continue

        # Auto-clip end_frame to total_frames instead of failing
        if end > total_frames:
            generation_logger.warning(f"[COMPOSITE] Config {i}: end_frame {end} > total_frames {total_frames}, clipping to {total_frames}")
            end = total_frames
            config["end_frame"] = end

        if start >= end:
            raise ValueError(f"Config {i}: start_frame {start} >= end_frame {end}")

        config["treatment"] = normalize_structure_treatment(
            config.get("treatment", "adjust"),
            context=f"structure config {i}",
        )
        valid_configs.append(config)

    # Second pass: check for overlaps on valid configs
    prev_end = -1
    for i, config in enumerate(valid_configs):
        start = config["start_frame"]
        end = config["end_frame"]

        if start < prev_end:
            raise ValueError(f"Config {i}: frame range [{start}, {end}) overlaps with previous config ending at {prev_end}")

        prev_end = end

    generation_logger.debug_anomaly("COMPOSITE", f"Validated {len(valid_configs)} structure video configs (from {len(sorted_configs)} input)")
    for i, cfg in enumerate(valid_configs):
        generation_logger.debug(f"  Config {i}: frames [{cfg['start_frame']}, {cfg['end_frame']}) from {Path(cfg['path']).name}")

    return valid_configs


def create_composite_guidance_video(
    structure_configs: List[dict],
    total_frames: int,
    structure_type: str,
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Optional[Path] = None,
) -> Path:
    """
    Create a single composite guidance video from multiple structure video sources.

    This function:
    1. Creates a timeline filled with neutral frames
    2. For each config, loads and processes source frames
    3. Places processed frames at the correct timeline positions
    4. Encodes everything as a single video

    Args:
        structure_configs: List of config dicts, each with:
            - path: Source video path/URL
            - start_frame: Start position in output timeline
            - end_frame: End position (exclusive) in output timeline
            - treatment: Optional "adjust" or "clip" (default: "adjust")
            - source_start_frame: Optional start frame in source video
            - source_end_frame: Optional end frame in source video
        total_frames: Total frames in the output timeline
        structure_type: "flow", "canny", "depth", "raw", or "uni3c"
        target_resolution: (width, height) tuple
        target_fps: Output video FPS
        motion_strength: Flow motion strength
        canny_intensity: Canny edge intensity
        depth_contrast: Depth map contrast
        download_dir: Directory for downloading source videos

    Returns:
        Path to the created composite guidance video
    """
    generation_logger.debug_anomaly("COMPOSITE", f"Creating composite guidance video...")
    generation_logger.debug(f"  Total frames: {total_frames}")
    generation_logger.debug(f"  Structure type: {structure_type}")
    generation_logger.debug(f"  Resolution: {target_resolution[0]}x{target_resolution[1]}")
    generation_logger.debug(f"  Configs: {len(structure_configs)}")

    # Validate configs
    sorted_configs = validate_structure_video_configs(structure_configs, total_frames)

    if not sorted_configs:
        raise ValueError("No valid structure video configs provided")

    # Initialize timeline with neutral frames
    generation_logger.debug_anomaly("COMPOSITE", f"Initializing {total_frames} neutral frames...")
    neutral_frame = create_neutral_frame(structure_type, target_resolution)
    composite_frames = [neutral_frame.copy() for _ in range(total_frames)]

    # Track which frame ranges are filled
    filled_ranges = []

    # Process each config
    for config_idx, config in enumerate(sorted_configs):
        source_path = config["path"]
        start_frame = config["start_frame"]
        end_frame = config["end_frame"]
        frames_needed = end_frame - start_frame
        treatment = config.get("treatment", "adjust")

        generation_logger.debug_anomaly("COMPOSITE", f"Processing config {config_idx}: {Path(source_path).name}")
        generation_logger.debug(f"  Timeline range: [{start_frame}, {end_frame}) = {frames_needed} frames")

        # Download if URL
        if source_path.startswith(("http://", "https://")):
            if download_dir is None:
                download_dir = Path("./temp_structure_downloads")
            source_path = download_structure_source(
                source_path,
                config_idx=config_idx,
                download_dir=download_dir,
            )

        # Load source frames with optional range extraction
        source_frames = load_structure_video_frames_with_range(
            structure_video_path=source_path,
            target_frame_count=frames_needed,
            target_fps=target_fps,
            target_resolution=target_resolution,
            treatment=treatment,
            crop_to_fit=True,
            source_start_frame=config.get("source_start_frame", 0),
            source_end_frame=config.get("source_end_frame"),
        )

        # Process frames with chosen preprocessor
        generation_logger.debug(f"  Processing with '{structure_type}' preprocessor...")
        processed_frames = process_structure_frames(
            source_frames,
            structure_type,
            motion_strength,
            canny_intensity,
            depth_contrast,
        )

        # Ensure we have exactly the frames needed
        if len(processed_frames) > frames_needed:
            processed_frames = processed_frames[:frames_needed]
            generation_logger.debug(f"  Trimmed to {frames_needed} frames")
        elif len(processed_frames) < frames_needed:
            # Pad with last frame if short
            while len(processed_frames) < frames_needed:
                processed_frames.append(processed_frames[-1].copy())
            generation_logger.debug(f"  Padded to {frames_needed} frames")

        # Place processed frames into composite
        for i, frame in enumerate(processed_frames):
            frame_idx = start_frame + i
            if 0 <= frame_idx < total_frames:
                composite_frames[frame_idx] = frame

        filled_ranges.append((start_frame, end_frame))
        generation_logger.debug(f"  Placed {len(processed_frames)} frames at positions {start_frame}-{end_frame-1}")

    # Log coverage summary
    total_filled = sum(end - start for start, end in filled_ranges)
    total_neutral = total_frames - total_filled
    generation_logger.debug_anomaly("COMPOSITE", f"Coverage summary:")
    generation_logger.debug(f"  Filled frames: {total_filled} ({100*total_filled/total_frames:.1f}%)")
    generation_logger.debug(f"  Neutral frames: {total_neutral} ({100*total_neutral/total_frames:.1f}%)")

    # Encode as video
    generation_logger.debug_anomaly("COMPOSITE", f"Encoding composite video to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try WGP's save_video first, fall back to cv2
    try:
        save_video = get_wan2gp_save_video_callable()

        video_tensor = np.stack(composite_frames, axis=0)  # [T, H, W, C]

        save_video(
            video_tensor,
            save_file=str(output_path),
            fps=target_fps,
            codec_type='libx264_8',
            normalize=False,
            value_range=(0, 255)
        )
    except ImportError:
        # cv2 fallback for encoding
        import cv2
        generation_logger.debug_anomaly("COMPOSITE", "Using cv2 fallback for video encoding")

        w, h = target_resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

        for frame_rgb in composite_frames:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

    if not output_path.exists():
        raise ValueError(f"Failed to create composite video at {output_path}")

    file_size_mb = output_path.stat().st_size / BYTES_PER_MB
    generation_logger.debug_anomaly("COMPOSITE", f"Created composite video: {output_path.name} ({file_size_mb:.2f} MB)")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path
