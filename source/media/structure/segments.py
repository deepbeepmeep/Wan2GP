"""Segment-level structure guidance application and positioning."""

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from source.core.log import generation_logger
from source.media.structure.tracker import GuidanceTracker
from source.media.structure.download import download_and_extract_motion_frames
from source.media.structure.compositing import (
    create_composite_guidance_video,
)

__all__ = [
    "StructureMotionApplicationResult",
    "apply_structure_motion_with_tracking",
    "apply_structure_motion_with_tracking_result",
    "segment_has_structure_overlap",
    "calculate_segment_guidance_position",
    "calculate_segment_stitched_position",
    "extract_segment_structure_guidance",
]


@dataclass
class StructureMotionApplicationResult:
    """Outcome of applying precomputed structure-guidance frames to a segment."""

    status: str
    frames: List[np.ndarray]
    filled_frames: int = 0
    error: str | None = None
    diagnostics: str | None = None


def apply_structure_motion_with_tracking_result(
    *,
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_guidance_video_url: str | None = None,
    segment_processing_dir: Path | None = None,
    structure_type: str = "flow",
    structure_guidance_frame_offset: int = 0,
) -> StructureMotionApplicationResult:
    """Apply structure motion and return a status-rich result object."""
    unguidanced_ranges = guidance_tracker.get_unguidanced_ranges()

    if not unguidanced_ranges:
        generation_logger.debug_anomaly("STRUCTURE_VIDEO", "No unguidanced frames found")
        return StructureMotionApplicationResult(
            status="skipped",
            frames=frames_for_guide_list,
            diagnostics="No unguidanced frames found",
        )

    total_unguidanced = sum(end - start + 1 for start, end in unguidanced_ranges)
    if not structure_guidance_video_url:
        generation_logger.debug(
            f"[STRUCTURE_VIDEO] Skipping {total_unguidanced} unguidanced frames across "
            f"{len(unguidanced_ranges)} ranges: no structure guidance video supplied"
        )
        return StructureMotionApplicationResult(
            status="skipped",
            frames=frames_for_guide_list,
            diagnostics="No structure guidance video provided",
        )

    if not segment_processing_dir:
        error = "segment_processing_dir required when using structure_guidance_video_url"
        generation_logger.error(f"[ERROR] {error}")
        return StructureMotionApplicationResult(
            status="failed",
            frames=frames_for_guide_list,
            error=error,
        )

    try:
        guidance_frames = download_and_extract_motion_frames(
            structure_motion_video_url=structure_guidance_video_url,
            frame_start=structure_guidance_frame_offset,
            frame_count=total_unguidanced,
            download_dir=segment_processing_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        generation_logger.error(f"[ERROR] FAST PATH FAILED: {exc}", exc_info=True)
        generation_logger.error("[ERROR] Structure guidance could not be applied")
        return StructureMotionApplicationResult(
            status="failed",
            frames=frames_for_guide_list,
            error=str(exc),
        )

    updated_frames = frames_for_guide_list.copy()
    guidance_frame_idx = 0
    frames_filled = 0

    for range_start, range_end in unguidanced_ranges:
        for frame_idx in range(range_start, range_end + 1):
            if guidance_frame_idx >= len(guidance_frames):
                break
            updated_frames[frame_idx] = guidance_frames[guidance_frame_idx]
            guidance_tracker.mark_single_frame(frame_idx)
            guidance_frame_idx += 1
            frames_filled += 1

    if frames_filled == total_unguidanced:
        status = "applied"
        diagnostics = None
    elif frames_filled > 0:
        status = "partial"
        diagnostics = (
            f"Filled {frames_filled}/{total_unguidanced} requested guidance frames"
        )
    else:
        status = "failed"
        diagnostics = f"Filled 0/{total_unguidanced} requested guidance frames"

    generation_logger.debug_block(
        "STRUCTURE_MOTION",
        {
            "status": status,
            "type": structure_type,
            "source": structure_guidance_video_url,
            "frames_filled": frames_filled,
            "frames_requested": total_unguidanced,
        },
    )
    return StructureMotionApplicationResult(
        status=status,
        frames=updated_frames,
        filled_frames=frames_filled,
        diagnostics=diagnostics,
    )


def apply_structure_motion_with_tracking(
    frames_for_guide_list: List[np.ndarray],
    guidance_tracker: GuidanceTracker,
    structure_video_path: str | None,  # Not used by segments (orchestrator uses it)
    structure_video_treatment: str,  # Not used by segments (orchestrator applies treatment)
    parsed_res_wh: Tuple[int, int],  # Not used by segments (orchestrator uses it)
    fps_helpers: int,  # Not used by segments (orchestrator uses it)
    structure_type: str = "flow",
    motion_strength: float = 1.0,  # Not used by segments (orchestrator applies it)
    canny_intensity: float = 1.0,  # Not used by segments (orchestrator applies it)
    depth_contrast: float = 1.0,  # Not used by segments (orchestrator applies it)
    structure_guidance_video_url: str | None = None,
    segment_processing_dir: Path | None = None,
    structure_guidance_frame_offset: int = 0,
) -> List[np.ndarray]:
    """
    Apply structure guidance to unguidanced frames (called by segment workers).

    IMPORTANT: This function marks frames as guided on the tracker as it fills them.

    The orchestrator pre-computes all structure guidance (flow/canny/depth visualizations)
    with motion_strength/intensity/contrast already applied, then uploads to Supabase.
    Segments download and insert their portion of the pre-computed guidance frames.

    Args:
        frames_for_guide_list: Current guide frames
        guidance_tracker: Tracks which frames have guidance (MUTATED by this function)
        structure_video_path: Not used by segments (orchestrator uses it to create guidance video)
        structure_video_treatment: Not used by segments (orchestrator applies treatment when creating guidance video)
        parsed_res_wh: Target resolution - not used by segments (orchestrator uses it)
        fps_helpers: Target FPS - not used by segments (orchestrator uses it)
        structure_type: Type of preprocessing ("flow", "canny", or "depth") - for logging only
        motion_strength: Not used by segments (orchestrator applies it when creating guidance video)
        canny_intensity: Not used by segments (orchestrator applies it when creating guidance video)
        depth_contrast: Not used by segments (orchestrator applies it when creating guidance video)
        structure_guidance_video_url: URL/path to pre-computed guidance video from orchestrator (REQUIRED)
        segment_processing_dir: Directory for downloads (required)
        structure_guidance_frame_offset: Starting frame offset in the guidance video

    Returns:
        Updated frames list with structure guidance applied
    """
    result = apply_structure_motion_with_tracking_result(
        frames_for_guide_list=frames_for_guide_list,
        guidance_tracker=guidance_tracker,
        structure_guidance_video_url=structure_guidance_video_url,
        segment_processing_dir=segment_processing_dir,
        structure_type=structure_type,
        structure_guidance_frame_offset=structure_guidance_frame_offset,
    )
    return result.frames


def segment_has_structure_overlap(
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
    structure_videos: List[dict],
) -> bool:
    """
    Check if a segment overlaps with any structure video config.

    Uses GUIDANCE timeline (not stitched) because structure_videos start_frame/end_frame
    are specified in the guidance timeline by the UX.

    Args:
        segment_index: Index of the segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments (unused, kept for API compat)
        structure_videos: List of structure video configs

    Returns:
        True if segment overlaps with at least one config, False otherwise
    """
    if not structure_videos:
        return False

    # Use GUIDANCE timeline (matches UX) - not stitched timeline
    seg_start, seg_frames = calculate_segment_guidance_position(
        segment_index, segment_frames_expanded,
    )
    seg_end = seg_start + seg_frames

    for cfg in structure_videos:
        cfg_start = cfg.get("start_frame", 0)
        cfg_end = cfg.get("end_frame", 0)

        # Check overlap
        if cfg_start < seg_end and cfg_end > seg_start:
            generation_logger.debug_anomaly("OVERLAP_CHECK", f"Segment {segment_index} [{seg_start}, {seg_end}) overlaps with config [{cfg_start}, {cfg_end})")
            return True

    generation_logger.debug_anomaly("OVERLAP_CHECK", f"Segment {segment_index} [{seg_start}, {seg_end}) has NO overlap with any config")
    return False


def calculate_segment_guidance_position(
    segment_index: int,
    segment_frames_expanded: List[int],
) -> Tuple[int, int]:
    """
    Calculate a segment's start position and frame count in the GUIDANCE timeline.

    The guidance timeline is the raw total of all segment frames (no overlaps removed).
    This matches the UX timeline where users set structure video start_frame/end_frame.

    Args:
        segment_index: Index of the segment (0-based)
        segment_frames_expanded: List of frame counts per segment

    Returns:
        Tuple of (guidance_start, frame_count) for this segment
    """
    guidance_start = sum(segment_frames_expanded[:segment_index])
    frame_count = segment_frames_expanded[segment_index] if segment_index < len(segment_frames_expanded) else 81

    generation_logger.debug_anomaly("SEGMENT_POS", f"Segment {segment_index}: guidance_start={guidance_start}, frame_count={frame_count} (guidance timeline)")
    return guidance_start, frame_count


def calculate_segment_stitched_position(
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
) -> Tuple[int, int]:
    """
    Calculate a segment's start position and frame count in the STITCHED timeline.

    The stitched timeline has overlaps removed - this is the final video length.
    NOTE: For structure video overlap checking, use calculate_segment_guidance_position instead,
    as the UX shows structure configs in the guidance timeline.

    Args:
        segment_index: Index of the segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments

    Returns:
        Tuple of (stitched_start, frame_count) for this segment
    """
    stitched_start = 0
    for idx in range(segment_index):
        segment_frames = segment_frames_expanded[idx]
        if idx == 0:
            stitched_start = segment_frames
        else:
            overlap = frame_overlap_expanded[idx - 1] if idx > 0 and idx - 1 < len(frame_overlap_expanded) else 0
            stitched_start += segment_frames - overlap

    # Adjust for the first segment's contribution
    if segment_index > 0:
        overlap = frame_overlap_expanded[segment_index - 1] if segment_index - 1 < len(frame_overlap_expanded) else 0
        stitched_start -= overlap
    else:
        stitched_start = 0

    frame_count = segment_frames_expanded[segment_index] if segment_index < len(segment_frames_expanded) else 81

    generation_logger.debug_anomaly("SEGMENT_POS", f"Segment {segment_index}: stitched_start={stitched_start}, frame_count={frame_count} (stitched timeline)")
    return stitched_start, frame_count


def extract_segment_structure_guidance(
    structure_videos: List[dict],
    segment_index: int,
    segment_frames_expanded: List[int],
    frame_overlap_expanded: List[int],
    target_resolution: Tuple[int, int],
    target_fps: int,
    output_path: Path,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
    download_dir: Optional[Path] = None,
    segment_start_frame: Optional[int] = None,
    segment_frame_count: Optional[int] = None,
) -> Optional[Path]:
    """
    Extract structure guidance for a single segment from the full structure_videos config.

    This is the segment-level function that allows standalone segments to compute
    their own portion of structure guidance without needing the orchestrator to
    pre-compute a full composite video.

    Args:
        structure_videos: Full array of structure video configs (same format as orchestrator)
        segment_index: Index of this segment (0-based)
        segment_frames_expanded: List of frame counts per segment
        frame_overlap_expanded: List of overlap values between segments
        target_resolution: (width, height) tuple
        target_fps: Target FPS
        output_path: Where to save the segment's guidance video
        motion_strength: Flow motion strength
        canny_intensity: Canny edge intensity
        depth_contrast: Depth map contrast
        download_dir: Directory for downloading source videos
        segment_start_frame: Optional precomputed GUIDANCE-timeline start frame
        segment_frame_count: Optional precomputed segment frame count

    Returns:
        Path to the created segment guidance video, or None if no configs apply
    """
    # Calculate total guidance timeline for context
    total_guidance_frames = sum(segment_frames_expanded)
    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"========== SEGMENT {segment_index} GUIDANCE EXTRACTION ==========")
    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Total guidance timeline: {total_guidance_frames} frames")
    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Segment frames: {segment_frames_expanded}")

    if not structure_videos:
        generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"No structure_videos provided, skipping")
        return None

    # Calculate this segment's position in the GUIDANCE timeline (matches UX)
    if segment_start_frame is not None and segment_frame_count is not None:
        seg_start, seg_frames = segment_start_frame, segment_frame_count
    else:
        seg_start, seg_frames = calculate_segment_guidance_position(
            segment_index, segment_frames_expanded,
        )
    seg_end = seg_start + seg_frames

    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Segment {segment_index} covers GUIDANCE frames [{seg_start}, {seg_end})")

    # Log structure video configs for debugging
    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Checking {len(structure_videos)} structure_videos configs:")
    for i, cfg in enumerate(structure_videos):
        cfg_start = cfg.get("start_frame", 0)
        cfg_end = cfg.get("end_frame", 0)
        overlaps = cfg_start < seg_end and cfg_end > seg_start
        status = "OVERLAPS" if overlaps else "no overlap"
        generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"  Config {i}: [{cfg_start}, {cfg_end}) {status}")

    # Extract structure_type from configs (all must be same type)
    structure_types = set()
    for cfg in structure_videos:
        cfg_type = cfg.get("structure_type", cfg.get("type", "flow"))
        structure_types.add(cfg_type)

    if len(structure_types) > 1:
        raise ValueError(f"All structure_videos must have same type, found: {structure_types}")

    structure_type = structure_types.pop() if structure_types else "flow"

    # Find configs that overlap with this segment's frame range
    relevant_configs = []
    for cfg in structure_videos:
        cfg_start = cfg["start_frame"]
        cfg_end = cfg["end_frame"]

        # Check if this config overlaps with the segment's range
        if cfg_start < seg_end and cfg_end > seg_start:
            # Calculate overlap region in stitched timeline
            overlap_start = max(cfg_start, seg_start)
            overlap_end = min(cfg_end, seg_end)
            # Transform to segment-local coordinates
            local_start = overlap_start - seg_start
            local_end = overlap_end - seg_start

            # Calculate which portion of SOURCE video to use
            # The original config maps source range to stitched range [cfg_start, cfg_end)
            # We need the portion that corresponds to [overlap_start, overlap_end)
            cfg_duration = cfg_end - cfg_start
            src_start_orig = cfg.get("source_start_frame", 0)
            src_end_orig = cfg.get("source_end_frame")

            # If source_end not specified, we'll let the loader handle it
            # For proportional mapping, we need source video length
            if src_end_orig is not None:
                src_duration = src_end_orig - src_start_orig
                # Proportional mapping into source video
                overlap_start_in_cfg = overlap_start - cfg_start
                overlap_end_in_cfg = overlap_end - cfg_start

                new_src_start = src_start_orig + (overlap_start_in_cfg / cfg_duration) * src_duration
                new_src_end = src_start_orig + (overlap_end_in_cfg / cfg_duration) * src_duration
            else:
                # Can't do proportional without knowing source length
                # Let the loader handle the full source range
                new_src_start = src_start_orig
                new_src_end = src_end_orig

            transformed_cfg = {
                "path": cfg["path"],
                "start_frame": local_start,
                "end_frame": local_end,
                "treatment": cfg.get("treatment", "adjust"),
                "source_start_frame": int(new_src_start) if new_src_start is not None else 0,
                "source_end_frame": int(new_src_end) if new_src_end is not None else None,
                "structure_type": structure_type,
                "motion_strength": cfg.get("motion_strength", motion_strength),
            }

            generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Config overlaps: stitched [{cfg_start},{cfg_end}) -> local [{local_start},{local_end})")
            generation_logger.debug(f"  Source frames: [{new_src_start}, {new_src_end})")
            relevant_configs.append(transformed_cfg)

    if not relevant_configs:
        # No overlap = no structure guidance needed for this segment
        # Return None so the segment proceeds without structure guidance entirely
        # This is cleaner than creating an all-neutral video that gets zeroed anyway
        generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"No configs overlap with segment {segment_index}, skipping structure guidance")
        return None

    generation_logger.debug_anomaly("SEGMENT_GUIDANCE", f"Found {len(relevant_configs)} overlapping configs")

    # Create mini-composite using the transformed configs
    return create_composite_guidance_video(
        structure_configs=relevant_configs,
        total_frames=seg_frames,
        structure_type=structure_type,
        target_resolution=target_resolution,
        target_fps=target_fps,
        output_path=output_path,
        motion_strength=motion_strength,
        canny_intensity=canny_intensity,
        depth_contrast=depth_contrast,
        download_dir=download_dir,
    )
