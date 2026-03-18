"""
Guide Video Builder for Travel Segments

Handles guide video creation for VACE models, including:
- Previous segment video retrieval (local or remote)
- Input image preparation for guide creation
- Structure guidance configuration and local segment extraction
- Guide video creation via create_guide_video_for_travel_segment
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any

from source.core.log import travel_logger
from source.media.video import create_guide_video_for_travel_segment
from source.media.video.ffmpeg_ops import create_video_from_frames_list
from source.core.params.structure_guidance import StructureGuidanceConfig
from source.core.params.travel_guidance import TravelGuidanceConfig
from source.core.db.dependencies.task_dependencies_queries import get_predecessor_output_via_edge_function
from source.utils.download_utils import download_file, download_video_if_url
from source.utils.output_paths import prepare_output_path


def _build_local_travel_guidance_video(
    proc: Any,
    travel_guidance_config: TravelGuidanceConfig,
) -> Optional[Path]:
    """Build a segment-local composite guidance clip when no shared clip is provided."""
    ctx = proc.ctx
    if not travel_guidance_config.videos:
        return None

    from source.media.structure import create_composite_guidance_video, calculate_segment_stitched_position

    segment_frames_expanded = ctx.orchestrator_details.get(
        "segment_frames_expanded",
        [ctx.total_frames_for_segment],
    )
    frame_overlap_expanded = ctx.orchestrator_details.get("frame_overlap_expanded", [0])
    segment_start, _segment_frame_count = calculate_segment_stitched_position(
        ctx.segment_idx,
        segment_frames_expanded,
        frame_overlap_expanded,
    )
    segment_end = segment_start + ctx.total_frames_for_segment

    local_configs = []
    for video in travel_guidance_config.videos:
        video_start = video.start_frame
        video_end = segment_end if video.end_frame is None else video.end_frame
        overlap_start = max(video_start, segment_start)
        overlap_end = min(video_end, segment_end)
        if overlap_start >= overlap_end:
            continue

        local_config = video.to_dict()
        local_config["start_frame"] = overlap_start - segment_start
        local_config["end_frame"] = overlap_end - segment_start
        local_configs.append(local_config)

    if not local_configs:
        return None

    output_path = ctx.segment_processing_dir / f"travel_guidance_seg{ctx.segment_idx}_{uuid.uuid4().hex[:6]}.mp4"
    return create_composite_guidance_video(
        structure_configs=local_configs,
        total_frames=ctx.total_frames_for_segment,
        structure_type=travel_guidance_config.get_preprocessor_type(),
        target_resolution=ctx.parsed_res_wh,
        target_fps=ctx.orchestrator_details.get("fps_helpers", 16),
        output_path=output_path,
        motion_strength=travel_guidance_config.strength,
        canny_intensity=travel_guidance_config.canny_intensity,
        depth_contrast=travel_guidance_config.depth_contrast,
        download_dir=ctx.segment_processing_dir,
    )


def _create_ltx_control_guide_video(
    proc: Any,
    travel_guidance_config: TravelGuidanceConfig,
    guide_video_final_path: Path,
) -> Optional[Path]:
    """Use the preprocessed control clip directly for LTX guided travel."""
    ctx = proc.ctx

    guidance_video_url = travel_guidance_config.guidance_video_url
    frame_offset = travel_guidance_config.frame_offset

    if not guidance_video_url:
        local_guidance_path = _build_local_travel_guidance_video(proc, travel_guidance_config)
        if local_guidance_path is None:
            return None
        guidance_video_url = str(local_guidance_path)
        frame_offset = 0

    from source.media.structure.download import download_and_extract_motion_frames
    import cv2

    guidance_frames = download_and_extract_motion_frames(
        structure_motion_video_url=guidance_video_url,
        frame_start=frame_offset,
        frame_count=ctx.total_frames_for_segment,
        download_dir=ctx.segment_processing_dir,
    )

    if not guidance_frames:
        return None

    control_frames_bgr = [
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for frame_rgb in guidance_frames
    ]
    return create_video_from_frames_list(
        control_frames_bgr,
        guide_video_final_path,
        ctx.orchestrator_details.get("fps_helpers", 16),
        ctx.parsed_res_wh,
    )

def get_previous_segment_video(proc: Any) -> Optional[str]:
    """Get previous segment video output for guide creation."""
    ctx = proc.ctx

    chain_segments = ctx.orchestrator_details.get("chain_segments", True)
    if not chain_segments:
         travel_logger.debug(f"[INDEPENDENT_SEGMENTS] chain_segments=False: Skipping previous segment video lookup for segment {ctx.segment_idx}", task_id=ctx.task_id)
         return None

    is_first_segment = ctx.segment_params.get("is_first_segment", ctx.segment_idx == 0)

    if is_first_segment and ctx.orchestrator_details.get("continue_from_video_resolved_path"):
        # First segment continuing from video
        return ctx.orchestrator_details.get("continue_from_video_resolved_path")
    elif not is_first_segment:
        # Subsequent segment - get predecessor output
        task_dependency_id, raw_path_from_db = get_predecessor_output_via_edge_function(ctx.task_id)
        if task_dependency_id and raw_path_from_db:
            travel_logger.debug(f"Seg {ctx.segment_idx}: Found predecessor output: {raw_path_from_db}", task_id=ctx.task_id)

            # Handle Supabase public URLs by downloading them locally for guide processing
            if raw_path_from_db.startswith("http"):
                try:
                    travel_logger.debug(f"Seg {ctx.segment_idx}: Detected remote URL for previous segment: {raw_path_from_db}. Downloading...", task_id=ctx.task_id)

                    # Import download utilities
                    remote_url = raw_path_from_db
                    local_filename = Path(remote_url).name
                    # Store under segment_processing_dir to keep things tidy
                    local_download_path = ctx.segment_processing_dir / f"prev_{ctx.segment_idx:02d}_{local_filename}"

                    # Ensure directory exists
                    ctx.segment_processing_dir.mkdir(parents=True, exist_ok=True)

                    # Perform download if file not already present
                    if not local_download_path.exists():
                        travel_logger.debug(f"Seg {ctx.segment_idx}: Downloading from {remote_url}", task_id=ctx.task_id)
                        download_file(remote_url, ctx.segment_processing_dir, local_download_path.name)
                        travel_logger.debug(f"Seg {ctx.segment_idx}: Downloaded previous segment video to {local_download_path}", task_id=ctx.task_id)

                        # Verify download was successful and file has content
                        if local_download_path.exists() and local_download_path.stat().st_size > 0:
                            travel_logger.debug(f"Seg {ctx.segment_idx}: Download verified - file size: {local_download_path.stat().st_size:,} bytes", task_id=ctx.task_id)
                        else:
                            raise Exception(f"Download failed or resulted in empty file: {local_download_path}")
                    else:
                        travel_logger.debug(f"Seg {ctx.segment_idx}: Local copy of previous segment video already exists at {local_download_path}", task_id=ctx.task_id)
                        # Verify cached file is still valid
                        if local_download_path.stat().st_size == 0:
                            travel_logger.debug(f"Seg {ctx.segment_idx}: Cached file is empty, re-downloading...", task_id=ctx.task_id)
                            local_download_path.unlink()  # Remove empty file
                            download_file(remote_url, ctx.segment_processing_dir, local_download_path.name)

                    resolved_path = str(local_download_path.resolve())
                    travel_logger.debug(f"Seg {ctx.segment_idx}: Returning local path for guide creation: {resolved_path}", task_id=ctx.task_id)
                    return resolved_path

                except (OSError, ValueError, RuntimeError) as e_dl_prev:
                    travel_logger.warning(f"Seg {ctx.segment_idx}: Failed to download remote previous segment video: {e_dl_prev}", task_id=ctx.task_id)
                    # Return the original URL - this will likely cause an error downstream but preserves existing behavior
                    return raw_path_from_db
            else:
                # Path from DB is already absolute (Supabase)
                return raw_path_from_db
        else:
            travel_logger.warning(f"Seg {ctx.segment_idx}: Could not retrieve predecessor output", task_id=ctx.task_id)
            return None
    else:
        # First segment from scratch - no previous video
        return None

def prepare_input_images_for_guide(proc: "TravelSegmentProcessor") -> List[str]:
    """Prepare input images for guide video creation."""
    ctx = proc.ctx

    # For individual segment tasks, prefer the segment-specific image list
    # This contains just [start_image, end_image] for the specific segment
    individual_params = ctx.segment_params.get("individual_segment_params", {})
    individual_images = individual_params.get("input_image_paths_resolved", [])

    if individual_images:
        # Individual segment mode - use the 2-image list
        input_images_resolved_for_guide = individual_images.copy()
        travel_logger.debug(f"[GUIDE_INPUT_DEBUG] Seg {ctx.segment_idx}: Using {len(input_images_resolved_for_guide)} images from individual_segment_params", task_id=ctx.task_id)
    else:
        # Full orchestrator mode - use all images
        input_images_resolved_original = ctx.orchestrator_details.get("input_image_paths_resolved", [])
        if not input_images_resolved_original:
            raise ValueError(f"Seg {ctx.segment_idx}: input_image_paths_resolved missing from orchestrator_details (task {ctx.task_id})")
        input_images_resolved_for_guide = input_images_resolved_original.copy()
        travel_logger.debug(f"[GUIDE_INPUT_DEBUG] Seg {ctx.segment_idx}: Using {len(input_images_resolved_for_guide)} images from orchestrator payload", task_id=ctx.task_id)

    return input_images_resolved_for_guide

def create_guide_video(proc: Any) -> Optional[Path]:
    """
    Create guide video for VACE models or debug mode.

    Returns:
        Path to created guide video, or None if not created/failed
    """
    ctx = proc.ctx

    # Initialize structure_type tracking (will be set later if applicable)
    proc._detected_structure_type = None
    proc._travel_guidance_config = None
    proc._structure_config = None

    travel_guidance_config = None
    if isinstance(ctx.segment_params.get("travel_guidance"), dict):
        travel_guidance_config = TravelGuidanceConfig.from_payload(
            ctx.segment_params["travel_guidance"],
            ctx.model_name,
        )
        proc._travel_guidance_config = travel_guidance_config

    if travel_guidance_config is not None and travel_guidance_config.kind in {"vace", "uni3c"}:
        structure_config = travel_guidance_config.to_structure_guidance_config()
    else:
        structure_config = StructureGuidanceConfig.from_params({
            **ctx.orchestrator_details,
            **ctx.segment_params
        })
    proc._structure_config = structure_config

    guide_required = (
        proc.is_vace_model
        or structure_config.has_guidance
        or (
            travel_guidance_config is not None
            and travel_guidance_config.is_ltx_control
            and travel_guidance_config.has_guidance
        )
    )

    # Always create guide videos when the model or guidance mode requires them.
    # For everything else, only create in debug mode.
    if not ctx.debug_enabled and not guide_required:
        travel_logger.debug(f"Task {ctx.task_id}: Debug mode disabled and non-VACE model, skipping guide video creation", task_id=ctx.task_id)
        return None

    if guide_required and not ctx.debug_enabled:
        travel_logger.debug(
            f"Task {ctx.task_id}: guide video is required for this segment; creating it",
            task_id=ctx.task_id,
        )

    try:
        # Generate unique guide video filename
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        guide_video_filename = f"{ctx.task_id}_seg{ctx.segment_idx:02d}_guide_{timestamp_short}_{unique_suffix}.mp4"
        guide_video_base_name = f"{ctx.task_id}_seg{ctx.segment_idx:02d}_guide_{timestamp_short}_{unique_suffix}"

        # Use prepare_output_path to ensure guide video goes to task_type directory
        guide_video_final_path, _ = prepare_output_path(
            task_id=ctx.task_id,
            filename=guide_video_filename,
            main_output_dir_base=ctx.main_output_dir_base,
            task_type="travel_segment"
        )

        if travel_guidance_config is not None and travel_guidance_config.is_ltx_control:
            proc._detected_structure_type = travel_guidance_config.get_preprocessor_type()
            direct_control_video = _create_ltx_control_guide_video(
                proc,
                travel_guidance_config,
                guide_video_final_path,
            )
            if direct_control_video and Path(direct_control_video).exists():
                travel_logger.debug(
                    f"[GUIDE_DEBUG] Segment {ctx.segment_idx}: Using direct LTX control clip {direct_control_video}",
                    task_id=ctx.task_id,
                )
                return Path(direct_control_video)
            raise ValueError(
                f"LTX control guidance for segment {ctx.segment_idx} could not be materialized"
            )

        # Get previous segment video for guide creation
        path_to_previous_segment_video_output_for_guide = get_previous_segment_video(proc)

        # Prepare input images for guide creation
        input_images_resolved_for_guide = prepare_input_images_for_guide(proc)

        # Determine segment positioning
        is_first_segment = ctx.segment_params.get("is_first_segment", ctx.segment_idx == 0)

        # If segments are not chained (VACE only), force "from scratch" behavior
        # This makes it ignore previous video context even if not the first segment
        chain_segments = ctx.orchestrator_details.get("chain_segments", True)

        if not chain_segments:
             is_first_segment_from_scratch = True
             travel_logger.debug(f"[INDEPENDENT_SEGMENTS] chain_segments=False: Forcing is_first_segment_from_scratch=True for segment {ctx.segment_idx}", task_id=ctx.task_id)
        else:
             is_first_segment_from_scratch = is_first_segment and not ctx.orchestrator_details.get("continue_from_video_resolved_path")

        # Calculate end anchor image index
        # Priority: consolidated_end_anchor > individual_segment detection > segment_idx + 1
        consolidated_end_anchor = ctx.segment_params.get("consolidated_end_anchor_idx")
        individual_params = ctx.segment_params.get("individual_segment_params", {})
        individual_images = individual_params.get("input_image_paths_resolved", [])

        if consolidated_end_anchor is not None:
            end_anchor_img_path_str_idx = consolidated_end_anchor
            travel_logger.debug(f"[CONSOLIDATED_SEGMENT] Using consolidated end anchor index {consolidated_end_anchor} for segment {ctx.segment_idx}", task_id=ctx.task_id)
        elif individual_images:
            # Individual segment mode: images are [start, end], so end anchor is always index 1
            end_anchor_img_path_str_idx = 1
            travel_logger.debug(f"[INDIVIDUAL_SEGMENT] Using end anchor index 1 for individual segment {ctx.segment_idx} (has {len(individual_images)} images)", task_id=ctx.task_id)
        else:
            end_anchor_img_path_str_idx = ctx.segment_idx + 1

        travel_logger.debug(f"[STRUCTURE_CONFIG] Segment {ctx.segment_idx}: {structure_config}", task_id=ctx.task_id)

        # Extract values from config for backward compatibility with existing code
        structure_video_path = structure_config.videos[0].path if structure_config.videos else None
        structure_video_treatment = structure_config.videos[0].treatment if structure_config.videos else "adjust"
        structure_type = structure_config.legacy_structure_type
        structure_video_motion_strength = structure_config.strength
        structure_canny_intensity = structure_config.canny_intensity
        structure_depth_contrast = structure_config.depth_contrast
        structure_guidance_video_url = structure_config.guidance_video_url
        structure_guidance_frame_offset = structure_config._frame_offset

        # Check for multi-structure video config
        # If present and no pre-computed guidance URL, compute segment's portion locally
        structure_videos = [v.to_dict() for v in structure_config.videos] if structure_config.videos else None

        # Store the detected structure_type for use in process_segment return
        proc._detected_structure_type = structure_type

        if structure_videos and not structure_guidance_video_url:
            if travel_guidance_config is not None:
                local_guidance_path = _build_local_travel_guidance_video(proc, travel_guidance_config)
            else:
                travel_logger.debug(f"[STRUCTURE_VIDEO] Segment {ctx.segment_idx}: Found structure_videos array, computing segment guidance locally", task_id=ctx.task_id)

                from source.media.structure import extract_segment_structure_guidance

                # Get segment layout from orchestrator payload
                segment_frames_expanded = ctx.orchestrator_details.get("segment_frames_expanded", [ctx.total_frames_for_segment])
                frame_overlap_expanded = ctx.orchestrator_details.get("frame_overlap_expanded", [0])

                # Generate path for segment's guidance video
                segment_guidance_filename = f"segment_guidance_{ctx.segment_idx}_{uuid.uuid4().hex[:6]}.mp4"
                segment_guidance_path = ctx.segment_processing_dir / segment_guidance_filename

                # Extract this segment's guidance
                local_guidance_path = extract_segment_structure_guidance(
                    structure_videos=structure_videos,
                    segment_index=ctx.segment_idx,
                    segment_frames_expanded=segment_frames_expanded,
                    frame_overlap_expanded=frame_overlap_expanded,
                    target_resolution=ctx.parsed_res_wh,
                    target_fps=ctx.orchestrator_details.get("fps_helpers", 16),
                    output_path=segment_guidance_path,
                    motion_strength=structure_video_motion_strength,
                    canny_intensity=structure_canny_intensity,
                    depth_contrast=structure_depth_contrast,
                    download_dir=ctx.segment_processing_dir)

            if local_guidance_path and Path(local_guidance_path).exists():
                travel_logger.debug(f"[STRUCTURE_VIDEO] Created local segment guidance: {local_guidance_path}", task_id=ctx.task_id)
                # Use as a local path string (NOT file://). The downstream extractor expects a filesystem path.
                structure_guidance_video_url = str(local_guidance_path)
                structure_guidance_frame_offset = 0  # Local guidance starts at frame 0
            else:
                # No overlap with any structure_videos config = no guidance for this segment
                # This is intentional, not a failure - segment proceeds without structure guidance
                travel_logger.debug(f"[STRUCTURE_VIDEO] Segment {ctx.segment_idx}: No overlap with structure_videos, proceeding without structure guidance", task_id=ctx.task_id)

        # Download structure video if it's a URL (defensive fallback if orchestrator didn't download)
        # Note: If structure_guidance_video_url is provided, this is not strictly needed as segments will use the pre-warped video
        if structure_video_path:
            structure_video_path = download_video_if_url(
                structure_video_path,
                download_target_dir=ctx.segment_processing_dir,
                task_id_for_logging=ctx.task_id,
                descriptive_name=f"structure_video_seg{ctx.segment_idx}"
            )

        # Log which structure type is being used
        if structure_video_path or structure_guidance_video_url:
            travel_logger.debug(f"[STRUCTURE_VIDEO] Segment {ctx.segment_idx} using structure type: {structure_type}", task_id=ctx.task_id)

        # Detect if this is a single image journey (1 image, no continuation)
        is_single_image_journey = proc._detect_single_image_journey()

        # Create guide video using shared function
        guide_video_path = create_guide_video_for_travel_segment(
            segment_idx_for_logging=ctx.segment_idx,
            end_anchor_image_index=end_anchor_img_path_str_idx,
            is_first_segment_from_scratch=is_first_segment_from_scratch,
            total_frames_for_segment=ctx.total_frames_for_segment,
            parsed_res_wh=ctx.parsed_res_wh,
            fps_helpers=ctx.orchestrator_details.get("fps_helpers", 16),
            input_images_resolved_for_guide=input_images_resolved_for_guide,
            path_to_previous_segment_video_output_for_guide=path_to_previous_segment_video_output_for_guide,
            output_target_dir=ctx.segment_processing_dir,
            guide_video_base_name=guide_video_base_name,
            segment_image_download_dir=ctx.segment_processing_dir,  # Use processing dir for downloads
            task_id_for_logging=ctx.task_id,
            orchestrator_details=ctx.orchestrator_details,
            segment_params=ctx.segment_params,
            single_image_journey=is_single_image_journey,  # Detect single image journeys correctly
            predefined_output_path=guide_video_final_path,
            structure_video_path=structure_video_path,
            structure_video_treatment=structure_video_treatment,
            structure_type=structure_type,
            structure_video_motion_strength=structure_video_motion_strength,
            structure_canny_intensity=structure_canny_intensity,
            structure_depth_contrast=structure_depth_contrast,
            structure_guidance_video_url=structure_guidance_video_url,
            structure_guidance_frame_offset=structure_guidance_frame_offset,
            # For uni3c mode, black out end frame so i2v handles it alone (no conflicting guidance)
            exclude_end_for_controlnet=(structure_type == "uni3c"))

        if guide_video_path and Path(guide_video_path).exists():
            travel_logger.debug(f"[GUIDE_DEBUG] Successfully created guide video: {guide_video_path}", task_id=ctx.task_id)
            return Path(guide_video_path)
        else:
            travel_logger.error(f"[GUIDE_ERROR] Guide video creation returned: {guide_video_path}", task_id=ctx.task_id)

            if guide_required:
                raise ValueError(f"Guide video is required for model '{ctx.model_name}' but creation failed")

            return None

    except (OSError, ValueError, RuntimeError) as e_guide:
        travel_logger.error(f"[GUIDE_ERROR] Guide video creation failed: {e_guide}", task_id=ctx.task_id, exc_info=True)

        if guide_required:
            raise ValueError(f"Guide video is required for model '{ctx.model_name}' but creation failed: {e_guide}") from e_guide

        return None
