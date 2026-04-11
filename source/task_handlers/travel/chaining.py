"""Travel chaining handler - post-WGP processing for travel segments."""

import shutil
import traceback
from pathlib import Path
import uuid
from datetime import datetime

# Import structured logging
from ...core.log import travel_logger

from ... import db_operations as db_ops
from ...core.db import config as db_config
from ...utils import (
    get_video_frame_count_and_fps,
    prepare_output_path,
    wait_for_file_stable)
from ...media.video import (
    apply_saturation_to_video_ffmpeg,
    apply_brightness_to_video_frames,
    apply_color_matching_to_video,
    overlay_start_end_images_above_video)

from .svi_config import SVI_STITCH_OVERLAP
from .debug_utils import debug_video_analysis

# --- SM_RESTRUCTURE: New function to handle chaining after WGP/Comfy sub-task ---
def _handle_travel_chaining_after_wgp(wgp_task_params: dict, actual_wgp_output_video_path: str | None, image_download_dir: Path | str | None = None, main_output_dir_base: Path | None = None) -> tuple[bool, str, str | None]:
    """
    Handles the chaining logic after a WGP  sub-task for a travel segment completes.
    This includes post-generation saturation and enqueuing the next segment or stitch task.
    Returns: (success_bool, message_str, final_video_path_for_db_str_or_none)
    The third element is the path that should be considered the definitive output of the WGP task
    (e.g., path to saturated video if saturation was applied).

    Args:
        main_output_dir_base: Worker's base output directory for resolving relative paths
    """
    chain_details = wgp_task_params.get("travel_chain_details")
    wgp_task_id = wgp_task_params.get("task_id", "unknown_wgp_task")

    if not chain_details:
        return False, f"Task {wgp_task_id}: Missing travel_chain_details. Cannot proceed with chaining.", None

    # actual_wgp_output_video_path comes from process_single_task.
    # Path is an absolute path or URL from Supabase.
    if not actual_wgp_output_video_path: # Check if it's None or empty string
        return False, f"Task {wgp_task_id}: WGP output video path is None or empty. Cannot chain.", None

    # This variable will track the absolute path of the video as it gets processed.
    video_to_process_abs_path: Path
    # This will hold the path to be stored in the DB (absolute path or URL)
    final_video_path_for_db = actual_wgp_output_video_path

    # Path is already absolute (Supabase URL or absolute path)
    video_to_process_abs_path = Path(actual_wgp_output_video_path)

    if not video_to_process_abs_path.exists():
        return False, f"Task {wgp_task_id}: Source video for chaining '{video_to_process_abs_path}' (from '{actual_wgp_output_video_path}') does not exist.", actual_wgp_output_video_path

    try:
        orchestrator_task_id_ref = chain_details["orchestrator_task_id_ref"]
        orchestrator_run_id = chain_details["orchestrator_run_id"]
        segment_idx_completed = chain_details["segment_index_completed"]
        # Support both canonical (orchestrator_details) and legacy (full_orchestrator_payload) key names
        orchestrator_details = chain_details.get("orchestrator_details") or chain_details.get("full_orchestrator_payload")
        _segment_processing_dir_for_saturation_str = chain_details["segment_processing_dir_for_saturation"]

        is_first_new_segment_after_continue = chain_details.get("is_first_new_segment_after_continue", False)

        # Determine the correct base output directory for prepare_output_path calls
        # Prefer worker's main_output_dir_base if provided, otherwise fall back to payload
        if main_output_dir_base:
            output_base_for_files = main_output_dir_base
        else:
            # Fallback: resolve from payload (may be relative)
            payload_dir_str = orchestrator_details.get("main_output_dir_for_run", "./outputs")
            payload_dir_path = Path(payload_dir_str)
            if not payload_dir_path.is_absolute():
                output_base_for_files = (Path.cwd() / payload_dir_path).resolve()
            else:
                output_base_for_files = payload_dir_path.resolve()
        is_subsequent_segment_val = chain_details.get("is_subsequent_segment", False)

        chain_length = len(orchestrator_details.get("segment_frames_expanded", []) or [])
        applied_post_steps = []

        # --- Always move WGP output to proper location first ---
        # Use consistent UUID-based naming and MOVE (not copy) to avoid duplicates
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        moved_filename = f"seg{segment_idx_completed:02d}_output_{timestamp_short}_{unique_suffix}{video_to_process_abs_path.suffix}"

        # Use prepare_output_path to ensure WGP output moves to task-type directory
        moved_video_abs_path, _ = prepare_output_path(
            task_id=wgp_task_id,
            filename=moved_filename,
            main_output_dir_base=output_base_for_files,
            task_type="travel_segment"
        )

        # MOVE (not copy) the WGP output to avoid creating duplicates
        try:
            # Ensure encoder has finished writing the source file
            wait_for_file_stable(video_to_process_abs_path, checks=3, interval=1.0)

            original_video_path = video_to_process_abs_path
            shutil.move(str(video_to_process_abs_path), str(moved_video_abs_path))
            debug_video_analysis(moved_video_abs_path, f"MOVED_WGP_OUTPUT_Seg{segment_idx_completed}", wgp_task_id)

            # Update paths for further processing
            video_to_process_abs_path = moved_video_abs_path
            final_video_path_for_db = str(moved_video_abs_path)  # Use absolute path as DB path

        except OSError as e_move:
            travel_logger.debug_anomaly(
                "CHAIN_MOVE_FAILED",
                f"Seg {segment_idx_completed}: move failed ({e_move}); using original path",
                task_id=wgp_task_id,
            )
            # If move failed, keep original paths for further processing
            final_video_path_for_db = str(video_to_process_abs_path)

        # ------------------------------------------------------------------
        # SVI Continuation Output Trim
        # ------------------------------------------------------------------
        # When using SVI continuation via `video_source`, Wan2GP's `wgp.py` prepends
        # the prefix video frames to the generated sample (so the output contains
        # both previous + new frames). For our segment-based travel pipeline we
        # want each segment output to be exactly the requested `video_length`
        # (4N+1) frames, so stitching does not double-count prior segments.
        try:
            use_svi = bool(orchestrator_details.get("use_svi") or wgp_task_params.get("use_svi") or chain_details.get("use_svi"))
            if use_svi and isinstance(segment_idx_completed, int) and segment_idx_completed > 0:
                expected_segment_frames = orchestrator_details.get("segment_frames_expanded", [])
                expected_len = expected_segment_frames[segment_idx_completed] if (
                    isinstance(expected_segment_frames, list)
                    and segment_idx_completed < len(expected_segment_frames)
                    and isinstance(expected_segment_frames[segment_idx_completed], int)
                ) else None

                if expected_len and expected_len > 0:
                    actual_frames, actual_fps = get_video_frame_count_and_fps(str(video_to_process_abs_path))
                    requested_video_length_i = None
                    overlap_size_i = None

                    # Optional: show what we actually requested from WGP (passed through task_registry)
                    try:
                        requested_video_length = wgp_task_params.get("video_length")
                        overlap_size = wgp_task_params.get("sliding_window_overlap") or SVI_STITCH_OVERLAP
                        if requested_video_length and overlap_size:
                            requested_video_length_i = int(requested_video_length)
                            overlap_size_i = int(overlap_size)
                    except (ValueError, KeyError, TypeError):
                        requested_video_length_i = None
                        overlap_size_i = None

                    travel_logger.debug(
                        f"[SVI_GROUND_TRUTH] Seg {segment_idx_completed}: "
                        f"output={video_to_process_abs_path}, actual_frames={actual_frames}, "
                        f"desired_new_frames={expected_len}, request_video_length={requested_video_length_i}, "
                        f"overlap={overlap_size_i or SVI_STITCH_OVERLAP}, "
                        f"video_source={bool(wgp_task_params.get('video_source'))}",
                        task_id=wgp_task_id,
                    )

                    if actual_frames and actual_frames > expected_len:
                        from source.media.video import extract_frame_range_to_video as extract_frame_range_to_video

                        # CRITICAL: For SVI continuation, we need to preserve the last 4 prefix frames
                        # for stitching overlap. So we keep expected_len + SVI_STITCH_OVERLAP frames.
                        # This ensures: last 4 prefix frames (overlap) + expected_len NEW frames
                        frames_to_keep = int(expected_len) + SVI_STITCH_OVERLAP
                        start_frame = max(0, int(actual_frames) - frames_to_keep)

                        trim_filename = f"seg{segment_idx_completed:02d}_trimmed_{timestamp_short}_{unique_suffix}{video_to_process_abs_path.suffix}"
                        trimmed_video_abs_path, _ = prepare_output_path(
                            task_id=wgp_task_id,
                            filename=trim_filename,
                            main_output_dir_base=output_base_for_files,
                            task_type="travel_segment"
                        )

                        fps_for_trim = float(actual_fps) if actual_fps and actual_fps > 0 else float(orchestrator_details.get("fps_helpers", 16))
                        trimmed_result = extract_frame_range_to_video(
                            input_video_path=str(video_to_process_abs_path),
                            output_video_path=str(trimmed_video_abs_path),
                            start_frame=start_frame,
                            end_frame=None,
                            fps=fps_for_trim)
                        # Verify trimmed output (debug-only)
                        trimmed_frames, trimmed_fps = get_video_frame_count_and_fps(trimmed_result)
                        debug_enabled = bool(
                            orchestrator_details.get("debug_mode_enabled", False)
                            or db_config.debug_mode
                        )
                        if debug_enabled:
                            trim_summary = {
                                "trimmed_frames": trimmed_frames,
                                "expected_frames": frames_to_keep,
                                "output": trimmed_result,
                            }

                            # Optional duplication detector: does the *final frame* appear earlier?
                            try:
                                import hashlib
                                import cv2

                                cap = cv2.VideoCapture(str(trimmed_result))
                                if cap.isOpened() and trimmed_frames and trimmed_frames > 1:
                                    last_idx = int(trimmed_frames) - 1
                                    # Sample a small set of earlier frames to avoid heavy IO
                                    sample_idxs = list(range(0, min(32, last_idx)))
                                    # Always include the overlap boundary and near-end region
                                    sample_idxs += [SVI_STITCH_OVERLAP - 1] if SVI_STITCH_OVERLAP - 1 < last_idx else []
                                    sample_idxs += list(range(max(0, last_idx - 24), last_idx))
                                    sample_idxs = sorted(set([i for i in sample_idxs if 0 <= i < last_idx]))

                                    def _hash_frame(frame_bgr) -> str:
                                        return hashlib.md5(frame_bgr.tobytes()).hexdigest()[:12]

                                    # Read last frame
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
                                    ok_last, fr_last = cap.read()
                                    last_hash = _hash_frame(fr_last) if ok_last and fr_last is not None else None

                                    dup_hits = []
                                    if last_hash:
                                        for i in sample_idxs:
                                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                                            ok, fr = cap.read()
                                            if not ok or fr is None:
                                                continue
                                            if _hash_frame(fr) == last_hash:
                                                dup_hits.append(i)
                                                if len(dup_hits) >= 5:
                                                    break
                                    cap.release()

                                    if last_hash:
                                        trim_summary["final_frame_hash"] = last_hash
                                        trim_summary["dup_hits"] = dup_hits[:5]
                            except (OSError, ValueError, RuntimeError):
                                pass
                            travel_logger.debug(
                                f"[SVI_GROUND_TRUTH] Seg {segment_idx_completed}: trim_result={trim_summary}, "
                                f"discard_prefix={start_frame}, keep={frames_to_keep}",
                                task_id=wgp_task_id,
                            )
                        debug_video_analysis(Path(trimmed_result), f"SVI_TRIMMED_Seg{segment_idx_completed}", wgp_task_id)
                        # Replace the working video with the trimmed one
                        try:
                            if video_to_process_abs_path.exists():
                                video_to_process_abs_path.unlink()
                        except OSError as e_unlink:
                            travel_logger.debug_anomaly("SVI_PREFIX_TRIM", f"Seg {segment_idx_completed}: Could not remove old working video {video_to_process_abs_path}: {e_unlink}", task_id=wgp_task_id)
                        video_to_process_abs_path = Path(trimmed_result)
                        final_video_path_for_db = str(video_to_process_abs_path)
        except (OSError, ValueError, RuntimeError) as e_svi_trim:
            travel_logger.debug_anomaly("SVI_PREFIX_TRIM", f"WARNING: Exception while trimming SVI segment output: {e_svi_trim}", task_id=wgp_task_id)

        # --- Post-generation Processing Chain ---
        # Saturation and Brightness are only applied to segments AFTER the first one.
        if is_subsequent_segment_val or is_first_new_segment_after_continue:

            # --- 1. Saturation ---
            sat_level = orchestrator_details.get("after_first_post_generation_saturation")
            if sat_level is not None and isinstance(sat_level, (float, int)) and sat_level >= 0.0 and abs(sat_level - 1.0) > 1e-6:
                sat_filename = f"{wgp_task_id}_seg{segment_idx_completed}_saturated.mp4"
                saturated_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=sat_filename,
                    main_output_dir_base=output_base_for_files,
                    task_type="travel_segment"
                )

                if apply_saturation_to_video_ffmpeg(str(video_to_process_abs_path), saturated_video_output_abs_path, sat_level):
                    debug_video_analysis(saturated_video_output_abs_path, f"SATURATED_Seg{segment_idx_completed}", wgp_task_id)
                    _cleanup_intermediate_video(orchestrator_details, video_to_process_abs_path, segment_idx_completed, "raw", wgp_task_id)

                    video_to_process_abs_path = saturated_video_output_abs_path
                    final_video_path_for_db = new_db_path
                    applied_post_steps.append(f"saturation:{sat_level}")
                else:
                    travel_logger.warning(f"Saturation failed for segment {segment_idx_completed}", task_id=wgp_task_id)
                    travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): Saturation failed. Continuing with unsaturated video.", task_id=wgp_task_id)

            # --- 2. Brightness ---
            brightness_adjust = orchestrator_details.get("after_first_post_generation_brightness", 0.0)
            if isinstance(brightness_adjust, (float, int)) and abs(brightness_adjust) > 1e-6:
                bright_filename = f"{wgp_task_id}_seg{segment_idx_completed}_brightened.mp4"
                brightened_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=bright_filename,
                    main_output_dir_base=output_base_for_files,
                    task_type="travel_segment"
                )

                processed_video = apply_brightness_to_video_frames(str(video_to_process_abs_path), brightened_video_output_abs_path, brightness_adjust, wgp_task_id)

                if processed_video and processed_video.exists():
                    debug_video_analysis(brightened_video_output_abs_path, f"BRIGHTENED_Seg{segment_idx_completed}", wgp_task_id)
                    _cleanup_intermediate_video(orchestrator_details, video_to_process_abs_path, segment_idx_completed, "saturated", wgp_task_id)

                    video_to_process_abs_path = brightened_video_output_abs_path
                    final_video_path_for_db = new_db_path
                    applied_post_steps.append(f"brightness:{brightness_adjust}")
                else:
                    travel_logger.warning(f"Brightness adjustment failed for segment {segment_idx_completed}", task_id=wgp_task_id)
                    travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): Brightness adjustment failed. Continuing with previous video version.", task_id=wgp_task_id)

        # --- 3. Color Matching (Applied to all segments if enabled) ---
        if chain_details.get("colour_match_videos"):
            start_ref = chain_details.get("cm_start_ref_path")
            end_ref = chain_details.get("cm_end_ref_path")
            travel_logger.debug(
                f"Color matching requested for segment {segment_idx_completed}: start_ref={start_ref}, end_ref={end_ref}",
                task_id=wgp_task_id,
            )

            if start_ref and end_ref and Path(start_ref).exists() and Path(end_ref).exists():
                cm_filename = f"{wgp_task_id}_seg{segment_idx_completed}_colormatched.mp4"
                cm_video_output_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=cm_filename,
                    main_output_dir_base=output_base_for_files,
                    task_type="travel_segment"
                )

                matched_video_path = apply_color_matching_to_video(
                    str(video_to_process_abs_path),
                    start_ref,
                    end_ref,
                    str(cm_video_output_abs_path),
                    lambda msg: travel_logger.debug(msg, task_id=wgp_task_id)
                )

                if matched_video_path and Path(matched_video_path).exists():
                    debug_video_analysis(Path(matched_video_path), f"COLORMATCHED_Seg{segment_idx_completed}", wgp_task_id)
                    _cleanup_intermediate_video(orchestrator_details, video_to_process_abs_path, segment_idx_completed, "pre-colormatch", wgp_task_id)

                    video_to_process_abs_path = Path(matched_video_path)
                    final_video_path_for_db = new_db_path
                    applied_post_steps.append("color_match")
                else:
                    travel_logger.warning(f"Color matching failed for segment {segment_idx_completed}", task_id=wgp_task_id)
                    travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): Color matching failed. Continuing with previous video version.", task_id=wgp_task_id)
            else:
                travel_logger.warning(f"Color matching skipped - missing or invalid reference images", task_id=wgp_task_id)
                travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): Skipping color matching due to missing or invalid reference image paths.", task_id=wgp_task_id)

        # --- 4. Optional: Overlay start/end images above the video ---
        if chain_details.get("show_input_images"):
            banner_start = chain_details.get("start_image_path")
            banner_end = chain_details.get("end_image_path")
            if banner_start and banner_end and Path(banner_start).exists() and Path(banner_end).exists():
                banner_filename = f"{wgp_task_id}_seg{segment_idx_completed}_banner{video_to_process_abs_path.suffix}"
                banner_video_abs_path, new_db_path = prepare_output_path(
                    task_id=wgp_task_id,
                    filename=banner_filename,
                    main_output_dir_base=output_base_for_files,
                    task_type="travel_segment"
                )

                if overlay_start_end_images_above_video(
                    start_image_path=banner_start,
                    end_image_path=banner_end,
                    input_video_path=str(video_to_process_abs_path),
                    output_video_path=str(banner_video_abs_path)):
                    debug_video_analysis(banner_video_abs_path, f"BANNER_OVERLAY_Seg{segment_idx_completed}", wgp_task_id)
                    _cleanup_intermediate_video(orchestrator_details, video_to_process_abs_path, segment_idx_completed, "pre-banner", wgp_task_id)

                    video_to_process_abs_path = banner_video_abs_path
                    final_video_path_for_db = new_db_path
                    applied_post_steps.append("banner_overlay")
                else:
                    travel_logger.warning(f"Banner overlay failed for segment {segment_idx_completed}", task_id=wgp_task_id)
                    travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): Banner overlay failed. Keeping previous video version.", task_id=wgp_task_id)
            else:
                travel_logger.warning(f"Banner overlay skipped - missing valid start/end images", task_id=wgp_task_id)
                travel_logger.debug_anomaly("WARNING", f"Chain (Seg {segment_idx_completed}): show_input_images enabled but valid start/end images not found.", task_id=wgp_task_id)

        # One consolidated CHAIN card summarising the whole post-WGP chain pipeline.
        from source.core.log.display_names import rel_path
        travel_logger.debug_block(
            "CHAIN",
            {
                "segment": segment_idx_completed,
                "run": orchestrator_run_id,
                "chain_length": chain_length,
                "post_steps": applied_post_steps or ["none"],
                "final_path": rel_path(final_video_path_for_db) if final_video_path_for_db else None,
            },
            task_id=wgp_task_id,
        )
        debug_video_analysis(video_to_process_abs_path, f"FINAL_CHAINED_Seg{segment_idx_completed}", wgp_task_id)
        msg = (
            f"Chain (Seg {segment_idx_completed}): Post-WGP processing complete. "
            f"steps={applied_post_steps or ['none']}, final_path={final_video_path_for_db}"
        )
        return True, msg, str(final_video_path_for_db)

    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e_chain:
        error_msg = f"Chain (Seg {chain_details.get('segment_index_completed', 'N/A')} for WGP {wgp_task_id}): Failed during chaining: {e_chain}"
        travel_logger.error(error_msg, task_id=wgp_task_id)
        travel_logger.debug(traceback.format_exc(), task_id=wgp_task_id)

        # Notify orchestrator of chaining failure
        orchestrator_task_id_ref = chain_details.get("orchestrator_task_id_ref") if chain_details else None
        if orchestrator_task_id_ref:
            try:
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg[:500]  # Truncate to avoid DB overflow
                )
                travel_logger.debug(f"Chain: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to chaining failure", task_id=wgp_task_id)
            except (RuntimeError, ValueError, OSError) as e_orch:
                travel_logger.debug(f"Chain: Warning - could not update orchestrator status: {e_orch}", task_id=wgp_task_id)

        return False, error_msg, str(final_video_path_for_db) # Return path as it was before error

def _cleanup_intermediate_video(orchestrator_payload, video_path: Path, segment_idx: int, stage: str, task_id: str = ""):
    """Helper to cleanup intermediate video files during chaining."""
    # Delete intermediates **only** when every cleanup-bypass flag is false.
    # That now includes the worker-server global debug flag (db_config.debug_mode)
    # so that running the server with --debug automatically preserves files.
    if (
        not orchestrator_payload.get("skip_cleanup_enabled", False)
        and not orchestrator_payload.get("debug_mode_enabled", False)
        and not db_config.debug_mode
        and video_path.exists()
    ):
        try:
            video_path.unlink()
            travel_logger.debug(f"Chain (Seg {segment_idx}): Removed intermediate '{stage}' video {video_path}", task_id=task_id)
        except OSError as e_del:
            travel_logger.debug(f"Chain (Seg {segment_idx}): Warning - could not remove intermediate video {video_path}: {e_del}", task_id=task_id)


# Public alias for cross-module use.
handle_travel_chaining_after_wgp = _handle_travel_chaining_after_wgp
