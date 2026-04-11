"""Travel stitch task handler - stitches travel segments into final video."""

import shutil
import traceback
from pathlib import Path
import time
import uuid
from datetime import datetime

try:
    import cv2
    _COLOR_MATCH_DEPS_AVAILABLE = True
except ImportError:
    _COLOR_MATCH_DEPS_AVAILABLE = False

# Import structured logging
from ...core.log import travel_logger, safe_json_repr

from ... import db_operations as db_ops
from ...core.db import config as db_config
from ...utils import (
    generate_unique_task_id,
    get_video_frame_count_and_fps,
    parse_resolution,
    prepare_output_path,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
    wait_for_file_stable)
from .orchestrator import DEFAULT_SEED_BASE, UPSCALE_SEED_OFFSET

from ...media.video import (
    add_audio_to_video,
    extract_frames_from_video,
    create_video_from_frames_list,
    cross_fade_overlap_frames)
from ...media.video.ffmpeg_ops import mux_audio_from_segments
from ...media.video.ingest import download_file_if_url

from .debug_utils import debug_video_analysis, flush_ram_snapshots, log_ram_usage
from .ffmpeg_fallback import attempt_ffmpeg_crossfade_fallback


def _as_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
    return normalized


def _get_segment_output_source(entry) -> str:
    if isinstance(entry, tuple) and len(entry) >= 2:
        return entry[1]
    if isinstance(entry, dict):
        value = entry.get("video_file_path") or entry.get("output_location")
        return value if isinstance(value, str) else ""
    return ""


def _parse_explicit_clip_request(stitch_params: dict) -> dict | None:
    clip_urls = _as_string_list(stitch_params.get("clip_urls"))
    if not clip_urls:
        return None

    if len(clip_urls) < 2:
        raise ValueError("Explicit stitch mode requires at least two clip_urls.")

    raw_overlaps = stitch_params.get("frame_overlap_settings_expanded")
    if not isinstance(raw_overlaps, list):
        raise ValueError("Explicit stitch mode requires frame_overlap_settings_expanded.")

    overlaps = []
    for index, overlap in enumerate(raw_overlaps):
        if not isinstance(overlap, (int, float)):
            raise ValueError(f"Explicit stitch overlap at index {index} must be numeric.")
        overlap_int = int(overlap)
        if overlap_int <= 0:
            raise ValueError(f"Explicit stitch overlap at index {index} must be positive.")
        overlaps.append(overlap_int)

    if len(overlaps) != len(clip_urls) - 1:
        raise ValueError(
            "Explicit stitch mode requires one frame overlap value per clip boundary."
        )

    return {
        "clip_urls": clip_urls,
        "expanded_frame_overlaps": overlaps,
        "audio_url": stitch_params.get("audio_url"),
    }

def _handle_travel_stitch_task(task_params_from_db: dict, main_output_dir_base: Path, stitch_task_id_str: str):
    log_ram_usage("Stitch start", task_id=stitch_task_id_str)
    stitch_params = task_params_from_db  # Contains orchestrator_details
    stitch_success = False
    final_video_location_for_db = None

    try:
        # --- 1. Initialization & Parameter Extraction ---
        explicit_clip_request = _parse_explicit_clip_request(stitch_params)
        explicit_clips_mode = explicit_clip_request is not None
        orchestrator_task_id_ref = stitch_params.get("orchestrator_task_id_ref")
        orchestrator_run_id = stitch_params.get("orchestrator_run_id")
        # Support both canonical (orchestrator_details) and legacy (full_orchestrator_payload) key names
        orchestrator_details = stitch_params.get("orchestrator_details") or stitch_params.get("full_orchestrator_payload") or {}
        if not isinstance(orchestrator_details, dict):
            orchestrator_details = {}
        explicit_audio_url = explicit_clip_request.get("audio_url") if explicit_clip_request else stitch_params.get("audio_url")

        travel_logger.debug_block(
            "STITCH_SETUP",
            {
                "task": stitch_task_id_str,
                "run_id": orchestrator_run_id,
                "orchestrator_task_id": orchestrator_task_id_ref,
                "clips_mode": "explicit" if explicit_clips_mode else "orchestrated",
                "orchestrator_details": bool(orchestrator_details),
                "param_keys": sorted(stitch_params.keys()),
                "audio": explicit_audio_url,
            },
            task_id=stitch_task_id_str,
        )

        if not explicit_clips_mode and not all([orchestrator_task_id_ref, orchestrator_run_id, orchestrator_details]):
            msg = f"Stitch task {stitch_task_id_str} missing critical orchestrator refs or orchestrator_details."
            travel_logger.error(msg, task_id=stitch_task_id_str)
            return False, msg

        if not explicit_clips_mode:
            # Validate required keys early to produce clear errors instead of KeyError
            _stitch_required = {"num_new_segments_to_generate", "parsed_resolution_wh", "segment_frames_expanded"}
            _stitch_missing = _stitch_required - orchestrator_details.keys()
            if _stitch_missing:
                msg = f"Stitch task {stitch_task_id_str}: orchestrator_details missing required keys: {sorted(_stitch_missing)}"
                travel_logger.error(msg, task_id=stitch_task_id_str)
                return False, msg

        project_id_for_stitch = stitch_params.get("project_id")
        current_run_base_output_dir_str = stitch_params.get("current_run_base_output_dir",
                                                            orchestrator_details.get("main_output_dir_for_run", str(main_output_dir_base.resolve())))
        current_run_base_output_dir = Path(current_run_base_output_dir_str)

        # Use the base directory directly without creating stitch-specific subdirectories
        stitch_processing_dir = current_run_base_output_dir
        stitch_processing_dir.mkdir(parents=True, exist_ok=True)
        travel_logger.debug(f"Stitch Task {stitch_task_id_str}: Processing in {stitch_processing_dir.resolve()}", task_id=stitch_task_id_str)

        if explicit_clips_mode:
            num_expected_new_segments = len(explicit_clip_request["clip_urls"])
            parsed_res_wh_from_payload = None
            parsed_res_wh = None
            final_fps = stitch_params.get("fps") or 0
            expanded_frame_overlaps = explicit_clip_request["expanded_frame_overlaps"]
            crossfade_sharp_amt = stitch_params.get("crossfade_sharp_amt", 0.3)
            initial_continued_video_path_str = None
            travel_logger.debug(
                f"[STITCH_DEBUG] explicit_mode clips={num_expected_new_segments}, fps={final_fps}, "
                f"overlaps={expanded_frame_overlaps}",
                task_id=stitch_task_id_str,
            )
        else:
            num_expected_new_segments = orchestrator_details["num_new_segments_to_generate"]

            # Parse resolution from payload - but DON'T snap to model grid yet!
            # The actual resolution will be determined from the input segment videos.
            # Snapping is only needed for generation, not for stitching existing videos.
            parsed_res_wh_str = orchestrator_details["parsed_resolution_wh"]
            try:
                parsed_res_wh_from_payload = parse_resolution(parsed_res_wh_str)
                if parsed_res_wh_from_payload is None:
                    raise ValueError(f"parse_resolution returned None for input: {parsed_res_wh_str}")
                # NOTE: We use this as a fallback only. Actual resolution comes from input videos.
            except (ValueError, KeyError, TypeError) as e_parse_res_stitch:
                msg = f"Stitch Task {stitch_task_id_str}: Invalid format or error parsing parsed_resolution_wh '{parsed_res_wh_str}': {e_parse_res_stitch}"
                travel_logger.error(msg, task_id=stitch_task_id_str)
                return False, msg
            travel_logger.debug(
                f"[STITCH_DEBUG] payload_resolution={parsed_res_wh_from_payload}, use_actual_video_resolution=True",
                task_id=stitch_task_id_str,
            )

            # Placeholder - will be set from actual input video after loading
            parsed_res_wh = None

            final_fps = orchestrator_details.get("fps_helpers", 24)
            # CRITICAL: Use stitch_params overlay settings, NOT the orchestrator's default!
            # For SVI mode, frame_overlap_settings_expanded contains [4, 4, ...] (SVI_STITCH_OVERLAP)
            # For VACE mode, it contains the configured overlap values
            # Fallback to orchestrator's frame_overlap_expanded only if not provided
            expanded_frame_overlaps = stitch_params.get("frame_overlap_settings_expanded") or orchestrator_details.get("frame_overlap_expanded", [])
            crossfade_sharp_amt = orchestrator_details.get("crossfade_sharp_amt", 0.3)
            initial_continued_video_path_str = orchestrator_details.get("continue_from_video_resolved_path")
            travel_logger.debug(
                f"[STITCH_DEBUG] orchestrated_mode expected_segments={num_expected_new_segments}, "
                f"fps={final_fps}, overlap_count={len(expanded_frame_overlaps)}",
                task_id=stitch_task_id_str,
            )

        travel_logger.debug(
            f"Stitch overlap settings: count={len(expanded_frame_overlaps)}, sample={expanded_frame_overlaps[:5]}",
            task_id=stitch_task_id_str,
        )

        # Extract upscale parameters
        upscale_factor = orchestrator_details.get("upscale_factor", 0.0) # Default to 0.0 if not present
        upscale_model_name = orchestrator_details.get("upscale_model_name") # Default to None if not present

        # --- 2. Collect Paths to All Segment Videos ---
        segment_video_paths_for_stitch = []
        if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists():
            travel_logger.debug(f"Stitch: Prepending initial continued video: {initial_continued_video_path_str}", task_id=stitch_task_id_str)
            # Check the continue video properties (resolution comparison deferred until actual resolution is determined)
            cap = cv2.VideoCapture(str(initial_continued_video_path_str))
            if cap.isOpened():
                continue_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                continue_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                continue_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                travel_logger.debug(f"Stitch: Continue video properties - Resolution: {continue_width}x{continue_height}, Frames: {continue_frame_count}", task_id=stitch_task_id_str)
                # Note: Resolution will be determined from first video in list (which will be this continue video)
            else:
                travel_logger.debug(f"Stitch: ERROR - Could not open continue video for property check", task_id=stitch_task_id_str)
            segment_video_paths_for_stitch.append(str(Path(initial_continued_video_path_str).resolve()))

        if explicit_clips_mode:
            completed_segment_outputs_from_db = list(enumerate(explicit_clip_request["clip_urls"]))
            travel_logger.debug(
                f"Explicit stitch mode using {len(completed_segment_outputs_from_db)} clip URLs directly",
                task_id=stitch_task_id_str,
            )
        else:
            # Fetch completed segments with a small retry loop to handle race conditions
            max_stitch_fetch_retries = 6  # Allow up to ~18s total wait
            completed_segment_outputs_from_db = []

            retry_results = []

            for attempt in range(max_stitch_fetch_retries):
                try:
                    completed_segment_outputs_from_db = db_ops.get_completed_segment_outputs_for_stitch(orchestrator_run_id, project_id=project_id_for_stitch) or []
                except (RuntimeError, ValueError, OSError) as e_db_query:
                    travel_logger.error(f"DB query failed: {e_db_query}", task_id=stitch_task_id_str)
                    completed_segment_outputs_from_db = []

                retry_results.append(len(completed_segment_outputs_from_db))

                if len(completed_segment_outputs_from_db) >= num_expected_new_segments:
                    break
                if attempt < max_stitch_fetch_retries - 1:  # Don't sleep after the last attempt
                    time.sleep(3)
            travel_logger.debug(
                f"[STITCH_DEBUG] fetch_summary expected={num_expected_new_segments}, attempts={len(retry_results)}, "
                f"results={retry_results}, final_rows={len(completed_segment_outputs_from_db)}",
                task_id=stitch_task_id_str,
            )
        travel_logger.debug(
            f"Stitch Task {stitch_task_id_str}: completed_segments_ready={len(completed_segment_outputs_from_db)}",
            task_id=stitch_task_id_str,
        )

        # ------------------------------------------------------------------
        # 2b. Resolve each returned video path (relative path or URL)
        # ------------------------------------------------------------------
        path_resolution_stats = {
            "empty": 0,
            "relative": 0,
            "remote": 0,
            "absolute": 0,
            "downloaded": 0,
            "resolved": 0,
            "missing": 0,
        }
        derived_segment_frames = []
        for seg_idx, video_path_str_from_db in completed_segment_outputs_from_db:
            resolved_video_path_for_stitch: Path | None = None

            if not video_path_str_from_db:
                path_resolution_stats["empty"] += 1
                continue

            # Case A: Relative path that starts with files/ - resolve from current working directory
            if video_path_str_from_db.startswith("files/") or video_path_str_from_db.startswith("public/files/"):
                path_resolution_stats["relative"] += 1
                # Resolve relative to current working directory
                base_dir = Path.cwd()
                absolute_path_candidate = (base_dir / "public" / video_path_str_from_db.lstrip("public/")).resolve()
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                else:
                    path_resolution_stats["missing"] += 1
                    travel_logger.warning(f"Stitch: Resolved absolute path '{absolute_path_candidate}' for segment {seg_idx} is missing.", task_id=stitch_task_id_str)

            # Case B: Remote public URL (Supabase storage)
            elif video_path_str_from_db.startswith("http"):
                path_resolution_stats["remote"] += 1
                try:
                    resolved_video_path_for_stitch = Path(download_file_if_url(
                        video_path_str_from_db,
                        stitch_processing_dir,
                        task_id_for_logging=stitch_task_id_str,
                        descriptive_name=f"seg{seg_idx:02d}",
                    )).resolve()
                    path_resolution_stats["downloaded"] += 1
                except (OSError, ValueError, RuntimeError) as e_dl:
                    travel_logger.error(f"Download failed for segment {seg_idx}: {e_dl}", task_id=stitch_task_id_str)
                    path_resolution_stats["missing"] += 1
                    travel_logger.warning(f"Stitch: Failed to download remote video for segment {seg_idx}: {e_dl}", task_id=stitch_task_id_str)

            # Case C: Provided absolute/local path
            else:
                path_resolution_stats["absolute"] += 1
                absolute_path_candidate = Path(video_path_str_from_db).resolve()
                if absolute_path_candidate.exists() and absolute_path_candidate.is_file():
                    resolved_video_path_for_stitch = absolute_path_candidate
                else:
                    path_resolution_stats["missing"] += 1
                    travel_logger.warning(f"Stitch: Absolute path '{absolute_path_candidate}' for segment {seg_idx} does not exist or is not a file.", task_id=stitch_task_id_str)

            if resolved_video_path_for_stitch is not None:
                segment_video_paths_for_stitch.append(str(resolved_video_path_for_stitch))
                path_resolution_stats["resolved"] += 1
                try:
                    frame_count, clip_fps = get_video_frame_count_and_fps(str(resolved_video_path_for_stitch))
                    derived_segment_frames.append(frame_count or 0)
                    if explicit_clips_mode and not final_fps and clip_fps:
                        final_fps = clip_fps
                except (OSError, ValueError, RuntimeError):
                    derived_segment_frames.append(0)

            else:
                travel_logger.warning(f"Unable to resolve video for segment {seg_idx}; will be excluded from stitching.", task_id=stitch_task_id_str)

        travel_logger.debug(
            f"[STITCH_DEBUG] Path resolution summary: resolved={path_resolution_stats['resolved']}, "
            f"downloaded={path_resolution_stats['downloaded']}, missing={path_resolution_stats['missing']}, "
            f"empty={path_resolution_stats['empty']}, relative={path_resolution_stats['relative']}, "
            f"remote={path_resolution_stats['remote']}, absolute={path_resolution_stats['absolute']}, "
            f"first_video={segment_video_paths_for_stitch[0] if segment_video_paths_for_stitch else 'none'}",
            task_id=stitch_task_id_str,
        )
        if explicit_clips_mode:
            orchestrator_details["segment_frames_expanded"] = derived_segment_frames
        expected_segment_frames = orchestrator_details["segment_frames_expanded"]
        frame_count_mismatches = []
        frame_count_errors = []
        for idx, video_path in enumerate(segment_video_paths_for_stitch):
            try:
                frame_count, fps = get_video_frame_count_and_fps(video_path)
                expected_frames = expected_segment_frames[idx] if idx < len(expected_segment_frames) else "unknown"
                if expected_frames != "unknown" and frame_count != expected_frames:
                    frame_count_mismatches.append(
                        f"{idx}:expected={expected_frames},actual={frame_count},fps={fps}"
                    )
            except (OSError, ValueError, RuntimeError) as e_debug:
                frame_count_errors.append(f"{idx}:{type(e_debug).__name__}")
        travel_logger.debug(
            f"Pre-stitch frame summary: videos={len(segment_video_paths_for_stitch)}, "
            f"mismatches={frame_count_mismatches or 'none'}, errors={frame_count_errors or 'none'}",
            task_id=stitch_task_id_str,
        )

        total_videos_for_stitch = (1 if initial_continued_video_path_str and Path(initial_continued_video_path_str).exists() else 0) + num_expected_new_segments
        if len(segment_video_paths_for_stitch) < total_videos_for_stitch:
            # This is a warning because some segments might have legitimately failed and been skipped by their handlers.
            # The stitcher should proceed with what it has, unless it has zero or one video when multiple were expected.
            travel_logger.warning(f"Stitch: Expected {total_videos_for_stitch} videos for stitch, but found {len(segment_video_paths_for_stitch)}. Stitching with available videos.", task_id=stitch_task_id_str)

        if not segment_video_paths_for_stitch:
            travel_logger.error(f"Stitch: No valid segment videos found to stitch. DB returned {len(completed_segment_outputs_from_db)} segments, but none resolved to valid paths.", task_id=stitch_task_id_str)
            raise ValueError("Stitch: No valid segment videos found to stitch.")
        if len(segment_video_paths_for_stitch) == 1 and total_videos_for_stitch > 1:
            travel_logger.debug(f"Stitch: Only one video segment found ({segment_video_paths_for_stitch[0]}) but {total_videos_for_stitch} were expected. Using this single video as the 'stitched' output.", task_id=stitch_task_id_str)
            # No actual stitching needed, just move/copy this single video to final dest.

        # --- 2c. Determine ACTUAL resolution from input videos ---
        # CRITICAL: Use the actual resolution of the input segment videos, not the snapped payload resolution.
        # This prevents dimension changes during stitching (e.g., 902x508 -> 896x496).
        first_video_path = segment_video_paths_for_stitch[0]
        try:
            cap = cv2.VideoCapture(first_video_path)
            if cap.isOpened():
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                parsed_res_wh = (actual_width, actual_height)
                travel_logger.debug(f"Using actual video resolution: {actual_width}x{actual_height}", task_id=stitch_task_id_str)
                if explicit_clips_mode:
                    _, detected_fps = get_video_frame_count_and_fps(str(first_video_path))
                    if not final_fps and detected_fps:
                        final_fps = detected_fps

                # Log if there's a difference from payload resolution
                if parsed_res_wh_from_payload and parsed_res_wh != parsed_res_wh_from_payload:
                    travel_logger.debug(f"Payload resolution was {parsed_res_wh_from_payload}, actual is {parsed_res_wh}", task_id=stitch_task_id_str)
            else:
                cap.release()
                raise ValueError(f"Could not open first video: {first_video_path}")
        except (OSError, ValueError, RuntimeError) as e_res:
            # Fallback to payload resolution (without snapping) if we can't read the video
            travel_logger.warning(f"Could not read resolution from video, using payload: {e_res}", task_id=stitch_task_id_str)
            parsed_res_wh = parsed_res_wh_from_payload
        if not final_fps:
            final_fps = 24

        # --- 3. Stitching (Crossfade or Concatenate) ---
        current_stitched_video_path: Path | None = None # This will hold the path to the current version of the stitched video

        if len(segment_video_paths_for_stitch) == 1:
            # If only one video, copy it directly using prepare_output_path
            source_single_video_path = Path(segment_video_paths_for_stitch[0])
            single_video_filename = f"{stitch_task_id_str}_single_video{source_single_video_path.suffix}"

            current_stitched_video_path, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=single_video_filename,
                main_output_dir_base=main_output_dir_base,
                task_type="travel_stitch"
            )
            shutil.copy2(str(source_single_video_path), str(current_stitched_video_path))
            travel_logger.debug(f"Stitch: Only one video found. Copied {source_single_video_path} to {current_stitched_video_path}", task_id=stitch_task_id_str)
        else: # More than one video, proceed with stitching logic
            num_stitch_points = len(segment_video_paths_for_stitch) - 1
            actual_overlaps_for_stitching = []
            if initial_continued_video_path_str:
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]
            else:
                actual_overlaps_for_stitching = expanded_frame_overlaps[:num_stitch_points]

            overlap_summary = {
                "videos": len(segment_video_paths_for_stitch),
                "joins": num_stitch_points,
                "overlaps": actual_overlaps_for_stitching,
                "positive_overlap": any(o > 0 for o in actual_overlaps_for_stitching),
            }
            travel_logger.debug(
                f"[STITCH_DEBUG] overlap_summary={safe_json_repr(overlap_summary)}",
                task_id=stitch_task_id_str,
            )
            if len(actual_overlaps_for_stitching) != num_stitch_points:
                travel_logger.debug(
                    f"Overlap count mismatch: overlaps={len(actual_overlaps_for_stitching)}, joins={num_stitch_points}",
                    task_id=stitch_task_id_str,
                )

            any_positive_overlap = overlap_summary["positive_overlap"]

            raw_stitched_video_filename = f"{stitch_task_id_str}_stitched_intermediate.mp4"
            path_for_raw_stitched_video, _ = prepare_output_path(
                task_id=stitch_task_id_str,
                filename=raw_stitched_video_filename,
                main_output_dir_base=main_output_dir_base,
                task_type="travel_stitch"
            )

            if any_positive_overlap:
                travel_logger.debug(
                    f"[STITCH_DEBUG] crossfade_start videos={len(segment_video_paths_for_stitch)}, "
                    f"joins={num_stitch_points}, output={path_for_raw_stitched_video}",
                    task_id=stitch_task_id_str,
                )

                # Wait for all segment videos to be stable before extracting frames
                stable_paths = []
                for idx, video_path in enumerate(segment_video_paths_for_stitch):
                    if not Path(video_path).exists():
                        stable_paths.append(False)
                        continue

                    file_stable = wait_for_file_stable(video_path, checks=5, interval=1.0)
                    stable_paths.append(bool(file_stable))

                if not all(stable_paths):
                    unstable_indices = [i for i, stable in enumerate(stable_paths) if not stable]
                    raise ValueError(f"Stitch: One or more segment videos are not stable or missing: indices {unstable_indices}")

                travel_logger.debug(
                    f"[STITCH_DEBUG] stability_check stable={sum(1 for stable in stable_paths if stable)}/{len(stable_paths)}",
                    task_id=stitch_task_id_str,
                )

                # Retry frame extraction with backoff and re-download for corrupted videos
                max_extraction_attempts = 3
                all_segment_frames_lists = None
                retry_log = []  # Track all retry attempts for detailed error reporting

                for attempt in range(max_extraction_attempts):
                    attempt_start_time = time.time()
                    all_segment_frames_lists = [extract_frames_from_video(p) for p in segment_video_paths_for_stitch]

                    failed_segments = []
                    successful_segments = []
                    for idx, frame_list in enumerate(all_segment_frames_lists):
                        if frame_list is not None and len(frame_list) > 0:
                            successful_segments.append(idx)
                        else:
                            failed_segments.append(idx)
                    travel_logger.debug(
                        f"Frame extraction attempt {attempt + 1} summary: "
                        f"successful={successful_segments}, failed={failed_segments}",
                        task_id=stitch_task_id_str,
                    )

                    # Log this attempt
                    attempt_duration = time.time() - attempt_start_time
                    attempt_info = {
                        "attempt": attempt + 1,
                        "duration_seconds": round(attempt_duration, 2),
                        "successful_segments": successful_segments,
                        "failed_segments": failed_segments,
                        "redownloads": []
                    }

                    # Check if all extractions succeeded
                    if all(f_list is not None and len(f_list) > 0 for f_list in all_segment_frames_lists):
                        retry_log.append(attempt_info)
                        break

                    # If not the last attempt, try to re-download corrupted videos before retry
                    if attempt < max_extraction_attempts - 1 and failed_segments:
                        wait_time = 3 + (attempt * 2)  # Progressive backoff: 3s, 5s, 7s
                        travel_logger.debug(f"Frame extraction failed for segments {failed_segments}. Attempting re-download and retry in {wait_time} seconds...", task_id=stitch_task_id_str)

                        # Try to re-download failed segments (only for remote URLs)
                        redownload_attempted = False
                        for failed_idx in failed_segments:
                            if failed_idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[failed_idx]
                                video_path_str_from_db = _get_segment_output_source(seg_output)

                                # Check if it's a remote URL that can be re-downloaded
                                if video_path_str_from_db.startswith("http"):
                                    try:
                                        failed_video_path = Path(segment_video_paths_for_stitch[failed_idx])
                                        travel_logger.debug(f"Re-downloading corrupted segment {failed_idx} from {video_path_str_from_db}", task_id=stitch_task_id_str)

                                        redownload_start = time.time()

                                        # Delete corrupted file
                                        if failed_video_path.exists():
                                            failed_video_path.unlink()
                                            travel_logger.debug(f"Deleted corrupted file: {failed_video_path}", task_id=stitch_task_id_str)

                                        # Re-download
                                        download_file(video_path_str_from_db, stitch_processing_dir, failed_video_path.name)

                                        # Wait for stability
                                        if wait_for_file_stable(failed_video_path, checks=5, interval=1.0):
                                            travel_logger.debug(f"Re-downloaded segment {failed_idx} successfully", task_id=stitch_task_id_str)
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": True
                                            })
                                            redownload_attempted = True
                                        else:
                                            travel_logger.debug(f"Re-downloaded segment {failed_idx} not stable", task_id=stitch_task_id_str)
                                            redownload_duration = time.time() - redownload_start
                                            attempt_info["redownloads"].append({
                                                "segment_idx": failed_idx,
                                                "source_url": video_path_str_from_db,
                                                "duration_seconds": round(redownload_duration, 2),
                                                "success": False,
                                                "error": "File not stable after download"
                                            })

                                    except (OSError, ValueError, RuntimeError) as e_redownload:
                                        travel_logger.error(f"Re-download failed for segment {failed_idx}: {e_redownload}", task_id=stitch_task_id_str)
                                        redownload_duration = time.time() - redownload_start
                                        attempt_info["redownloads"].append({
                                            "segment_idx": failed_idx,
                                            "source_url": video_path_str_from_db,
                                            "duration_seconds": round(redownload_duration, 2),
                                            "success": False,
                                            "error": str(e_redownload)
                                        })
                                else:
                                    travel_logger.debug(f"Segment {failed_idx} is not a remote URL, cannot re-download: {video_path_str_from_db}", task_id=stitch_task_id_str)
                                    attempt_info["redownloads"].append({
                                        "segment_idx": failed_idx,
                                        "source_url": video_path_str_from_db,
                                        "duration_seconds": 0,
                                        "success": False,
                                        "error": "Not a remote URL - cannot re-download"
                                    })

                        time.sleep(wait_time)

                    # Log this attempt (whether successful or failed)
                    retry_log.append(attempt_info)
                else:
                    # All attempts failed - generate detailed error report
                    failed_segments = [i for i, f_list in enumerate(all_segment_frames_lists) if not (f_list is not None and len(f_list) > 0)]

                    # Build detailed error message
                    error_details = []
                    error_details.append(f"Frame extraction failed for segments {failed_segments} after {max_extraction_attempts} attempts")
                    error_details.append(f"Total segments in stitch: {len(segment_video_paths_for_stitch)}")

                    # Add per-segment analysis
                    for idx, video_path in enumerate(segment_video_paths_for_stitch):
                        video_path_obj = Path(video_path)
                        status = "SUCCESS" if idx not in failed_segments else "FAILED"

                        if video_path_obj.exists():
                            try:
                                file_size = video_path_obj.stat().st_size
                                # Try to get basic video info
                                cap = cv2.VideoCapture(str(video_path))
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
                                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else -1
                                cap.release()

                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, {frame_count} frames, {fps:.1f} fps)")
                            except (OSError, ValueError, RuntimeError):
                                file_size = video_path_obj.stat().st_size if video_path_obj.exists() else 0
                                error_details.append(f"  Segment {idx} [{status}]: {video_path_obj.name} ({file_size:,} bytes, properties unreadable)")
                        else:
                            error_details.append(f"  Segment {idx} [{status}]: {video_path} (FILE MISSING)")

                    # Add source information if available
                    if 'completed_segment_outputs_from_db' in locals():
                        error_details.append("Source URLs:")
                        for idx in failed_segments:
                            if idx < len(completed_segment_outputs_from_db):
                                seg_output = completed_segment_outputs_from_db[idx]
                                source_url = seg_output.get("video_file_path", "Unknown")
                                error_details.append(f"  Failed segment {idx} source: {source_url}")

                    # Add retry history
                    if retry_log:
                        error_details.append("Retry History:")
                        for log_entry in retry_log:
                            attempt_summary = f"  Attempt {log_entry['attempt']}: {log_entry['duration_seconds']}s, Success:{len(log_entry['successful_segments'])}, Failed:{len(log_entry['failed_segments'])}"
                            if log_entry['redownloads']:
                                redownload_summary = []
                                for rd in log_entry['redownloads']:
                                    rd_status = "OK" if rd['success'] else "FAIL"
                                    redownload_summary.append(f"Seg{rd['segment_idx']}({rd_status}{rd['duration_seconds']}s)")
                                attempt_summary += f", Redownloads:[{','.join(redownload_summary)}]"
                            error_details.append(attempt_summary)

                    # Before failing, try FFmpeg-based cross-fade as fallback
                    travel_logger.debug(f"Frame extraction failed completely. Attempting FFmpeg cross-fade fallback...", task_id=stitch_task_id_str)
                    try:
                        ffmpeg_result = attempt_ffmpeg_crossfade_fallback(
                            segment_video_paths_for_stitch,
                            actual_overlaps_for_stitching,
                            path_for_raw_stitched_video,
                            stitch_task_id_str,
                            lambda msg: travel_logger.debug(msg, task_id=stitch_task_id_str)
                        )
                        if ffmpeg_result:
                            travel_logger.essential(f"FFmpeg cross-fade fallback succeeded!", task_id=stitch_task_id_str)
                            current_stitched_video_path = path_for_raw_stitched_video
                        else:
                            detailed_error = "Stitch: Both frame extraction and FFmpeg cross-fade fallback failed. " + " | ".join(error_details)
                            raise ValueError(detailed_error)
                    except (OSError, ValueError, RuntimeError) as e_ffmpeg:
                        detailed_error = f"Stitch: Frame extraction failed and FFmpeg fallback also failed ({str(e_ffmpeg)}). " + " | ".join(error_details)
                        raise ValueError(detailed_error)

                final_stitched_frames = []

                # Process each stitch point
                for i in range(num_stitch_points):
                    frames_prev_segment = all_segment_frames_lists[i]
                    frames_curr_segment = all_segment_frames_lists[i+1]
                    current_overlap_val = actual_overlaps_for_stitching[i]

                    if i == 0:
                        # For the first stitch point, add frames from segment 0 up to the overlap
                        if current_overlap_val > 0:
                            # Add frames before the overlap region
                            frames_before_overlap = frames_prev_segment[:-current_overlap_val]
                            final_stitched_frames.extend(frames_before_overlap)
                        else:
                            # No overlap, add all frames from segment 0
                            final_stitched_frames.extend(frames_prev_segment)
                    else:
                        pass

                    if current_overlap_val > 0:
                        # Check if we should regenerate anchor frames (skip blending the anchor)
                        regenerate_anchors = orchestrator_details.get("regenerate_anchors", False)

                        if regenerate_anchors and current_overlap_val > 1:
                            # Regenerate anchor mode: crossfade all but the last frame (anchor)
                            # The anchor frame will be taken directly from current segment
                            crossfade_count = current_overlap_val - 1

                            # Remove the overlap frames (minus 1 for anchor) from accumulated
                            if i > 0:
                                frames_to_remove = min(crossfade_count, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]

                            # Blend the non-anchor overlapping frames
                            frames_prev_for_fade = frames_prev_segment[-crossfade_count:] if crossfade_count > 0 else []
                            frames_curr_for_fade = frames_curr_segment[:crossfade_count]
                            faded_frames = cross_fade_overlap_frames(frames_prev_for_fade, frames_curr_for_fade, crossfade_count, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)

                            # Add the regenerated anchor frame directly (no blend)
                            anchor_frame = frames_curr_segment[crossfade_count]
                            final_stitched_frames.append(anchor_frame)

                            # Adjust start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                        else:
                            # Normal crossfade mode: blend all overlap frames
                            # Remove the overlap frames already appended from the previous segment so that
                            # they can be replaced by the blended cross-fade frames for this stitch point.
                            if i > 0:
                                frames_to_remove = min(current_overlap_val, len(final_stitched_frames))
                                if frames_to_remove > 0:
                                    del final_stitched_frames[-frames_to_remove:]
                            # Blend the overlapping frames
                            faded_frames = cross_fade_overlap_frames(frames_prev_segment, frames_curr_segment, current_overlap_val, "linear_sharp", crossfade_sharp_amt)
                            final_stitched_frames.extend(faded_frames)

                            # Normal start index for remaining frames
                            start_index_for_curr_tail = current_overlap_val
                    else:
                        start_index_for_curr_tail = 0

                    # Add the non-overlapping part of the current segment
                    if len(frames_curr_segment) > start_index_for_curr_tail:
                        frames_to_add = frames_curr_segment[start_index_for_curr_tail:]
                        final_stitched_frames.extend(frames_to_add)

                if not final_stitched_frames: raise ValueError("Stitch: No frames produced after cross-fade logic.")

                # With proper cross-fade: output = sum(all frames) - sum(overlaps)
                # Because overlapped frames are blended, not duplicated
                total_input_frames = sum(len(frames) for frames in all_segment_frames_lists)
                total_overlaps = sum(actual_overlaps_for_stitching)
                expected_output_frames = total_input_frames - total_overlaps
                actual_output_frames = len(final_stitched_frames)
                travel_logger.debug(f"FINAL CROSS-FADE SUMMARY: Total input frames: {total_input_frames}, Total overlaps: {total_overlaps}, Expected output: {expected_output_frames}, Actual output: {actual_output_frames}, Match: {expected_output_frames == actual_output_frames}", task_id=stitch_task_id_str)

                current_stitched_video_path = create_video_from_frames_list(final_stitched_frames, path_for_raw_stitched_video, final_fps, parsed_res_wh)

                # Attempt to restore audio from segment videos (e.g. LTX-2 generates audio)
                audio_output = path_for_raw_stitched_video.with_name(
                    path_for_raw_stitched_video.stem + "_with_audio.mp4"
                )
                audio_result = mux_audio_from_segments(
                    silent_video=current_stitched_video_path,
                    segment_videos=segment_video_paths_for_stitch,
                    output_path=audio_output,
                    overlap_frames=actual_overlaps_for_stitching,
                    fps=final_fps,
                )
                if audio_result and audio_result.exists():
                    # Replace silent video with audio version
                    current_stitched_video_path.unlink(missing_ok=True)
                    audio_result.rename(current_stitched_video_path)
                    travel_logger.debug("Audio muxed from segment videos into stitched output", task_id=stitch_task_id_str)

            else:
                travel_logger.debug(f"Stitch: Using simple FFmpeg concatenation. Output to: {path_for_raw_stitched_video}", task_id=stitch_task_id_str)
                try:
                    from ...utils import stitch_videos_ffmpeg as stitch_videos_ffmpeg
                except ImportError:
                    travel_logger.error(f"Failed to import 'stitch_videos_ffmpeg'. Cannot proceed with stitching.", task_id=stitch_task_id_str)
                    raise

                if stitch_videos_ffmpeg(segment_video_paths_for_stitch, str(path_for_raw_stitched_video)):
                    current_stitched_video_path = path_for_raw_stitched_video
                else:
                    raise RuntimeError(f"Stitch: Simple FFmpeg concatenation failed for output {path_for_raw_stitched_video}.")

        if not current_stitched_video_path or not current_stitched_video_path.exists():
            raise RuntimeError(f"Stitch: Stitching process failed, output video not found at {current_stitched_video_path}")

        video_path_after_optional_upscale = current_stitched_video_path

        if isinstance(upscale_factor, (float, int)) and upscale_factor > 1.0 and upscale_model_name:
            travel_logger.debug(f"Starting upscale process: {upscale_factor}x using model {upscale_model_name}", task_id=stitch_task_id_str)

            original_frames_count, original_fps = get_video_frame_count_and_fps(str(current_stitched_video_path))
            if original_frames_count is None or original_frames_count == 0:
                raise ValueError(f"Stitch: Cannot get frame count or 0 frames for video {current_stitched_video_path} before upscaling.")

            travel_logger.debug(f"Upscale input video: {original_frames_count} frames @ {original_fps} FPS", task_id=stitch_task_id_str)
            travel_logger.debug(f"Upscale target resolution: {int(parsed_res_wh[0] * upscale_factor)}x{int(parsed_res_wh[1] * upscale_factor)}", task_id=stitch_task_id_str)

            target_width_upscaled = int(parsed_res_wh[0] * upscale_factor)
            target_height_upscaled = int(parsed_res_wh[1] * upscale_factor)

            upscale_sub_task_id = generate_unique_task_id(f"upscale_stitch_{orchestrator_run_id}_")

            upscale_payload = {
                "task_id": upscale_sub_task_id,
                "project_id": stitch_params.get("project_id"),
                "model": upscale_model_name,
                "video_source_path": str(current_stitched_video_path.resolve()),
                "resolution": f"{target_width_upscaled}x{target_height_upscaled}",
                "frames": original_frames_count,
                "prompt": orchestrator_details.get("original_task_args",{}).get("upscale_prompt", "cinematic, masterpiece, high detail, 4k"),
                "seed": orchestrator_details.get("seed_for_upscale", orchestrator_details.get("seed_base", DEFAULT_SEED_BASE) + UPSCALE_SEED_OFFSET),
            }

            upscaler_engine_to_use = stitch_params.get("execution_engine_for_upscale", "wgp")

            db_ops.add_task_to_db(
                task_payload=upscale_payload,
                task_type_str=upscaler_engine_to_use
            )
            travel_logger.debug(f"Enqueued upscale sub-task {upscale_sub_task_id} ({upscaler_engine_to_use}). Waiting...", task_id=stitch_task_id_str)

            poll_interval_ups = orchestrator_details.get("poll_interval", 15)
            poll_timeout_ups = orchestrator_details.get("poll_timeout_upscale", orchestrator_details.get("poll_timeout", 30 * 60) * 2)

            travel_logger.debug(f"Upscale polling for completion (timeout: {poll_timeout_ups}s, interval: {poll_interval_ups}s)", task_id=stitch_task_id_str)

            upscaled_video_db_location = db_ops.poll_task_status(
                task_id=upscale_sub_task_id,
                poll_interval_seconds=poll_interval_ups,
                timeout_seconds=poll_timeout_ups
            )
            travel_logger.debug(f"Upscale poll result: {upscaled_video_db_location}", task_id=stitch_task_id_str)

            if upscaled_video_db_location:
                # Path is already absolute (Supabase URL or absolute path)
                upscaled_video_abs_path: Path = Path(upscaled_video_db_location)

                if upscaled_video_abs_path.exists():
                    travel_logger.debug(f"Upscale completed successfully: {upscaled_video_abs_path}", task_id=stitch_task_id_str)

                    # Analyze upscaled result
                    try:
                        upscaled_frame_count, upscaled_fps = get_video_frame_count_and_fps(str(upscaled_video_abs_path))
                        travel_logger.debug(f"Upscaled result: {upscaled_frame_count} frames @ {upscaled_fps} FPS", task_id=stitch_task_id_str)

                        # Compare frame counts
                        if upscaled_frame_count != original_frames_count:
                            travel_logger.warning(f"Frame count changed during upscale: {original_frames_count} -> {upscaled_frame_count}", task_id=stitch_task_id_str)
                    except (OSError, ValueError, RuntimeError) as e_post_upscale:
                        travel_logger.warning(f"Could not analyze upscaled video: {e_post_upscale}", task_id=stitch_task_id_str)

                    video_path_after_optional_upscale = upscaled_video_abs_path

                    if not orchestrator_details.get("skip_cleanup_enabled", False) and \
                       not orchestrator_details.get("debug_mode_enabled", False) and \
                       current_stitched_video_path.exists() and current_stitched_video_path != video_path_after_optional_upscale:
                        try:
                            current_stitched_video_path.unlink()
                            travel_logger.debug(f"Stitch: Removed non-upscaled video {current_stitched_video_path} after successful upscale.", task_id=stitch_task_id_str)
                        except OSError as e_del_non_upscaled:
                            travel_logger.debug(f"Stitch: Warning - could not remove non-upscaled video {current_stitched_video_path}: {e_del_non_upscaled}", task_id=stitch_task_id_str)
                else:
                    travel_logger.error(f"Upscale output missing at {upscaled_video_abs_path}. Using non-upscaled video.", task_id=stitch_task_id_str)
            else:
                travel_logger.error(f"Upscale sub-task {upscale_sub_task_id} failed or timed out. Using non-upscaled video.", task_id=stitch_task_id_str)

        elif upscale_factor > 1.0 and not upscale_model_name:
            travel_logger.warning(f"Upscale factor {upscale_factor} > 1.0 but no upscale_model_name provided. Skipping upscale.", task_id=stitch_task_id_str)
        else:
            travel_logger.debug(f"No upscaling requested (factor: {upscale_factor})", task_id=stitch_task_id_str)

        if explicit_audio_url:
            try:
                audio_output_path = video_path_after_optional_upscale.with_name(
                    f"{video_path_after_optional_upscale.stem}_with_audio{video_path_after_optional_upscale.suffix}"
                )
                muxed_audio_path = add_audio_to_video(
                    input_video_path=str(video_path_after_optional_upscale),
                    audio_url=explicit_audio_url,
                    output_video_path=str(audio_output_path),
                    temp_dir=str(stitch_processing_dir),
                )
                if muxed_audio_path and Path(muxed_audio_path).exists():
                    video_path_after_optional_upscale = Path(muxed_audio_path)
                    travel_logger.debug("External audio muxed into stitched output", task_id=stitch_task_id_str)
            except (OSError, ValueError, RuntimeError) as audio_err:
                travel_logger.warning(
                    f"Stitch: Failed to add external audio (continuing without): {audio_err}",
                    task_id=stitch_task_id_str,
                )

        # Use consistent UUID-based naming for final video
        timestamp_short = datetime.now().strftime("%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        if upscale_factor > 1.0:
            final_video_filename = f"{stitch_task_id_str}_upscaled_{upscale_factor:.1f}x_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"
        else:
            final_video_filename = f"{stitch_task_id_str}_output_{timestamp_short}_{unique_suffix}{video_path_after_optional_upscale.suffix}"

        final_video_path, initial_db_location = prepare_output_path_with_upload(
            task_id=stitch_task_id_str,
            filename=final_video_filename,
            main_output_dir_base=stitch_processing_dir,
            task_type="travel_stitch")

        # Move the video to final location if it's not already there
        if video_path_after_optional_upscale.resolve() != final_video_path.resolve():
            travel_logger.debug(f"Stitch Task {stitch_task_id_str}: Moving {video_path_after_optional_upscale} to {final_video_path}", task_id=stitch_task_id_str)
            shutil.move(str(video_path_after_optional_upscale), str(final_video_path))
        else:
            travel_logger.debug(f"Stitch Task {stitch_task_id_str}: Video already at final destination {final_video_path}", task_id=stitch_task_id_str)

        # Handle Supabase upload (if configured) and get final location for DB
        final_video_location_for_db = upload_and_get_final_output_location(
            final_video_path,
            initial_db_location)

        travel_logger.info(f"Stitch complete: Final video saved to {final_video_path}", task_id=stitch_task_id_str)

        # Analyze final result
        try:
            final_frame_count, final_fps = get_video_frame_count_and_fps(str(final_video_path))
            final_duration = final_frame_count / final_fps if final_fps > 0 else 0
            travel_logger.debug(f"Final video: {final_frame_count} frames @ {final_fps} FPS = {final_duration:.2f}s", task_id=stitch_task_id_str)
            travel_logger.debug(
                f"[STITCH_DEBUG] final_analysis input_segments={len(segment_video_paths_for_stitch)}, "
                f"overlap_count={len(expanded_frame_overlaps)}, overlap_sample={expanded_frame_overlaps[:5]}",
                task_id=stitch_task_id_str,
            )
            # Calculate expected final length for analysis
            try:
                # Ground-truth expected length: compute from the actual decoded segment frame counts.
                # This avoids misleading results when orchestrator payload contains expanded/per-frame arrays.
                actual_segment_counts = []
                for p in segment_video_paths_for_stitch:
                    try:
                        fc, _fps = get_video_frame_count_and_fps(str(p))
                        if fc is None:
                            continue
                        actual_segment_counts.append(int(fc))
                    except (OSError, ValueError, RuntimeError):
                        continue

                if actual_segment_counts:
                    total_input_frames = sum(actual_segment_counts)
                    total_overlaps = sum(expanded_frame_overlaps) if expanded_frame_overlaps else 0
                    expected_final_length = total_input_frames - total_overlaps
                    travel_logger.debug(f"Expected final frames (from actual segments): {expected_final_length}, Actual final frames: {final_frame_count}", task_id=stitch_task_id_str)
                    if final_frame_count != expected_final_length:
                        travel_logger.warning(f"FINAL LENGTH MISMATCH! Expected {expected_final_length}, got {final_frame_count}", task_id=stitch_task_id_str)
                else:
                    travel_logger.debug(f"Expected final frames: Not available (could not count input segments), Actual final frames: {final_frame_count}", task_id=stitch_task_id_str)
            except (OSError, ValueError, RuntimeError, TypeError) as e:
                travel_logger.debug(f"Expected final frames: Not calculated ({e}), Actual final frames: {final_frame_count}", task_id=stitch_task_id_str)

            # Detailed analysis of the final video
            debug_video_analysis(final_video_path, "FINAL_STITCHED_VIDEO", stitch_task_id_str)

        except (OSError, ValueError, RuntimeError) as e_final_analysis:
            travel_logger.warning(f"Could not analyze final video: {e_final_analysis}", task_id=stitch_task_id_str)

        # Note: Individual segments already have banner overlays applied when show_input_images is enabled,
        # so the stitched video will automatically include them. No additional overlay needed here.

        stitch_success = True

        # --- Cleanup Downloaded Segment Files ---
        cleanup_enabled = (
            not orchestrator_details.get("skip_cleanup_enabled", False) and
            not orchestrator_details.get("debug_mode_enabled", False) and
            not db_config.debug_mode
        )

        if cleanup_enabled:
            files_cleaned = 0
            total_size_cleaned = 0

            for video_path_str in segment_video_paths_for_stitch:
                video_path = Path(video_path_str)

                # Skip the initial continued video (not downloaded)
                if (initial_continued_video_path_str and
                    str(video_path.resolve()) == str(Path(initial_continued_video_path_str).resolve())):
                    continue

                # Only delete files in our processing directory (downloaded files)
                if video_path.exists() and stitch_processing_dir in video_path.parents:
                    try:
                        file_size = video_path.stat().st_size
                        video_path.unlink()
                        files_cleaned += 1
                        total_size_cleaned += file_size
                        travel_logger.debug(f"Stitch: Cleaned up downloaded segment {video_path.name} ({file_size:,} bytes)", task_id=stitch_task_id_str)
                    except OSError as e_cleanup:
                        travel_logger.debug(f"Stitch: Failed to clean up {video_path}: {e_cleanup}", task_id=stitch_task_id_str)

            if files_cleaned > 0:
                travel_logger.debug(f"Removed {files_cleaned} downloaded files ({total_size_cleaned:,} bytes)", task_id=stitch_task_id_str)
        else:
            travel_logger.debug(f"Skipping cleanup (debug mode or cleanup disabled)", task_id=stitch_task_id_str)

        # Note: Orchestrator will be marked complete by worker.py after stitch upload completes
        # This ensures the orchestrator gets the final storage URL, not a local path

        # Return the final video path so the stitch task itself gets uploaded via Edge Function
        log_ram_usage("Stitch end (success)", task_id=stitch_task_id_str)
        return stitch_success, str(final_video_path.resolve())

    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e:
        travel_logger.error(f"Stitch: Unexpected error during stitching: {e}", task_id=stitch_task_id_str)
        travel_logger.debug(traceback.format_exc(), task_id=stitch_task_id_str)

        # Notify orchestrator of stitch failure
        if 'orchestrator_task_id_ref' in locals() and orchestrator_task_id_ref:
            try:
                error_msg = f"Stitch task failed: {str(e)[:200]}"
                db_ops.update_task_status(
                    orchestrator_task_id_ref,
                    db_ops.STATUS_FAILED,
                    error_msg
                )
                travel_logger.debug(f"Stitch: Marked orchestrator task {orchestrator_task_id_ref} as FAILED due to exception", task_id=stitch_task_id_str)
            except (RuntimeError, ValueError, OSError) as e_orch:
                travel_logger.debug(f"Stitch: Warning - could not update orchestrator status: {e_orch}", task_id=stitch_task_id_str)

        log_ram_usage("Stitch end (error)", task_id=stitch_task_id_str)
        return False, f"Stitch task failed: {str(e)[:200]}"
    finally:
        flush_ram_snapshots(stitch_task_id_str)


# Public alias for cross-module use.
handle_travel_stitch_task = _handle_travel_stitch_task
