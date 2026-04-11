"""
Edit Video Orchestrator - Regenerate selected portions of a video

Takes a source video and portions_to_regenerate, extracts the "keeper" 
segments (portions NOT being regenerated), then uses the join_clips 
infrastructure to regenerate transitions between keeper clips.

Example:
    Source video: 164 frames total (0-163)
    portions_to_regenerate: [
        {start_frame: 16, end_frame: 33},   # 18 frames to regenerate
        {start_frame: 49, end_frame: 66}    # 18 frames to regenerate
    ]

    Resulting keeper clips:
    - Clip 0: frames 0-15   (16 frames)
    - Clip 1: frames 34-48  (15 frames)  
    - Clip 2: frames 67-163 (97 frames)

    Join tasks created:
    - join_0: Clip 0 + Clip 1 → regenerates transition (was frames 16-33)
    - join_1: (Clip 0+1 result) + Clip 2 → regenerates transition (was frames 49-66)

The extracted keeper clips are passed to the shared _create_join_chain_tasks() 
function which creates the same join_clips_segment tasks as join_clips_orchestrator.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict

from ..utils import download_video_if_url, get_video_frame_count_and_fps
from ..media.video import extract_frame_range_to_video, ensure_video_fps, get_video_frame_count_ffprobe, get_video_fps_ffprobe

# Import shared functions from join subpackage
from source.task_handlers.join.shared import (
    extract_join_settings_from_payload,
    check_existing_join_tasks)
from source.task_handlers.join.task_builder import create_join_chain_tasks
from source.task_handlers.join.vlm_enhancement import (
    extract_boundary_frames_for_vlm,
    generate_vlm_prompts_for_joins)
from source.core.log import task_logger

def _calculate_keeper_segments(
    portions: List[dict],
    total_frames: int,
    replace_mode: bool = False
) -> List[dict]:
    """
    Calculate keeper segments (parts NOT being regenerated).
    
    Args:
        portions: List of portions to regenerate with start_frame, end_frame
        total_frames: Total frames in source video
        replace_mode: If True, the start and end frames of each portion become
                      ANCHORS that are preserved, and only content BETWEEN them
                      is regenerated. If False, the entire portion is regenerated.
        
    Returns:
        List of {start_frame, end_frame, frame_count} dicts for keeper segments
    """
    # Sort portions by start_frame
    sorted_portions = sorted(portions, key=lambda p: p["start_frame"])
    
    mode_desc = "REPLACE MODE (anchors preserved)" if replace_mode else "STANDARD MODE (full replacement)"
    task_logger.debug_block(
        "SETUP",
        {
            "mode": mode_desc,
            "total_frames": total_frames,
            "portion_count": len(sorted_portions),
            "portions": [
                {
                    "start": p["start_frame"],
                    "end": p["end_frame"],
                    "frame_count": p.get("frame_count") or (p["end_frame"] - p["start_frame"] + 1),
                }
                for p in sorted_portions
            ],
        },
    )
    
    # Validate portions don't overlap
    for i in range(len(sorted_portions) - 1):
        if sorted_portions[i]["end_frame"] >= sorted_portions[i + 1]["start_frame"]:
            raise ValueError(
                f"Portions overlap: portion {i} ends at {sorted_portions[i]['end_frame']}, "
                f"portion {i+1} starts at {sorted_portions[i + 1]['start_frame']}"
            )
    
    keepers = []
    
    # In replace_mode:
    #   - start_frame becomes the END of the previous keeper (anchor preserved)
    #   - end_frame becomes the START of the next keeper (anchor preserved)
    #   - Only frames BETWEEN anchors are regenerated
    #
    # Example: portion start_frame=417, end_frame=427 in replace_mode:
    #   - Keeper 0: frames 0-417 (includes anchor frame 417)
    #   - Regenerate: frames 418-426 (between anchors)
    #   - Keeper 1: frames 427-end (includes anchor frame 427)
    
    # Keeper before first portion
    first_start = sorted_portions[0]["start_frame"]
    if replace_mode:
        # In replace mode, include start_frame as anchor in keeper
        keeper_end = first_start  # Include the anchor
    else:
        # Standard mode: keeper ends before portion starts
        keeper_end = first_start - 1
    
    if keeper_end >= 0:
        keeper = {
            "start_frame": 0,
            "end_frame": keeper_end,
            "frame_count": keeper_end + 1
        }
        keepers.append(keeper)
    
    # Keepers between portions
    for i in range(len(sorted_portions) - 1):
        if replace_mode:
            # Include end_frame of portion i as start (anchor)
            # Include start_frame of portion i+1 as end (anchor)
            gap_start = sorted_portions[i]["end_frame"]  # Anchor from portion i
            gap_end = sorted_portions[i + 1]["start_frame"]  # Anchor from portion i+1
        else:
            gap_start = sorted_portions[i]["end_frame"] + 1
            gap_end = sorted_portions[i + 1]["start_frame"] - 1
        
        if gap_end >= gap_start:
            keeper = {
                "start_frame": gap_start,
                "end_frame": gap_end,
                "frame_count": gap_end - gap_start + 1
            }
            keepers.append(keeper)
    
    # Keeper after last portion
    last_end = sorted_portions[-1]["end_frame"]
    if replace_mode:
        # In replace mode, include end_frame as anchor in keeper
        keeper_start = last_end  # Include the anchor
    else:
        keeper_start = last_end + 1
    
    if keeper_start < total_frames:
        keeper = {
            "start_frame": keeper_start,
            "end_frame": total_frames - 1,
            "frame_count": total_frames - keeper_start
        }
        keepers.append(keeper)
    task_logger.debug_block(
        "SETUP",
        {
            "keeper_count": len(keepers),
            "keepers": keepers,
        },
    )
    
    return keepers

def _get_clip_url_or_path(
    local_path: Path,
    project_id: str | None,
    task_id: str,
    clip_name: str) -> str:
    """
    Get the URL or path for an extracted clip.
    
    For now, returns the local file path. The join_clips_segment tasks
    will handle downloading if the path is a URL, and the final output
    will be uploaded by the edge function when the task completes.
    
    Note: If you need to upload intermediate clips to Supabase storage
    for distributed workers, you can extend this function to use the
    Supabase storage upload utilities.
    
    Returns:
        Local file path as string (or URL if upload is implemented)
    """
    # For local/distributed processing, the local path works since workers
    # share filesystem or can download URLs. The join_clips_segment handler
    # already supports both paths and URLs via download_video_if_url().
    
    # NOTE: For fully distributed deployments without shared storage,
    # this function should upload to Supabase storage and return the URL.
    # Currently not needed because workers share filesystem.
    
    return str(local_path)

def _preprocess_portions_to_regenerate(
    source_video_url: str,
    source_video_fps: float,
    source_video_total_frames: int,
    portions_to_regenerate: List[dict],
    per_join_settings: List[dict],
    work_dir: Path,
    orchestrator_task_id: str,
    orchestrator_project_id: str | None,
    replace_mode: bool = False
) -> Dict:
    """
    Pre-process portions_to_regenerate into keeper clips for join orchestration.

    Args:
        source_video_url: URL or path to source video
        source_video_fps: FPS of source video
        source_video_total_frames: Total frame count of source video
        portions_to_regenerate: List of portions to regenerate
        per_join_settings: Per-join settings from orchestrator payload
        work_dir: Working directory for extracted clips
        orchestrator_task_id: Task ID for logging
        orchestrator_project_id: Project ID for Supabase uploads
        replace_mode: If True, keep start/end frames as anchors, regenerate between

    Returns:
        {
            "success": bool,
            "error": str | None,
            "clip_list": List[dict],
            "per_join_settings": List[dict]
        }
    """
    try:
        # Download source video if it's a URL
        source_video_path = download_video_if_url(
            source_video_url,
            download_target_dir=work_dir,
            task_id_for_logging=orchestrator_task_id,
            descriptive_name="source_video"
        )
        source_video = Path(source_video_path)
        
        if not source_video.exists():
            return {"success": False, "error": f"Source video not found: {source_video_path}"}
        
        target_fps = source_video_fps or 16
        ffprobe_fps_before = get_video_fps_ffprobe(str(source_video))
        task_logger.debug_block(
            "SETUP",
            {
                "stage": "source_baseline",
                "payload_frames": source_video_total_frames,
                "target_fps": target_fps,
                "detected_fps": ffprobe_fps_before,
            },
            task_id=orchestrator_task_id,
        )

        # Time↔frame sanity checks for each portion (if times are provided)
        sorted_portions_dbg = sorted(portions_to_regenerate, key=lambda p: p.get("start_frame", 0))
        for i, p in enumerate(sorted_portions_dbg):
            sf = p.get("start_frame")
            ef = p.get("end_frame")
            st = p.get("start_time_seconds") or p.get("start_time")
            et = p.get("end_time_seconds") or p.get("end_time")
            if sf is None or ef is None:
                continue
            if st is not None and et is not None:
                try:
                    # These conversions assume 0-indexed frames where frame ~= time * fps
                    # We log both payload-fps and ffprobe-fps estimates to spot drift.
                    sf_payload = int(round(float(st) * float(target_fps)))
                    ef_payload = int(round(float(et) * float(target_fps))) - 1  # end_time is typically exclusive
                    sf_ff = int(round(float(st) * float(ffprobe_fps_before))) if ffprobe_fps_before else None
                    ef_ff = int(round(float(et) * float(ffprobe_fps_before))) - 1 if ffprobe_fps_before else None
                    task_logger.debug_block(
                        "SETUP",
                        {
                            "stage": f"portion_{i}",
                            "frames": (sf, ef),
                            "times": (st, et),
                            "payload_fps_frames": (sf_payload, ef_payload),
                            "ffprobe_fps_frames": None if ffprobe_fps_before is None else (sf_ff, ef_ff),
                        },
                        task_id=orchestrator_task_id,
                    )
                except (ValueError, KeyError, TypeError) as conv_err:
                    task_logger.debug_anomaly("EDIT_VIDEO", f"Portion {i}: time↔frame conversion error: {conv_err}")
        
        # Use tight FPS tolerance (0.01) to force resampling when there's any meaningful FPS difference
        # This is critical because frame indices are calculated for target_fps - even a 0.3 FPS difference
        # causes ~0.5s drift over 400 frames, making extracted frames appear "too early" or "too late"
        source_video_before_ensure = source_video
        try:
            source_video = ensure_video_fps(
                input_video_path=source_video,
                target_fps=target_fps,
                output_dir=work_dir,
                fps_tolerance=0.01,  # Tight tolerance to ensure frame-accurate extraction
            )
        except (OSError, ValueError, RuntimeError) as e:
            return {"success": False, "error": f"Failed to ensure video is at {target_fps} fps: {e}"}

        source_resampled = Path(source_video_before_ensure) != Path(source_video)
        
        # Get actual frame count from the (potentially resampled) video
        # Use ffprobe for accuracy - OpenCV's CAP_PROP_FRAME_COUNT is unreliable
        actual_frames_ffprobe = get_video_frame_count_ffprobe(str(source_video))
        actual_frames_opencv, actual_fps = get_video_frame_count_and_fps(str(source_video))
        actual_fps_ffprobe = get_video_fps_ffprobe(str(source_video))
        task_logger.debug_block(
            "SETUP",
            {
                "stage": "source_post_ensure_fps",
                "resampled": source_resampled,
                "ffprobe_frames": actual_frames_ffprobe,
                "ffprobe_fps": actual_fps_ffprobe,
                "opencv_frames": actual_frames_opencv,
                "opencv_fps": actual_fps,
            },
            task_id=orchestrator_task_id,
        )
        
        # Prefer ffprobe count, fall back to OpenCV
        actual_frames = actual_frames_ffprobe or actual_frames_opencv
        if not actual_frames:
            return {"success": False, "error": "Could not determine source video frame count"}
        
        # Log discrepancies for debugging
        if actual_frames_ffprobe and actual_frames_opencv and actual_frames_ffprobe != actual_frames_opencv:
            task_logger.debug_anomaly("EDIT_VIDEO", f"Frame count: ffprobe={actual_frames_ffprobe}, OpenCV={actual_frames_opencv} (using ffprobe)")
        
        # Trust payload's frame count if close to actual - it was calculated for this video
        # Only use actual if significantly different (e.g., after resampling changed frame count)
        if source_video_total_frames:
            diff = abs(source_video_total_frames - actual_frames)
            if diff <= 3:
                # Small difference - trust payload (frontend calculated correctly)
                task_logger.debug_anomaly("EDIT_VIDEO", f"Using payload frame count: {source_video_total_frames} (actual: {actual_frames}, diff: {diff})")
                actual_frames = source_video_total_frames
            else:
                task_logger.debug_anomaly("EDIT_VIDEO", f"Payload frame count differs significantly: payload={source_video_total_frames}, actual={actual_frames}, diff={diff}")
        
        source_video_total_frames = actual_frames
        source_video_fps = actual_fps or target_fps
        
        task_logger.debug_block(
            "SETUP",
            {
                "stage": "source_selected",
                "frames": source_video_total_frames,
                "fps": source_video_fps,
                "target_fps": target_fps,
            },
            task_id=orchestrator_task_id,
        )
        
        # Calculate keeper segments
        keeper_segments = _calculate_keeper_segments(
            portions=portions_to_regenerate,
            total_frames=source_video_total_frames,
            replace_mode=replace_mode
        )
        
        if len(keeper_segments) < 2:
            return {"success": False, "error": f"Need at least 2 keeper segments, got {len(keeper_segments)}"}
        
        # Create keeper clips directory
        keepers_dir = work_dir / "keeper_clips"
        keepers_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract each keeper clip
        extracted_clips = []
        for i, keeper in enumerate(keeper_segments):
            clip_name = f"keeper_{i}.mp4"
            local_path = keepers_dir / clip_name
            
            try:
                extract_frame_range_to_video(
                    input_video_path=source_video,
                    output_video_path=local_path,
                    start_frame=keeper["start_frame"],
                    end_frame=keeper["end_frame"],
                    fps=source_video_fps)
            except (OSError, ValueError, RuntimeError) as e:
                return {"success": False, "error": f"Failed to extract keeper clip {i}: {e}"}
            
            # Get URL or path for the clip
            clip_url = _get_clip_url_or_path(
                local_path=local_path,
                project_id=orchestrator_project_id,
                task_id=orchestrator_task_id,
                clip_name=clip_name)
            
            extracted_clips.append({
                "url": clip_url,
                "name": f"keeper_{i}",
                "start_frame": keeper["start_frame"],
                "end_frame": keeper["end_frame"],
                "frame_count": keeper["frame_count"]
            })
        
        task_logger.debug_block(
            "SETUP",
            {
                "stage": "keepers_extracted",
                "clip_count": len(extracted_clips),
                "clips": extracted_clips,
            },
            task_id=orchestrator_task_id,
        )
        
        # Build per_join_settings from portions
        # Each join corresponds to a transition between keeper clips
        # 
        # gap_frame_count priority (highest to lowest):
        #   1. portion.gap_frame_count (per-portion override)
        #   2. per_join_settings[i].gap_frame_count (from orchestrator payload)
        #   3. orchestrator_payload.gap_frame_count (default for all joins)
        #
        sorted_portions = sorted(portions_to_regenerate, key=lambda p: p["start_frame"])
        new_per_join_settings = []
        
        for i, portion in enumerate(sorted_portions):
            # Start with existing per_join_settings if provided
            if i < len(per_join_settings):
                join_setting = per_join_settings[i].copy()
            else:
                join_setting = {}
            
            # Per-portion prompt override
            if "prompt" in portion:
                join_setting["prompt"] = portion["prompt"]
            
            # Per-portion gap_frame_count override
            # This allows setting different gaps for each transition
            if "gap_frame_count" in portion:
                join_setting["gap_frame_count"] = portion["gap_frame_count"]
            
            new_per_join_settings.append(join_setting)
            
            portion_frame_count = portion.get("frame_count") or (portion["end_frame"] - portion["start_frame"] + 1)
            gap_override = join_setting.get("gap_frame_count")
            gap_info = f", gap={gap_override}" if gap_override else ""
            task_logger.debug_block(
                "SETUP",
                {
                    "stage": f"join_{i}",
                    "portion_frames": portion_frame_count,
                    "portion_range": (portion["start_frame"], portion["end_frame"]),
                    "gap_override": gap_override,
                },
                task_id=orchestrator_task_id,
            )
        
        return {
            "success": True,
            "error": None,
            "clip_list": extracted_clips,
            "per_join_settings": new_per_join_settings
        }
        
    except (OSError, ValueError, RuntimeError) as e:
        task_logger.error(f"[EDIT_VIDEO] Preprocess portions failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def _handle_edit_video_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None) -> Tuple[bool, str]:
    """
    Handle edit_video_orchestrator task - regenerate selected portions of a video.
    
    This orchestrator:
    1. Takes a source video and portions_to_regenerate
    2. Extracts "keeper" clips (portions NOT being regenerated)
    3. Creates join_clips_segment tasks to regenerate transitions between keepers
    
    Args:
        task_params_from_db: Task parameters containing orchestrator_details with:
            - source_video_url: URL or path to source video
            - source_video_fps: FPS of source video (default 16)
            - source_video_total_frames: Total frames in source video
            - portions_to_regenerate: List of {start_frame, end_frame, prompt?} dicts
            - All standard join_clips settings (context_frame_count, model, etc.)
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization

    Returns:
        (success: bool, message: str)
    """
    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            task_logger.debug_anomaly("EDIT_VIDEO", "ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        
        # Extract required fields for edit_video
        source_video_url = orchestrator_payload.get("source_video_url")
        portions_to_regenerate = orchestrator_payload.get("portions_to_regenerate", [])
        run_id = orchestrator_payload.get("run_id")
        
        if not source_video_url:
            return False, "source_video_url is required for edit_video_orchestrator"
        
        if not portions_to_regenerate or len(portions_to_regenerate) < 1:
            return False, "portions_to_regenerate must contain at least 1 portion"
        
        if not run_id:
            return False, "run_id is required"
        
        task_logger.debug_block(
            "SETUP",
            {
                "task": orchestrator_task_id_str,
                "run_id": run_id,
                "source_video": source_video_url,
                "portion_count": len(portions_to_regenerate),
                "payload_keys": sorted(orchestrator_payload.keys()),
            },
            task_id=orchestrator_task_id_str,
        )
        
        # === EARLY IDEMPOTENCY CHECK (before expensive preprocessing/VLM work) ===
        # Each portion to regenerate = one join task
        num_joins_expected = len(portions_to_regenerate)
        idempotency_check = check_existing_join_tasks(orchestrator_task_id_str, num_joins_expected)
        if idempotency_check is not None:
            return idempotency_check
        
        # Extract join settings (same as join_clips_orchestrator)
        join_settings = extract_join_settings_from_payload(orchestrator_payload)
        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        
        # IMPORTANT: Override replace_mode to False for join tasks.
        # edit_video_orchestrator uses replace_mode to determine how keepers are extracted
        # (anchors at boundaries, gap region excluded). The keepers are already "clean" -
        # they don't contain gap frames. But join_clips in replace_mode expects clips to
        # CONTAIN gap frames at their boundaries that it will remove. This mismatch causes:
        # 1. Wrong context frame extraction (looking before gap position that doesn't exist)
        # 2. Double-trimming of frames (gap already excluded from keepers, then trimmed again)
        # The join tasks should use INSERT mode since they're bridging clean keeper clips.
        join_settings["replace_mode"] = False
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))
        
        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"edit_video_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # === 2. PREPROCESS: Extract keeper clips ===
        replace_mode = orchestrator_payload.get("replace_mode", False)
        preprocess_result = _preprocess_portions_to_regenerate(
            source_video_url=source_video_url,
            source_video_fps=orchestrator_payload.get("source_video_fps", 16),
            source_video_total_frames=orchestrator_payload.get("source_video_total_frames"),
            portions_to_regenerate=portions_to_regenerate,
            per_join_settings=per_join_settings,
            work_dir=current_run_output_dir,
            orchestrator_task_id=orchestrator_task_id_str,
            orchestrator_project_id=orchestrator_project_id,
            replace_mode=replace_mode
        )
        
        if not preprocess_result["success"]:
            error_msg = preprocess_result.get("error", "Unknown preprocessing error")
            task_logger.debug_anomaly("EDIT_VIDEO", f"Preprocessing failed: {error_msg}")
            return False, f"Preprocessing failed: {error_msg}"
        
        clip_list = preprocess_result["clip_list"]
        per_join_settings = preprocess_result["per_join_settings"]
        
        num_joins = len(clip_list) - 1
        task_logger.debug_block(
            "SETUP",
            {
                "stage": "preprocessed",
                "keeper_clips": len(clip_list),
                "join_count": num_joins,
                "per_join_settings": len(per_join_settings),
            },
            task_id=orchestrator_task_id_str,
        )
        
        # === 3. VLM PROMPT ENHANCEMENT (optional) ===
        enhance_prompt = orchestrator_payload.get("enhance_prompt", False)
        vlm_enhanced_prompts: List[Optional[str]] = [None] * num_joins
        
        if enhance_prompt:
            vlm_device = orchestrator_payload.get("vlm_device", "cuda")
            vlm_temp_dir = current_run_output_dir / "vlm_temp"
            vlm_temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                base_prompt = join_settings.get("prompt", "")
                gap_frame_count = join_settings.get("gap_frame_count", 53)
                _fps = orchestrator_payload.get("source_video_fps", 16)
                
                # Use replace_mode=False for VLM frame extraction because keeper clips are
                # already clean - anchors are at the boundaries (last/first frames), not
                # hidden behind gap frames that need to be skipped.
                image_pairs = extract_boundary_frames_for_vlm(
                    clip_list=clip_list,
                    temp_dir=vlm_temp_dir,
                    orchestrator_task_id=orchestrator_task_id_str,
                    replace_mode=False,  # Keepers have anchors at boundaries
                    gap_frame_count=gap_frame_count)
                
                vlm_enhanced_prompts = generate_vlm_prompts_for_joins(
                    image_quads=image_pairs,
                    base_prompt=base_prompt,
                    vlm_device=vlm_device)
                
            except (RuntimeError, ValueError, OSError) as vlm_error:
                task_logger.warning(f"EDIT_VIDEO VLM enhancement failed, using base prompts: {vlm_error}", exc_info=True)
                vlm_enhanced_prompts = [None] * num_joins
        
        # === CREATE JOIN CHAIN (using shared core function) ===
        parent_generation_id = (
            task_params_from_db.get("parent_generation_id")
            or orchestrator_payload.get("parent_generation_id")
        )
        success, message = create_join_chain_tasks(
            clip_list=clip_list,
            run_id=run_id,
            join_settings=join_settings,
            per_join_settings=per_join_settings,
            vlm_enhanced_prompts=vlm_enhanced_prompts,
            current_run_output_dir=current_run_output_dir,
            orchestrator_task_id_str=orchestrator_task_id_str,
            orchestrator_project_id=orchestrator_project_id,
            orchestrator_payload=orchestrator_payload,
            parent_generation_id=parent_generation_id)
        
        return success, message
        
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        msg = f"Failed during edit_video orchestration: {e}"
        task_logger.error(f"EDIT_VIDEO error: {msg}", exc_info=True)
        return False, msg


# Public alias for cross-module use.
handle_edit_video_orchestrator_task = _handle_edit_video_orchestrator_task
