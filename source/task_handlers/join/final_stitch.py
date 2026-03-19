"""
Join final stitch handler - stitch all clips and transitions together in one pass.

This is the second phase of the parallel join architecture:
1. Multiple join_clips_segment tasks generate transitions in parallel (transition_only=True)
2. This task stitches all original clips + transitions together in a single encode pass
"""

import json
import shutil
from pathlib import Path
from typing import Tuple

# Import shared utilities
from ...utils import (
    get_video_frame_count_and_fps,
    download_video_if_url,
    prepare_output_path_with_upload,
    upload_and_get_final_output_location)
from ...media.video import (
    ensure_video_fps,
    extract_frames_from_video,
    extract_frame_range_to_video,
    stitch_videos_with_crossfade,
    add_audio_to_video)
from ... import db_operations as db_ops
from source.core.log import orchestrator_logger


def _materialize_chain_passthrough_output(
    *,
    chain_output: str,
    task_id: str,
    main_output_dir_base: Path,
) -> str:
    """Create a task-scoped local artifact for chain passthrough completion."""
    task_output_path, _ = prepare_output_path_with_upload(
        task_id=task_id,
        filename=f"{task_id}_joined.mp4",
        main_output_dir_base=main_output_dir_base,
        task_type="join_final_stitch",
    )
    task_output_path.parent.mkdir(parents=True, exist_ok=True)

    previous_output_path = Path(chain_output)
    if previous_output_path.exists():
        if previous_output_path.resolve() != task_output_path.resolve():
            shutil.copy2(previous_output_path, task_output_path)
        if task_output_path.stat().st_size == 0:
            raise ValueError(
                f"Chain passthrough local output is empty for task {task_id}: {task_output_path}"
            )
        return str(task_output_path)

    downloaded_output_path = download_video_if_url(
        chain_output,
        download_target_dir=task_output_path.parent,
        task_id_for_logging=task_id,
        descriptive_name="chain_passthrough",
    )
    if not downloaded_output_path:
        raise ValueError(
            f"Failed to download chain passthrough output for task {task_id}: {chain_output}"
        )

    downloaded_path = Path(downloaded_output_path)
    if downloaded_path.resolve() != task_output_path.resolve():
        if task_output_path.exists():
            task_output_path.unlink()
        shutil.move(str(downloaded_path), str(task_output_path))

    if not task_output_path.exists() or task_output_path.stat().st_size == 0:
        raise ValueError(
            f"Chain passthrough output was not materialized for task {task_id}: {task_output_path}"
        )

    return str(task_output_path)


def handle_join_final_stitch(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str) -> Tuple[bool, str]:
    """
    Handle join_final_stitch task: stitch all clips and transitions together in one pass.

    This is the second phase of the parallel join architecture:
    1. Multiple join_clips_segment tasks generate transitions in parallel (transition_only=True)
    2. This task stitches all original clips + transitions together in a single encode pass

    Args:
        task_params_from_db: Task parameters including:
            - clip_list: List of original clip dicts with 'url' and optional 'name'
            - transition_task_ids: List of transition task IDs to fetch outputs from
            - gap_from_clip1: Frames to trim from end of each clip (except last)
            - gap_from_clip2: Frames to trim from start of each clip (except first)
            - blend_frames: Frames to crossfade at each boundary
            - fps: Output FPS
            - audio_url: Optional audio to add to final output
        main_output_dir_base: Base output directory
        task_id: Task ID for logging

    Returns:
        Tuple of (success: bool, output_path_or_message: str)
    """
    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Starting final stitch handler")

    try:
        # --- Chain Mode Passthrough ---
        # When created by the chain pattern, the last join in the chain already produced
        # the fully concatenated video. We just pass through its output (+ optional audio).
        chain_mode = task_params_from_db.get("chain_mode", False)
        if chain_mode:
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Chain mode \u2014 passthrough from last chain join")
            transition_task_ids = task_params_from_db.get("transition_task_ids", [])
            if not transition_task_ids:
                return False, "chain_mode=True but no transition_task_ids (last chain join ID) provided"

            # The single ID is the last join in the chain
            last_join_id = transition_task_ids[0]
            chain_output = db_ops.get_task_output_location_from_db(last_join_id)
            if not chain_output:
                return False, f"Failed to get output from chain join task {last_join_id}"

            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Chain output from {last_join_id}: {chain_output[:100]}...")

            # The chain join's output_location is the final video URL/path — use it directly.
            # In chain mode, the last join_clips_child already muxes audio when is_last_join=True,
            # so no additional audio processing is needed here.
            audio_url = task_params_from_db.get("audio_url")
            if audio_url:
                orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Audio requested \u2014 already muxed by last chain join, passthrough")

            task_scoped_output = _materialize_chain_passthrough_output(
                chain_output=chain_output,
                task_id=task_id,
                main_output_dir_base=main_output_dir_base,
            )
            return True, task_scoped_output

        # --- 1. Extract Parameters ---
        clip_list = task_params_from_db.get("clip_list", [])
        transition_task_ids = task_params_from_db.get("transition_task_ids", [])
        blend_frames = task_params_from_db.get("blend_frames", 15)
        target_fps = task_params_from_db.get("fps", 16)
        audio_url = task_params_from_db.get("audio_url")

        # NOTE: gap_from_clip1/gap_from_clip2 are intentionally NOT read from task_params
        # because the orchestrator calculated them from raw gap_frame_count (before 4n+1
        # quantization), while segment tasks use quantized gap_for_guide. This mismatch
        # caused a 1-frame alignment bug. Gap values MUST come from each transition's
        # output_location (ground truth from VACE). Fallbacks only for legacy compatibility.
        gap_from_clip1 = task_params_from_db.get("gap_from_clip1", 8)  # Legacy fallback only
        gap_from_clip2 = task_params_from_db.get("gap_from_clip2", 9)  # Legacy fallback only

        num_clips = len(clip_list)
        num_transitions = len(transition_task_ids)
        expected_transitions = num_clips - 1

        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: {num_clips} clips, {num_transitions} transitions")
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: blend_frames={blend_frames} (gap values from per-transition output, fallback={gap_from_clip1}/{gap_from_clip2})")

        if num_clips < 2:
            return False, "clip_list must contain at least 2 clips"

        if num_transitions != expected_transitions:
            return False, f"Expected {expected_transitions} transitions for {num_clips} clips, got {num_transitions}"

        # --- 2. Create Working Directory ---
        stitch_dir = main_output_dir_base / f"final_stitch_{task_id[:8]}"
        stitch_dir.mkdir(parents=True, exist_ok=True)

        # --- 3. Fetch Transition Outputs ---
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Fetching transition outputs...")
        transitions = []

        for i, trans_task_id in enumerate(transition_task_ids):
            # Get the output from the completed transition task
            trans_output = db_ops.get_task_output_location_from_db(trans_task_id)
            if not trans_output:
                return False, f"Failed to get output for transition task {trans_task_id}"

            # Parse the JSON output from transition_only mode
            try:
                trans_data = json.loads(trans_output)
                trans_url = trans_data.get("transition_url")
                if not trans_url:
                    return False, f"Transition task {trans_task_id} output missing transition_url"

                # Extract per-transition blend values
                # context_from_clip1 = frames from clip before transition
                # context_from_clip2 = frames from clip after transition
                ctx_clip1 = trans_data.get("context_from_clip1", blend_frames)
                ctx_clip2 = trans_data.get("context_from_clip2", blend_frames)
                trans_blend = trans_data.get("blend_frames", min(ctx_clip1, ctx_clip2))

                trans_frames = trans_data.get("frames")
                gap_frames = trans_data.get("gap_frames")

                # Verify structure consistency: frames = ctx1 + gap + ctx2
                if trans_frames and gap_frames:
                    expected_total = ctx_clip1 + gap_frames + ctx_clip2
                    if expected_total != trans_frames:
                        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u26a0\ufe0f Transition {i} structure mismatch!")
                        orchestrator_logger.debug(f"[FINAL_STITCH]   frames={trans_frames}, but ctx1({ctx_clip1}) + gap({gap_frames}) + ctx2({ctx_clip2}) = {expected_total}")

                # Extract gap values from transition output (ground truth from VACE)
                trans_gap1 = trans_data.get("gap_from_clip1")
                trans_gap2 = trans_data.get("gap_from_clip2")

                # Log whether we're using ground truth or fallback values
                if trans_gap1 is not None and trans_gap2 is not None:
                    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Transition {i}: Using ground truth gap values from VACE: gap1={trans_gap1}, gap2={trans_gap2}")
                else:
                    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u26a0\ufe0f Transition {i}: Missing gap values in output, using fallback: gap1={gap_from_clip1}, gap2={gap_from_clip2}")
                    trans_gap1 = gap_from_clip1
                    trans_gap2 = gap_from_clip2

                transitions.append({
                    "url": trans_url,
                    "index": trans_data.get("transition_index", i),
                    "frames": trans_frames,
                    "fps": trans_data.get("fps"),  # FPS used during generation
                    "gap_frames": gap_frames,
                    "blend_frames": trans_blend,
                    "context_from_clip1": ctx_clip1,  # For clip->transition crossfade
                    "context_from_clip2": ctx_clip2,  # For transition->clip crossfade
                    "gap_from_clip1": trans_gap1,
                    "gap_from_clip2": trans_gap2,
                    # Additional debug info from transition output
                    "clip1_context_start_idx": trans_data.get("clip1_context_start_idx"),
                    "clip1_context_end_idx": trans_data.get("clip1_context_end_idx"),
                    "clip2_context_start_idx": trans_data.get("clip2_context_start_idx"),
                    "clip2_context_end_idx": trans_data.get("clip2_context_end_idx"),
                    "clip1_total_frames": trans_data.get("clip1_total_frames"),
                    "clip2_total_frames": trans_data.get("clip2_total_frames"),
                })
                orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Transition {i}: frames={trans_frames}, structure=[{ctx_clip1}+{gap_frames}+{ctx_clip2}], blend={trans_blend}")
            except json.JSONDecodeError:
                # Fallback: treat as direct URL (legacy mode)
                transitions.append({
                    "url": trans_output,
                    "index": i,
                    "blend_frames": blend_frames,
                    "context_from_clip1": blend_frames,
                    "context_from_clip2": blend_frames,
                })
                orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Transition {i} (raw URL, using defaults): {trans_output}")

        # Sort transitions by index
        transitions.sort(key=lambda t: t["index"])

        # --- ALIGNMENT VERIFICATION TABLE ---
        # This is the key diagnostic: shows exactly what each transition expects
        # and makes mismatches immediately obvious
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: ")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2551                      TRANSITION ALIGNMENT TABLE                              \u2551")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2551 Trans \u2502 Clip1 Frames \u2502 Clip1 Ctx Idx  \u2502 Clip2 Frames \u2502 Clip2 Ctx Idx  \u2502 Gap  \u2551")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")

        for i, trans in enumerate(transitions):
            clip1_total = trans.get("clip1_total_frames", "?")
            clip2_total = trans.get("clip2_total_frames", "?")
            ctx1_start = trans.get("clip1_context_start_idx", "?")
            ctx1_end = trans.get("clip1_context_end_idx", "?")
            ctx2_start = trans.get("clip2_context_start_idx", "?")
            ctx2_end = trans.get("clip2_context_end_idx", "?")
            gap1 = trans.get("gap_from_clip1", "?")
            gap2 = trans.get("gap_from_clip2", "?")

            # Format context indices
            ctx1_str = f"[{ctx1_start}:{ctx1_end})" if ctx1_start != "?" else "N/A"
            ctx2_str = f"[{ctx2_start}:{ctx2_end})" if ctx2_start != "?" else "N/A"

            orchestrator_logger.debug(f"[FINAL_STITCH] \u2551  {i:2d}   \u2502     {str(clip1_total):4s}     \u2502 {ctx1_str:14s} \u2502     {str(clip2_total):4s}     \u2502 {ctx2_str:14s} \u2502{gap1}/{gap2:2}\u2551")

        orchestrator_logger.debug(f"[FINAL_STITCH] \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
        orchestrator_logger.debug(f"[FINAL_STITCH] ")
        orchestrator_logger.debug(f"[FINAL_STITCH] Legend: Ctx Idx = context frame indices extracted for VACE")
        orchestrator_logger.debug(f"[FINAL_STITCH]         Gap = gap_from_clip1/gap_from_clip2 (frames trimmed from each clip)")
        orchestrator_logger.debug(f"[FINAL_STITCH] ")

        # Validate gap values are consistent across transitions (current architecture assumes this)
        if len(transitions) > 1:
            first_gap1 = transitions[0].get("gap_from_clip1", gap_from_clip1)
            first_gap2 = transitions[0].get("gap_from_clip2", gap_from_clip2)
            for t in transitions[1:]:
                t_gap1 = t.get("gap_from_clip1", gap_from_clip1)
                t_gap2 = t.get("gap_from_clip2", gap_from_clip2)
                if t_gap1 != first_gap1 or t_gap2 != first_gap2:
                    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u26a0\ufe0f WARNING - Inconsistent gap values across transitions!")
                    orchestrator_logger.debug(f"[FINAL_STITCH]   Transition 0: gap1={first_gap1}, gap2={first_gap2}")
                    orchestrator_logger.debug(f"[FINAL_STITCH]   Transition {t['index']}: gap1={t_gap1}, gap2={t_gap2}")

        # --- 4. Download All Videos ---
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Downloading clips and transitions...")

        clip_paths = []
        for i, clip in enumerate(clip_list):
            clip_url = clip.get("url")
            if not clip_url:
                return False, f"Clip {i} missing 'url'"

            local_path = download_video_if_url(
                clip_url,
                download_target_dir=stitch_dir,
                task_id_for_logging=task_id,
                descriptive_name=f"clip_{i}"
            )
            if not local_path:
                return False, f"Failed to download clip {i}: {clip_url}"
            clip_paths.append(Path(local_path))
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Downloaded clip {i}: {local_path}")

        transition_paths = []
        for i, trans in enumerate(transitions):
            trans_url = trans.get("url")
            local_path = download_video_if_url(
                trans_url,
                download_target_dir=stitch_dir,
                task_id_for_logging=task_id,
                descriptive_name=f"transition_{i}"
            )
            if not local_path:
                return False, f"Failed to download transition {i}: {trans_url}"
            transition_paths.append(Path(local_path))
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Downloaded transition {i}: {local_path}")

        # --- 4a. Resample clips to match transition FPS ---
        # Transitions were generated at a specific FPS (min of source clip FPS).
        # Downloaded clips may be at a higher native FPS, so resample to match.
        transition_fps = transitions[0].get("fps") if transitions else None
        if transition_fps:
            target_fps = transition_fps  # Use the FPS from generation for final output too
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Resampling clips to transition FPS: {transition_fps}")
            for i, clip_path in enumerate(clip_paths):
                try:
                    resampled = ensure_video_fps(clip_path, transition_fps, output_dir=stitch_dir)
                    if resampled != clip_path:
                        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Resampled clip {i} to {transition_fps}fps")
                        clip_paths[i] = resampled
                except (OSError, ValueError, RuntimeError) as e:
                    return False, f"Failed to resample clip {i} to {transition_fps}fps: {e}"
        else:
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: No FPS in transition metadata, using clips at native FPS (target_fps={target_fps})")

        # --- 4b. FRAME COUNT VERIFICATION ---
        # Verify actual clip frame counts match what transitions expected
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Verifying clip frame counts...")
        frame_count_mismatches = []

        for i, clip_path in enumerate(clip_paths):
            actual_frames, _ = get_video_frame_count_and_fps(str(clip_path))

            # Check against transitions that reference this clip
            # Transition i uses clip i as clip1 (if i < num_transitions)
            if i < len(transitions):
                expected_clip1 = transitions[i].get("clip1_total_frames")
                if expected_clip1 is not None and expected_clip1 != actual_frames:
                    frame_count_mismatches.append(
                        f"Clip {i}: actual={actual_frames}, trans[{i}] expected clip1={expected_clip1}"
                    )
                    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u26a0\ufe0f FRAME COUNT MISMATCH: Clip {i} has {actual_frames} frames, but transition {i} expected {expected_clip1}")

            # Transition i-1 uses clip i as clip2 (if i > 0)
            if i > 0:
                expected_clip2 = transitions[i - 1].get("clip2_total_frames")
                if expected_clip2 is not None and expected_clip2 != actual_frames:
                    frame_count_mismatches.append(
                        f"Clip {i}: actual={actual_frames}, trans[{i-1}] expected clip2={expected_clip2}"
                    )
                    orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u26a0\ufe0f FRAME COUNT MISMATCH: Clip {i} has {actual_frames} frames, but transition {i-1} expected {expected_clip2}")

        if frame_count_mismatches:
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u274c FOUND {len(frame_count_mismatches)} FRAME COUNT MISMATCHES!")
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: This may indicate clips were re-encoded or transitions used different source files.")
        else:
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: \u2713 All clip frame counts match transition expectations")

        # --- 4b. PIXEL VERIFICATION: Compare transition context frames with original clips ---
        # This verifies that VACE preserved context frames pixel-identically
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Verifying pixel alignment between clips and transitions...")

        for i, trans in enumerate(transitions):
            ctx1 = trans.get("context_from_clip1", 0)
            ctx2 = trans.get("context_from_clip2", 0)

            if ctx1 > 0 and i < len(clip_paths):
                # Compare: clip[i]'s last ctx1 frames (before gap) vs transition's first ctx1 frames
                try:
                    clip_frames_list = extract_frames_from_video(str(clip_paths[i]))
                    trans_frames_list = extract_frames_from_video(str(transition_paths[i]))

                    if clip_frames_list and trans_frames_list:
                        # Clip context: last (gap_from_clip1 + ctx1) to last gap_from_clip1 frames
                        # i.e., frames that will remain after trimming, specifically the last ctx1 of those
                        gap1 = trans.get("gap_from_clip1", gap_from_clip1)
                        clip_ctx_start = len(clip_frames_list) - gap1 - ctx1
                        clip_ctx_end = len(clip_frames_list) - gap1

                        clip_context = clip_frames_list[clip_ctx_start:clip_ctx_end]
                        trans_context = trans_frames_list[:ctx1]

                        if len(clip_context) == len(trans_context) == ctx1:
                            # Compare first and last frame of context region
                            import numpy as np

                            # First frame comparison
                            diff_first = np.abs(clip_context[0].astype(float) - trans_context[0].astype(float)).mean()
                            # Last frame comparison
                            diff_last = np.abs(clip_context[-1].astype(float) - trans_context[-1].astype(float)).mean()

                            if diff_first < 1.0 and diff_last < 1.0:
                                orchestrator_logger.debug(f"[PIXEL_CHECK] \u2713 Transition {i} START context: PIXEL IDENTICAL (diff={diff_first:.2f}, {diff_last:.2f})")
                            else:
                                orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i} START context: MISMATCH DETECTED!")
                                orchestrator_logger.debug(f"[PIXEL_CHECK]   First frame diff: {diff_first:.2f}, Last frame diff: {diff_last:.2f}")
                                orchestrator_logger.debug(f"[PIXEL_CHECK]   Clip frames [{clip_ctx_start}:{clip_ctx_end}] vs Transition frames [0:{ctx1}]")

                            # === OFFSET DETECTION: Compare trans[0] against clip frames at different offsets ===
                            # This helps identify if there's a systematic frame shift
                            orchestrator_logger.debug(f"[OFFSET_DETECT] Transition {i} START: Comparing trans[0] against clip frames at offsets -2 to +2")
                            orchestrator_logger.debug(f"[OFFSET_DETECT]   Expected alignment: trans[0] should match clip[{clip_ctx_start}]")

                            trans_first_frame = trans_context[0].astype(float)
                            offset_diffs = {}
                            for offset in range(-2, 3):  # -2, -1, 0, +1, +2
                                test_idx = clip_ctx_start + offset
                                if 0 <= test_idx < len(clip_frames_list):
                                    test_frame = clip_frames_list[test_idx].astype(float)
                                    diff = np.abs(trans_first_frame - test_frame).mean()
                                    offset_diffs[offset] = diff
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   offset={offset:+d} (clip[{test_idx}]): diff={diff:.2f}")

                            # Find best matching offset
                            if offset_diffs:
                                best_offset = min(offset_diffs, key=offset_diffs.get)
                                best_diff = offset_diffs[best_offset]
                                if best_offset != 0:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> BEST MATCH at offset={best_offset:+d} (diff={best_diff:.2f}) - ALIGNMENT BUG DETECTED!")
                                else:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> Best match at offset=0 (diff={best_diff:.2f}) - alignment appears correct")

                            # Also check the LAST frame of context for offset
                            orchestrator_logger.debug(f"[OFFSET_DETECT] Transition {i} START: Comparing trans[{ctx1-1}] against clip frames at offsets -2 to +2")
                            trans_last_ctx_frame = trans_context[-1].astype(float)
                            expected_last_idx = clip_ctx_end - 1  # Last frame of context
                            orchestrator_logger.debug(f"[OFFSET_DETECT]   Expected alignment: trans[{ctx1-1}] should match clip[{expected_last_idx}]")

                            offset_diffs_last = {}
                            for offset in range(-2, 3):
                                test_idx = expected_last_idx + offset
                                if 0 <= test_idx < len(clip_frames_list):
                                    test_frame = clip_frames_list[test_idx].astype(float)
                                    diff = np.abs(trans_last_ctx_frame - test_frame).mean()
                                    offset_diffs_last[offset] = diff
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   offset={offset:+d} (clip[{test_idx}]): diff={diff:.2f}")

                            if offset_diffs_last:
                                best_offset_last = min(offset_diffs_last, key=offset_diffs_last.get)
                                best_diff_last = offset_diffs_last[best_offset_last]
                                if best_offset_last != 0:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> BEST MATCH at offset={best_offset_last:+d} (diff={best_diff_last:.2f}) - ALIGNMENT BUG DETECTED!")
                                else:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> Best match at offset=0 (diff={best_diff_last:.2f}) - alignment appears correct")
                        else:
                            orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i}: Frame count mismatch for START context")
                            orchestrator_logger.debug(f"[PIXEL_CHECK]   Expected {ctx1}, got clip={len(clip_context)}, trans={len(trans_context)}")
                except (OSError, ValueError, RuntimeError) as e:
                    orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i}: Error comparing START context: {e}")

            if ctx2 > 0 and i + 1 < len(clip_paths):
                # Compare: transition's last ctx2 frames vs clip[i+1]'s first ctx2 frames (after gap)
                try:
                    next_clip_frames = extract_frames_from_video(str(clip_paths[i + 1]))
                    trans_frames_list = extract_frames_from_video(str(transition_paths[i]))

                    if next_clip_frames and trans_frames_list:
                        gap2 = trans.get("gap_from_clip2", gap_from_clip2)
                        # Next clip context: frames [gap2 : gap2 + ctx2]
                        next_clip_context = next_clip_frames[gap2:gap2 + ctx2]
                        trans_end_context = trans_frames_list[-ctx2:]

                        if len(next_clip_context) == len(trans_end_context) == ctx2:
                            import numpy as np

                            diff_first = np.abs(next_clip_context[0].astype(float) - trans_end_context[0].astype(float)).mean()
                            diff_last = np.abs(next_clip_context[-1].astype(float) - trans_end_context[-1].astype(float)).mean()

                            if diff_first < 1.0 and diff_last < 1.0:
                                orchestrator_logger.debug(f"[PIXEL_CHECK] \u2713 Transition {i} END context: PIXEL IDENTICAL (diff={diff_first:.2f}, {diff_last:.2f})")
                            else:
                                orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i} END context: MISMATCH DETECTED!")
                                orchestrator_logger.debug(f"[PIXEL_CHECK]   First frame diff: {diff_first:.2f}, Last frame diff: {diff_last:.2f}")
                                orchestrator_logger.debug(f"[PIXEL_CHECK]   Next clip frames [{gap2}:{gap2 + ctx2}] vs Transition frames [-{ctx2}:]")

                            # === OFFSET DETECTION for END context ===
                            orchestrator_logger.debug(f"[OFFSET_DETECT] Transition {i} END: Comparing trans[-{ctx2}] against next_clip frames at offsets -2 to +2")
                            expected_clip2_start = gap2
                            orchestrator_logger.debug(f"[OFFSET_DETECT]   Expected alignment: trans[-{ctx2}] should match next_clip[{expected_clip2_start}]")

                            trans_end_first_frame = trans_end_context[0].astype(float)
                            offset_diffs_end = {}
                            for offset in range(-2, 3):
                                test_idx = expected_clip2_start + offset
                                if 0 <= test_idx < len(next_clip_frames):
                                    test_frame = next_clip_frames[test_idx].astype(float)
                                    diff = np.abs(trans_end_first_frame - test_frame).mean()
                                    offset_diffs_end[offset] = diff
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   offset={offset:+d} (next_clip[{test_idx}]): diff={diff:.2f}")

                            if offset_diffs_end:
                                best_offset_end = min(offset_diffs_end, key=offset_diffs_end.get)
                                best_diff_end = offset_diffs_end[best_offset_end]
                                if best_offset_end != 0:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> BEST MATCH at offset={best_offset_end:+d} (diff={best_diff_end:.2f}) - ALIGNMENT BUG DETECTED!")
                                else:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> Best match at offset=0 (diff={best_diff_end:.2f}) - alignment appears correct")

                            # Also check last frame of END context
                            orchestrator_logger.debug(f"[OFFSET_DETECT] Transition {i} END: Comparing trans[-1] against next_clip frames at offsets -2 to +2")
                            expected_clip2_end = gap2 + ctx2 - 1
                            orchestrator_logger.debug(f"[OFFSET_DETECT]   Expected alignment: trans[-1] should match next_clip[{expected_clip2_end}]")

                            trans_very_last_frame = trans_end_context[-1].astype(float)
                            offset_diffs_end_last = {}
                            for offset in range(-2, 3):
                                test_idx = expected_clip2_end + offset
                                if 0 <= test_idx < len(next_clip_frames):
                                    test_frame = next_clip_frames[test_idx].astype(float)
                                    diff = np.abs(trans_very_last_frame - test_frame).mean()
                                    offset_diffs_end_last[offset] = diff
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   offset={offset:+d} (next_clip[{test_idx}]): diff={diff:.2f}")

                            if offset_diffs_end_last:
                                best_offset_end_last = min(offset_diffs_end_last, key=offset_diffs_end_last.get)
                                best_diff_end_last = offset_diffs_end_last[best_offset_end_last]
                                if best_offset_end_last != 0:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> BEST MATCH at offset={best_offset_end_last:+d} (diff={best_diff_end_last:.2f}) - ALIGNMENT BUG DETECTED!")
                                else:
                                    orchestrator_logger.debug(f"[OFFSET_DETECT]   >>> Best match at offset=0 (diff={best_diff_end_last:.2f}) - alignment appears correct")
                        else:
                            orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i}: Frame count mismatch for END context")
                except (OSError, ValueError, RuntimeError) as e:
                    orchestrator_logger.debug(f"[PIXEL_CHECK] \u26a0\ufe0f Transition {i}: Error comparing END context: {e}")

        # --- 5. Trim Clips and Build Stitch List ---
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Preparing clips for stitching...")

        import tempfile
        stitch_videos = []
        stitch_blends = []

        for i, clip_path in enumerate(clip_paths):
            clip_frames, clip_fps = get_video_frame_count_and_fps(str(clip_path))
            if not clip_frames:
                return False, f"Could not get frame count for clip {i}"

            # Determine trim amounts using PER-TRANSITION gap values (ground truth from VACE)
            # trim_start: how much to trim from START of this clip
            #   - Uses gap_from_clip2 from the PREVIOUS transition (i-1)
            #   - Because transition[i-1] connects clip[i-1] -> clip[i], and gap_from_clip2 is how much of clip[i]'s start was used as gap
            # trim_end: how much to trim from END of this clip
            #   - Uses gap_from_clip1 from the CURRENT transition (i)
            #   - Because transition[i] connects clip[i] -> clip[i+1], and gap_from_clip1 is how much of clip[i]'s end was used as gap

            if i > 0:
                # Get gap_from_clip2 from previous transition (transition i-1 connects to this clip)
                prev_trans = transitions[i - 1]
                trim_start = prev_trans.get("gap_from_clip2", gap_from_clip2)
                trim_start_source = f"trans[{i-1}].gap_from_clip2"
            else:
                trim_start = 0  # First clip: no start trim
                trim_start_source = "none (first clip)"

            if i < num_clips - 1:
                # Get gap_from_clip1 from current transition (this clip connects to transition i)
                curr_trans = transitions[i]
                trim_end = curr_trans.get("gap_from_clip1", gap_from_clip1)
                trim_end_source = f"trans[{i}].gap_from_clip1"
            else:
                trim_end = 0  # Last clip: no end trim
                trim_end_source = "none (last clip)"

            frames_to_keep = clip_frames - trim_start - trim_end
            if frames_to_keep <= 0:
                return False, f"Clip {i} has {clip_frames} frames but needs {trim_start + trim_end} trimmed"

            # Log trim details with source of values
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Clip {i}: {clip_frames} frames total")
            orchestrator_logger.debug(f"[FINAL_STITCH]   trim_start={trim_start} (from {trim_start_source}), trim_end={trim_end} (from {trim_end_source})")
            orchestrator_logger.debug(f"[FINAL_STITCH]   keeping frames [{trim_start}:{clip_frames - trim_end}] = {frames_to_keep} frames")

            # Extract trimmed clip
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=stitch_dir) as tf:
                trimmed_path = Path(tf.name)

            start_frame = trim_start
            end_frame = clip_frames - trim_end - 1 if trim_end > 0 else None

            try:
                extract_frame_range_to_video(
                    input_video_path=clip_path,
                    output_video_path=trimmed_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=target_fps)
            except (OSError, ValueError, RuntimeError) as e:
                return False, f"Failed to trim clip {i}: {e}"

            # Add to stitch list
            stitch_videos.append(trimmed_path)

            # Add transition after this clip (except after last clip)
            if i < num_clips - 1:
                # Use per-transition blend values:
                # - clip[i] -> transition[i]: context_from_clip1 (context from clip i in transition)
                # - transition[i] -> clip[i+1]: context_from_clip2 (context from clip i+1 in transition)
                trans_info = transitions[i]
                blend_clip_to_trans = trans_info.get("context_from_clip1", blend_frames)
                blend_trans_to_clip = trans_info.get("context_from_clip2", blend_frames)

                stitch_blends.append(blend_clip_to_trans)  # Blend between clip and transition
                stitch_videos.append(transition_paths[i])
                stitch_blends.append(blend_trans_to_clip)  # Blend between transition and next clip

                orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Crossfades for transition {i}: clip\u2192trans={blend_clip_to_trans}, trans\u2192clip={blend_trans_to_clip}")

        # --- 5b. FINAL FRAME ACCOUNTING ---
        # Show exactly how the final video will be composed
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: ")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2551                        FINAL VIDEO FRAME ACCOUNTING                         \u2551")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")

        final_frame_position = 0
        for idx, video_path in enumerate(stitch_videos):
            video_frames, _ = get_video_frame_count_and_fps(str(video_path))
            if video_frames is None:
                video_frames = 0

            # Determine if this is a clip or transition
            # Pattern: clip0, trans0, clip1, trans1, clip2, ... (clips at even indices, transitions at odd)
            is_clip = (idx % 2 == 0)
            source_idx = idx // 2

            if is_clip:
                # Get the original clip info
                orig_frames, _ = get_video_frame_count_and_fps(str(clip_paths[source_idx])) if source_idx < len(clip_paths) else (0, 0)

                # Reconstruct trim info
                if source_idx > 0:
                    t_start = transitions[source_idx - 1].get("gap_from_clip2", gap_from_clip2)
                else:
                    t_start = 0
                if source_idx < len(transitions):
                    t_end = transitions[source_idx].get("gap_from_clip1", gap_from_clip1)
                else:
                    t_end = 0

                source_desc = f"Clip {source_idx}: frames [{t_start}:{orig_frames - t_end}]"
            else:
                trans_idx = source_idx
                trans_info = transitions[trans_idx] if trans_idx < len(transitions) else {}
                ctx1 = trans_info.get("context_from_clip1", "?")
                gap = trans_info.get("gap_frames", "?")
                ctx2 = trans_info.get("context_from_clip2", "?")
                source_desc = f"Trans {trans_idx}: [{ctx1} ctx1 + {gap} gap + {ctx2} ctx2]"

            # Get blend info
            _blend_before = stitch_blends[idx - 1] if idx > 0 and idx - 1 < len(stitch_blends) else 0
            blend_after = stitch_blends[idx] if idx < len(stitch_blends) else 0

            end_frame = final_frame_position + video_frames - 1
            orchestrator_logger.debug(f"[FINAL_STITCH] \u2551 [{final_frame_position:5d}-{end_frame:5d}] {source_desc:<45s} ({video_frames:3d}f) \u2551")

            if blend_after > 0 and idx < len(stitch_videos) - 1:
                orchestrator_logger.debug(f"[FINAL_STITCH] \u2551              \u2195 crossfade {blend_after} frames                                     \u2551")

            final_frame_position += video_frames - blend_after  # Subtract overlap

        orchestrator_logger.debug(f"[FINAL_STITCH] \u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u2551 Expected total frames (approximate): {final_frame_position:<38d} \u2551")
        orchestrator_logger.debug(f"[FINAL_STITCH] \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
        orchestrator_logger.debug(f"[FINAL_STITCH] ")

        # --- 6. Stitch Everything Together ---
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Stitching {len(stitch_videos)} videos with {len(stitch_blends)} blend points...")

        final_output_path, initial_db_location = prepare_output_path_with_upload(
            task_id=task_id,
            filename=f"{task_id}_joined.mp4",
            main_output_dir_base=main_output_dir_base,
            task_type="join_final_stitch")

        try:
            stitch_videos_with_crossfade(
                video_paths=stitch_videos,
                blend_frame_counts=stitch_blends,
                output_video_path=final_output_path,
                fps=target_fps,
                crossfade_mode="linear_sharp",
                crossfade_sharp_amt=0.3)

            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Successfully stitched all videos")

        except (OSError, ValueError, RuntimeError) as e:
            return False, f"Failed to stitch videos: {e}"

        # --- 7. Add Audio (if provided) ---
        if audio_url:
            orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Adding audio from {audio_url}")
            try:
                audio_local = download_video_if_url(
                    audio_url,
                    download_target_dir=stitch_dir,
                    task_id_for_logging=task_id,
                    descriptive_name="audio"
                )
                if audio_local:
                    with_audio_path = final_output_path.with_name(f"{task_id}_joined_audio.mp4")
                    success = add_audio_to_video(
                        input_video_path=str(final_output_path),
                        audio_url=audio_local,
                        output_video_path=str(with_audio_path))
                    if success and with_audio_path.exists():
                        final_output_path = with_audio_path
                        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Audio added successfully")
            except (OSError, ValueError, RuntimeError) as audio_err:
                orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Failed to add audio (continuing without): {audio_err}")

        # --- 8. Verify and Upload ---
        if not final_output_path.exists():
            return False, "Final output file does not exist"

        file_size = final_output_path.stat().st_size
        if file_size == 0:
            return False, "Final output file is empty"

        final_frames, final_fps = get_video_frame_count_and_fps(str(final_output_path))
        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Final video: {final_frames} frames @ {final_fps} fps, {file_size} bytes")

        # Upload
        final_db_location = upload_and_get_final_output_location(
            local_file_path=final_output_path,
            initial_db_location=initial_db_location)

        orchestrator_logger.debug(f"[FINAL_STITCH] Task {task_id}: Complete - {final_db_location}")
        return True, final_db_location

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        error_msg = f"Unexpected error in final stitch handler: {e}"
        orchestrator_logger.debug(f"[FINAL_STITCH_ERROR] Task {task_id}: {error_msg}", exc_info=True)
        return False, error_msg
