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
    standardize_video_aspect_ratio,
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
    try:
        # --- Chain Mode Passthrough ---
        # When created by the chain pattern, the last join in the chain already produced
        # the fully concatenated video. We just pass through its output (+ optional audio).
        chain_mode = task_params_from_db.get("chain_mode", False)
        if chain_mode:
            transition_task_ids = task_params_from_db.get("transition_task_ids", [])
            if not transition_task_ids:
                return False, "chain_mode=True but no transition_task_ids (last chain join ID) provided"

            # The single ID is the last join in the chain
            last_join_id = transition_task_ids[0]
            chain_output = db_ops.get_task_output_location_from_db(last_join_id)
            if not chain_output:
                return False, f"Failed to get output from chain join task {last_join_id}"

            # The chain join's output_location is the final video URL/path — use it directly.
            # In chain mode, the last join_clips_child already muxes audio when is_last_join=True,
            # so no additional audio processing is needed here.
            audio_url = task_params_from_db.get("audio_url")

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

        orchestrator_logger.debug_block(
            "STITCH",
            {
                "task": task_id,
                "chain_mode": chain_mode,
                "clip_count": num_clips,
                "transition_count": num_transitions,
                "expected_transitions": expected_transitions,
                "blend_frames": blend_frames,
                "fallback_gap": (gap_from_clip1, gap_from_clip2),
                "target_fps": target_fps,
                "audio": bool(audio_url),
            },
            task_id=task_id,
        )

        if num_clips < 2:
            return False, "clip_list must contain at least 2 clips"

        if num_transitions != expected_transitions:
            return False, f"Expected {expected_transitions} transitions for {num_clips} clips, got {num_transitions}"

        # --- 2. Create Working Directory ---
        stitch_dir = main_output_dir_base / f"final_stitch_{task_id[:8]}"
        stitch_dir.mkdir(parents=True, exist_ok=True)

        # --- 3. Fetch Transition Outputs ---
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
                        orchestrator_logger.debug_anomaly(
                            "FINAL_STITCH",
                            (
                                f"transition {i} structure mismatch frames={trans_frames}, "
                                f"ctx1={ctx_clip1}, gap={gap_frames}, ctx2={ctx_clip2}, expected={expected_total}"
                            ),
                            task_id=task_id,
                        )

                # Extract gap values from transition output (ground truth from VACE)
                trans_gap1 = trans_data.get("gap_from_clip1")
                trans_gap2 = trans_data.get("gap_from_clip2")

                # Log whether we're using ground truth or fallback values
                if trans_gap1 is None or trans_gap2 is None:
                    orchestrator_logger.debug_anomaly(
                        "FINAL_STITCH",
                        f"transition {i} missing gap values, using fallback {gap_from_clip1}/{gap_from_clip2}",
                        task_id=task_id,
                    )
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
            except json.JSONDecodeError:
                # Fallback: treat as direct URL (legacy mode)
                transitions.append({
                    "url": trans_output,
                    "index": i,
                    "blend_frames": blend_frames,
                    "context_from_clip1": blend_frames,
                    "context_from_clip2": blend_frames,
                })
                orchestrator_logger.debug_anomaly(
                    "FINAL_STITCH",
                    f"transition {i} output was raw URL; using default blend/context values",
                    task_id=task_id,
                )

        # Sort transitions by index
        transitions.sort(key=lambda t: t["index"])

        transition_context_ranges = [
            (
                trans.get("index", idx),
                trans.get("clip1_context_start_idx"),
                trans.get("clip1_context_end_idx"),
                trans.get("clip2_context_start_idx"),
                trans.get("clip2_context_end_idx"),
                trans.get("gap_from_clip1", gap_from_clip1),
                trans.get("gap_from_clip2", gap_from_clip2),
            )
            for idx, trans in enumerate(transitions)
        ]
        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "transition_alignment",
                "transitions": len(transitions),
                "context_ranges": transition_context_ranges,
            },
            task_id=task_id,
        )

        # Validate gap values are consistent across transitions (current architecture assumes this)
        gap_inconsistencies = []
        if len(transitions) > 1:
            first_gap1 = transitions[0].get("gap_from_clip1", gap_from_clip1)
            first_gap2 = transitions[0].get("gap_from_clip2", gap_from_clip2)
            for t in transitions[1:]:
                t_gap1 = t.get("gap_from_clip1", gap_from_clip1)
                t_gap2 = t.get("gap_from_clip2", gap_from_clip2)
                if t_gap1 != first_gap1 or t_gap2 != first_gap2:
                    gap_inconsistencies.append(f"{t['index']}:{t_gap1}/{t_gap2}")
        if gap_inconsistencies:
            orchestrator_logger.debug_anomaly(
                "FINAL_STITCH",
                f"gap inconsistencies across transitions: {gap_inconsistencies}",
                task_id=task_id,
            )

        # --- 4. Download All Videos ---

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
        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "downloads",
                "clips": len(clip_paths),
                "transitions": len(transition_paths),
                "first_clip": clip_paths[0] if clip_paths else None,
                "first_transition": transition_paths[0] if transition_paths else None,
            },
            task_id=task_id,
        )

        # --- 4a. Resample clips to match transition FPS ---
        # Transitions are generated at a fixed FPS (default 16). Downloaded clips
        # may be at a higher native FPS, so resample to match the transition.
        transition_fps = transitions[0].get("fps") if transitions else None
        resampled_clips = []
        if transition_fps:
            target_fps = transition_fps  # Use the FPS from generation for final output too
            for i, clip_path in enumerate(clip_paths):
                try:
                    resampled = ensure_video_fps(clip_path, transition_fps, output_dir=stitch_dir)
                    if resampled != clip_path:
                        resampled_clips.append(i)
                        clip_paths[i] = resampled
                except (OSError, ValueError, RuntimeError) as e:
                    return False, f"Failed to resample clip {i} to {transition_fps}fps: {e}"

        # --- 4b. Standardize clip aspect ratios to match transition ---
        # The segment handler standardized clips to a common aspect ratio before
        # generation. Use the transition's resolution (ground truth) to match,
        # falling back to clip 1's dimensions if not available in metadata.
        aspect_reference = None
        standardized_clips = []
        try:
            import subprocess

            def _get_video_dimensions(video_path):
                probe_cmd = [
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                    str(video_path)
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return None, None
                w, h = result.stdout.strip().split(',')
                return int(w), int(h)

            # Prefer resolution from transition metadata (exact match for what was generated)
            trans_resolution = transitions[0].get("resolution") if transitions else None
            if trans_resolution and len(trans_resolution) == 2:
                ref_w, ref_h = int(trans_resolution[0]), int(trans_resolution[1])
                aspect_reference = "transition"
            else:
                ref_w, ref_h = _get_video_dimensions(clip_paths[0])
                aspect_reference = "clip_0"

            if ref_w and ref_h:
                ref_aspect = ref_w / ref_h
                for i, clip_path in enumerate(clip_paths):
                    clip_w, clip_h = _get_video_dimensions(clip_path)
                    if clip_w and clip_h and abs(clip_w / clip_h - ref_aspect) > 0.01:
                        standardized_path = stitch_dir / f"clip_{i}_standardized.mp4"
                        result = standardize_video_aspect_ratio(
                            input_video_path=clip_path,
                            output_video_path=standardized_path,
                            target_aspect_ratio=f"{ref_w}:{ref_h}",
                            task_id_for_logging=task_id)
                        if result is not None:
                            standardized_clips.append(i)
                            clip_paths[i] = standardized_path
                        else:
                            orchestrator_logger.debug_anomaly(
                                "FINAL_STITCH",
                                f"failed to standardize clip {i}, proceeding with original",
                                task_id=task_id,
                            )
        except (OSError, ValueError, RuntimeError) as e:
            orchestrator_logger.debug_anomaly(
                "FINAL_STITCH",
                f"aspect ratio check failed: {e}",
                task_id=task_id,
            )
        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "media_prep",
                "target_fps": target_fps,
                "transition_fps": transition_fps,
                "resampled_clips": resampled_clips or None,
                "aspect_reference": aspect_reference,
                "standardized_clips": standardized_clips or None,
            },
            task_id=task_id,
        )

        # --- 4c. FRAME COUNT VERIFICATION ---
        # Verify actual clip frame counts match what transitions expected
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

            # Transition i-1 uses clip i as clip2 (if i > 0)
            if i > 0:
                expected_clip2 = transitions[i - 1].get("clip2_total_frames")
                if expected_clip2 is not None and expected_clip2 != actual_frames:
                    frame_count_mismatches.append(
                        f"Clip {i}: actual={actual_frames}, trans[{i-1}] expected clip2={expected_clip2}"
                    )

        if frame_count_mismatches:
            orchestrator_logger.debug_anomaly(
                "FINAL_STITCH",
                f"frame count mismatches detected: {frame_count_mismatches}",
                task_id=task_id,
            )
        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "clip_frames",
                "mismatch_count": len(frame_count_mismatches),
                "mismatches": frame_count_mismatches or None,
            },
            task_id=task_id,
        )

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
                            import numpy as np

                            diff_first = np.abs(clip_context[0].astype(float) - trans_context[0].astype(float)).mean()
                            diff_last = np.abs(clip_context[-1].astype(float) - trans_context[-1].astype(float)).mean()
                            orchestrator_logger.debug(
                                f"[PIXEL_CHECK] Transition {i} START alignment verdict: "
                                f"mean_diff_first={diff_first:.2f}, mean_diff_last={diff_last:.2f}, "
                                f"aligned={diff_first < 1.0 and diff_last < 1.0}"
                            )
                        else:
                            orchestrator_logger.debug(
                                f"[PIXEL_CHECK] Transition {i} START alignment verdict: "
                                f"frame_count_mismatch expected={ctx1}, clip={len(clip_context)}, trans={len(trans_context)}"
                            )
                except (OSError, ValueError, RuntimeError) as e:
                    orchestrator_logger.debug_anomaly("PIXEL_CHECK", f"\u26a0\ufe0f Transition {i}: Error comparing START context: {e}")

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
                            orchestrator_logger.debug(
                                f"[PIXEL_CHECK] Transition {i} END alignment verdict: "
                                f"mean_diff_first={diff_first:.2f}, mean_diff_last={diff_last:.2f}, "
                                f"aligned={diff_first < 1.0 and diff_last < 1.0}"
                            )
                        else:
                            orchestrator_logger.debug(
                                f"[PIXEL_CHECK] Transition {i} END alignment verdict: "
                                f"frame_count_mismatch expected={ctx2}, clip={len(next_clip_context)}, trans={len(trans_end_context)}"
                            )
                except (OSError, ValueError, RuntimeError) as e:
                    orchestrator_logger.debug_anomaly("PIXEL_CHECK", f"\u26a0\ufe0f Transition {i}: Error comparing END context: {e}")

        # --- 5. Trim Clips and Build Stitch List ---

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

        crossfade_pairs = []
        final_frame_position = 0
        for idx, video_path in enumerate(stitch_videos):
            video_frames, _ = get_video_frame_count_and_fps(str(video_path))
            if video_frames is None:
                video_frames = 0
            blend_after = stitch_blends[idx] if idx < len(stitch_blends) else 0
            if blend_after > 0 and idx < len(stitch_videos) - 1:
                crossfade_pairs.append((idx, blend_after))
            final_frame_position += video_frames - blend_after  # Subtract overlap
        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "frame_accounting",
                "videos": len(stitch_videos),
                "crossfades": crossfade_pairs,
                "approx_total_frames": final_frame_position,
            },
            task_id=task_id,
        )

        # --- 6. Stitch Everything Together ---
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

        except (OSError, ValueError, RuntimeError) as e:
            return False, f"Failed to stitch videos: {e}"

        # --- 7. Add Audio (if provided) ---
        if audio_url:
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
            except (OSError, ValueError, RuntimeError) as audio_err:
                orchestrator_logger.debug_anomaly(
                    "FINAL_STITCH",
                    f"failed to add audio, continuing without it: {audio_err}",
                    task_id=task_id,
                )

        # --- 8. Verify and Upload ---
        if not final_output_path.exists():
            return False, "Final output file does not exist"

        file_size = final_output_path.stat().st_size
        if file_size == 0:
            return False, "Final output file is empty"

        final_frames, final_fps = get_video_frame_count_and_fps(str(final_output_path))

        # Upload
        final_db_location = upload_and_get_final_output_location(
            local_file_path=final_output_path,
            initial_db_location=initial_db_location)

        orchestrator_logger.debug_block(
            "STITCH",
            {
                "stage": "output",
                "output": final_db_location,
                "frames": final_frames,
                "fps": final_fps,
                "file_size": file_size,
            },
            task_id=task_id,
        )
        return True, final_db_location

    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        error_msg = f"Unexpected error in final stitch handler: {e}"
        orchestrator_logger.error(error_msg, task_id=task_id, exc_info=True)
        return False, error_msg
