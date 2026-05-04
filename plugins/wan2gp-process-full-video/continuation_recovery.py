from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from . import common
from . import media_io as media
from . import output_paths
from . import process_metadata
from . import status_ui

USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE = True


@dataclass(frozen=True)
class ContinuationRecoveryResult:
    merged_signatures: list[dict]
    blocked: bool = False


def merge_residual_continuations(
    ffmpeg_path: str,
    ffprobe_path: str,
    output_path: str,
    continuation_paths: list[str],
    *,
    video_codec: str,
    video_container: str,
    audio_codec_key: str,
    fps_float: float,
    selected_audio_track_no: int | None,
    source_audio_path: str | None = None,
    source_audio_start_seconds: float | None = None,
    merged_continuation_signatures: list[dict] | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    known_signatures = process_metadata.normalize_merged_continuation_signatures(merged_continuation_signatures)
    newly_merged_signatures: list[dict] = []
    undeleted_already_merged_paths: list[str] = []
    undeleted_newly_merged_paths: list[str] = []
    for continuation_path in continuation_paths:
        continuation_signature = process_metadata.make_continuation_signature(continuation_path)
        if continuation_signature is not None and process_metadata.continuation_signature_key(continuation_signature) in {process_metadata.continuation_signature_key(signature) for signature in known_signatures}:
            if os.path.isfile(continuation_path):
                try:
                    os.remove(continuation_path)
                except OSError:
                    undeleted_already_merged_paths.append(continuation_path)
            continue
        completed_frames, _ = media.probe_resume_frame_count(ffprobe_path, output_path, fps_float)
        continuation_frames, _ = media.probe_resume_frame_count(ffprobe_path, continuation_path, fps_float)
        merged_duration_seconds = (float(completed_frames + continuation_frames) / fps_float) if completed_frames > 0 or continuation_frames > 0 else 0.0
        audio_trim_seconds = media.probe_selected_audio_overhang(ffprobe_path, output_path, selected_audio_track_no, completed_frames / fps_float) if completed_frames > 0 else 0.0
        if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE:
            print(f"[Process Full Video] Residual merge: rebuilding audio from source for {merged_duration_seconds:.6f}s starting at {float(source_audio_start_seconds or 0.0):.6f}s")
        elif audio_trim_seconds > 0.0:
            print(f"[Process Full Video] Residual merge: trimming {audio_trim_seconds:.6f}s from continuation audio and clamping merged segment audio to visible video duration")
        committed_signatures = process_metadata.append_merged_continuation_signature(known_signatures, continuation_signature)
        media.concat_video_segments(
            ffmpeg_path,
            [output_path, continuation_path],
            output_path,
            video_codec,
            video_container,
            audio_codec_key,
            segment_audio_trim_seconds=[0.0, audio_trim_seconds],
            segment_audio_duration_seconds=[(float(completed_frames) / fps_float) if completed_frames > 0 else None, (float(continuation_frames) / fps_float) if continuation_frames > 0 else None],
            fps_float=fps_float,
            selected_audio_track_no=selected_audio_track_no,
            source_audio_path=source_audio_path if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_start_seconds=source_audio_start_seconds if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_duration_seconds=merged_duration_seconds if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
            source_audio_track_no=selected_audio_track_no if USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE else None,
        )
        known_signatures = committed_signatures
        newly_merged_signatures = process_metadata.append_merged_continuation_signature(newly_merged_signatures, continuation_signature)
        if os.path.isfile(continuation_path):
            try:
                os.remove(continuation_path)
            except OSError:
                undeleted_newly_merged_paths.append(continuation_path)
    return newly_merged_signatures, undeleted_already_merged_paths, undeleted_newly_merged_paths


def reconcile_frame_count_mismatch(
    ffprobe_path: str,
    output_path: str,
    continuation_paths: list[str],
    *,
    fps_float: float,
    merged_signatures: list[dict],
    verbose_level: int,
) -> tuple[list[dict], list[str], list[str]]:
    probed_frame_count, _ = media.probe_resume_frame_count(ffprobe_path, output_path, fps_float)
    recorded_frame_count = process_metadata.read_recorded_written_unique_frames(output_path)
    if recorded_frame_count <= 0 or probed_frame_count <= recorded_frame_count:
        return merged_signatures, continuation_paths, []

    remaining_extra_frames = probed_frame_count - recorded_frame_count
    updated_signatures = process_metadata.normalize_merged_continuation_signatures(merged_signatures)
    pending_paths: list[str] = []
    already_merged_paths: list[str] = []
    known_signature_keys = {process_metadata.continuation_signature_key(signature) for signature in updated_signatures}
    for continuation_path in continuation_paths:
        continuation_signature = process_metadata.make_continuation_signature(continuation_path)
        continuation_signature_key = process_metadata.continuation_signature_key(continuation_signature) if continuation_signature is not None else None
        if continuation_signature_key in known_signature_keys:
            already_merged_paths.append(continuation_path)
            continue
        continuation_frames, _ = media.probe_resume_frame_count(ffprobe_path, continuation_path, fps_float)
        if continuation_frames > 0 and continuation_frames <= remaining_extra_frames:
            remaining_extra_frames -= continuation_frames
            already_merged_paths.append(continuation_path)
            updated_signatures = process_metadata.append_merged_continuation_signature(updated_signatures, continuation_signature)
            known_signature_keys = {process_metadata.continuation_signature_key(signature) for signature in updated_signatures}
            continue
        pending_paths.append(continuation_path)

    if already_merged_paths:
        names = ", ".join(Path(path).name for path in already_merged_paths)
        common.plugin_info(f"Output contains {probed_frame_count} readable frame(s), but metadata recorded {recorded_frame_count}. Treating already included continuation file(s) as merged: {names}.")
        process_metadata.store_process_progress(output_path, written_unique_frames=probed_frame_count, merged_signatures=updated_signatures, verbose_level=verbose_level)
    elif remaining_extra_frames > 0:
        common.plugin_info(f"Output contains {probed_frame_count} readable frame(s), but metadata recorded {recorded_frame_count}. Using the real frame count for continuation.")
        process_metadata.store_process_progress(output_path, written_unique_frames=probed_frame_count, merged_signatures=updated_signatures, verbose_level=verbose_level)
    return updated_signatures, pending_paths, already_merged_paths


def recover_residual_continuations(
    *,
    plugin,
    ffmpeg_path: str,
    ffprobe_path: str,
    output_path: str,
    full_plans: list,
    output_container: str,
    fps_float: float,
    selected_audio_track: int | None,
    source_path: str,
    start_frame: int,
    merged_signatures: list[dict],
    verbose_level: int,
    ui_update,
    ui_skip,
):
    residual_continuation_paths = output_paths.list_continuation_output_paths(output_path)
    if not residual_continuation_paths:
        return ContinuationRecoveryResult(merged_signatures)

    merged_signatures, residual_continuation_paths, already_merged_paths = reconcile_frame_count_mismatch(
        ffprobe_path,
        output_path,
        residual_continuation_paths,
        fps_float=fps_float,
        merged_signatures=merged_signatures,
        verbose_level=verbose_level,
    )
    undeleted_already_merged_paths: list[str] = []
    for already_merged_path in already_merged_paths:
        try:
            os.remove(already_merged_path)
        except OSError:
            undeleted_already_merged_paths.append(already_merged_path)

    if not residual_continuation_paths:
        if undeleted_already_merged_paths:
            undeleted_names = ", ".join(Path(path).name for path in undeleted_already_merged_paths)
            common.plugin_info(f"Detected already-merged continuation file(s) still on disk, but they could not be deleted because they are still open: {undeleted_names}. Delete them manually when they are released.")
        return ContinuationRecoveryResult(merged_signatures)

    total_chunks_display = len(full_plans) if len(full_plans) > 0 else 1
    residual_names = ", ".join(Path(path).name for path in residual_continuation_paths)
    known_signature_keys = {process_metadata.continuation_signature_key(signature) for signature in merged_signatures}
    known_residual_paths = []
    for residual_path in residual_continuation_paths:
        residual_signature = process_metadata.make_continuation_signature(residual_path)
        if residual_signature is not None and process_metadata.continuation_signature_key(residual_signature) in known_signature_keys:
            known_residual_paths.append(residual_path)

    all_residual_paths_are_known = len(known_residual_paths) == len(residual_continuation_paths)
    if all_residual_paths_are_known:
        common.plugin_info(f"Found already-merged continuation file(s) still on disk: {residual_names}. Checking whether they can be deleted before continuing.")
        yield ui_update(status_ui.render_chunk_status_html(total_chunks_display, 0, 1, "Checking Continuations", f"Checking already-merged continuation file(s): {residual_names}"), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
    else:
        common.plugin_info(f"Found residual continuation file(s) from a previous unfinished merge: {residual_names}. Merging them into {Path(output_path).name} before continuing.")
        yield ui_update(status_ui.render_chunk_status_html(total_chunks_display, 0, 1, "Recovering Continuation", f"Recovering unfinished continuation merge: {residual_names}"), output_path if os.path.isfile(output_path) else ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)

    try:
        new_signatures, known_undeleted_paths, undeleted_newly_merged_paths = merge_residual_continuations(
            ffmpeg_path,
            ffprobe_path,
            output_path,
            residual_continuation_paths,
            video_codec=plugin.server_config.get("video_output_codec", "libx264_8"),
            video_container=output_container,
            audio_codec_key=plugin.server_config.get("audio_output_codec", "aac_128"),
            fps_float=fps_float,
            selected_audio_track_no=selected_audio_track,
            source_audio_path=source_path if selected_audio_track is not None else None,
            source_audio_start_seconds=(start_frame / fps_float) if selected_audio_track is not None else None,
            merged_continuation_signatures=merged_signatures,
        )
        for signature in new_signatures:
            merged_signatures = process_metadata.append_merged_continuation_signature(merged_signatures, signature)
        if new_signatures:
            recovered_frame_count, _ = media.probe_resume_frame_count(ffprobe_path, output_path, fps_float)
            process_metadata.store_process_progress(output_path, written_unique_frames=recovered_frame_count, merged_signatures=merged_signatures, verbose_level=verbose_level)
        undeleted_already_merged_paths.extend(known_undeleted_paths)
        if undeleted_already_merged_paths:
            undeleted_names = ", ".join(Path(path).name for path in undeleted_already_merged_paths)
            common.plugin_info(f"Detected already-merged continuation file(s) still on disk, but they could not be deleted because they are still open: {undeleted_names}. Delete them manually when they are released.")
        if undeleted_newly_merged_paths:
            undeleted_names = ", ".join(Path(path).name for path in undeleted_newly_merged_paths)
            common.plugin_info(f"Merged residual continuation file(s) into {Path(output_path).name}, but these continuation file(s) could not be deleted because they are still open: {undeleted_names}. Delete them manually when they are released.")
    except media.ContinuationMergeOutputLockedError:
        locked_message = f"{Path(output_path).name} is open, so the pending continuation merge could not replace it. Release the base file and start a process again."
        gr.Info(locked_message)
        yield ui_update(status_ui.render_chunk_status_html(total_chunks_display, 0, 1, "Merge Pending", locked_message), output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False)
        return ContinuationRecoveryResult(merged_signatures, blocked=True)
    except Exception as exc:
        raise gr.Error(f"Failed to merge the residual continuation file(s) before resuming. Please close any player using {output_path} and retry. {exc}") from exc

    return ContinuationRecoveryResult(merged_signatures)
