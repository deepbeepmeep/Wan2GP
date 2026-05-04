from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gradio as gr
import torch

from shared.utils.hdr import tonemap_hdr_tensor_to_uint8
from shared.utils.virtual_media import build_virtual_media_path

from . import constants as ui_constants
from . import frame_planning as frames
from . import media_io as media
from . import prompt_schedule as prompts
from . import status_ui
from . import video_buffers as video
from .mux_session import MuxSession


USER_PROCESS_OUTPUT_KEYS = {
    "output_path",
    "output_dir",
    "save_path",
    "image_save_path",
    "audio_save_path",
}


def build_task_settings(process_settings: dict, *, is_user_process: bool) -> dict:
    settings = copy.deepcopy(process_settings)
    if is_user_process:
        for key in USER_PROCESS_OUTPUT_KEYS:
            settings.pop(key, None)
        settings["repeat_generation"] = 1
        settings["batch_size"] = 1
    api_settings = settings.get("_api")
    settings["_api"] = dict(api_settings) if isinstance(api_settings, dict) else {}
    if is_user_process:
        settings["_api"] = {key: value for key, value in settings["_api"].items() if key == "return_media"}
    settings["_api"]["return_media"] = True
    return settings


@dataclass(frozen=True)
class ProcessContext:
    state: dict | None
    process_settings: dict
    model_type: str
    process_is_hdr: bool
    is_user_process: bool
    has_outpaint_setting: bool
    uses_builtin_outpaint_ui: bool
    use_lora_strength_override: bool
    active_process_strength: float
    active_target_ratio: str
    source_path: str
    selected_audio_track: int | None
    prompt_schedule: list[tuple[float, str]]
    default_prompt_text: str
    budget_resolution: str
    start_frame: int
    resumed_unique_frames: int
    requested_unique_frames: int
    overlap_frames: int
    processing_fps: float
    fps_float: float
    continued_mode: bool
    plans: list[frames.ChunkPlan]
    total_chunks_display: int
    ffmpeg_path: str
    use_live_av_mux: bool
    output_container: str
    exact_start_seconds: float
    timing_kwargs: Callable


@dataclass
class ChunkProgress:
    completed_chunks: int
    current_chunk_display: int
    chunk_output_paths: list[str]
    last_segment_path: str | None
    write_state: MuxSession
    written_unique_frames: int = 0
    resolved_resolution: str = ""
    resolved_width: int = 0
    resolved_height: int = 0


@dataclass(frozen=True)
class ChunkExecutionResult:
    written_unique_frames: int
    completed_chunks: int
    current_chunk_display: int
    resolved_resolution: str
    resolved_width: int
    resolved_height: int
    last_segment_path: str | None
    chunk_output_paths: list[str]


class ChunkExecutor:
    def __init__(self, *, plugin, api_session, active_job: dict, preview_state: dict, ui_update, ui_skip, reset_live_chunk_status) -> None:
        self.plugin = plugin
        self.api_session = api_session
        self.active_job = active_job
        self.preview_state = preview_state
        self.ui_update = ui_update
        self.ui_skip = ui_skip
        self.reset_live_chunk_status = reset_live_chunk_status

    def run(self, context: ProcessContext, progress: ChunkProgress):
        for chunk_index, plan in enumerate(context.plans, start=1):
            callbacks = status_ui.ChunkCallbacks()
            last_html = ""
            actual_done = context.resumed_unique_frames + progress.written_unique_frames
            plan_overlap_frames = plan.overlap_frames
            plan_requested_frames = plan.requested_frames
            actual_control_start_frame = context.start_frame + actual_done - plan_overlap_frames
            actual_control_end_frame = actual_control_start_frame + plan_requested_frames - 1
            overlap_buffer_start_frame = context.start_frame + actual_done - plan_overlap_frames
            model_video_length = plan_requested_frames if plan_overlap_frames <= 0 else plan_requested_frames - plan_overlap_frames + 1
            needs_video_source = context.continued_mode or plan_overlap_frames > 0
            print(
                f"[Process Full Video] Chunk {chunk_index}: control video {frames.describe_frame_range(actual_control_start_frame, plan_requested_frames)}; "
                + (f"overlap buffer {frames.describe_frame_range(overlap_buffer_start_frame, plan_overlap_frames)}" if needs_video_source else "overlap buffer not used")
            )
            settings = build_task_settings(context.process_settings, is_user_process=context.is_user_process)
            chunk_prompt_start_seconds = float(actual_done) / float(context.fps_float)
            settings["model_type"] = context.model_type
            settings["prompt"] = prompts.resolve_prompt_for_chunk(context.prompt_schedule, chunk_prompt_start_seconds, context.default_prompt_text)
            settings["resolution"] = progress.resolved_resolution or context.budget_resolution
            settings["video_length"] = model_video_length
            settings["sliding_window_overlap"] = plan_overlap_frames if plan_overlap_frames > 0 else 1
            settings["image_prompt_type"] = "V" if needs_video_source else ""
            settings["audio_prompt_type"] = "K"
            if context.is_user_process:
                settings["force_fps"] = "control"
            settings["video_guide"] = build_virtual_media_path(context.source_path, start_frame=actual_control_start_frame, end_frame=actual_control_end_frame, audio_track_no=context.selected_audio_track)
            if context.uses_builtin_outpaint_ui:
                settings["video_guide_outpainting_ratio"] = context.active_target_ratio
            elif not context.has_outpaint_setting:
                settings.pop("video_guide_outpainting_ratio", None)
            if context.use_lora_strength_override:
                settings["loras_multipliers"] = str(context.active_process_strength)
            if needs_video_source:
                settings["video_source"] = video.build_process_full_video_source_path(hdr=context.process_is_hdr)
            else:
                settings.pop("video_source", None)

            self.reset_live_chunk_status(context.state)
            job = self.api_session.submit_task(settings, callbacks=callbacks)
            self.active_job["job"] = job
            yield self.ui_update(start_enabled=False, abort_enabled=True)
            next_status_refresh_at = 0.0
            stop_requested = False
            while not job.done:
                if self.active_job.get("cancel_requested") and not stop_requested:
                    job.cancel()
                    progress.write_state.stopped = True
                    stop_requested = True
                now = time.monotonic()
                if now >= next_status_refresh_at:
                    html_value = status_ui.render_chunk_status_html(
                        context.total_chunks_display,
                        progress.completed_chunks,
                        progress.current_chunk_display,
                        callbacks.phase_label,
                        callbacks.status_text,
                        continued=context.continued_mode,
                        phase_current_step=callbacks.current_step,
                        phase_total_steps=callbacks.total_steps,
                        prefer_status_phase=True,
                        **context.timing_kwargs(progress.completed_chunks, callbacks.current_step, callbacks.total_steps),
                    )
                    next_status_refresh_at = now + ui_constants.STATUS_REFRESH_INTERVAL_SECONDS
                    if html_value != last_html:
                        last_html = html_value
                        yield self.ui_update(html_value)
                time.sleep(0.1)
            try:
                result = job.result()
            finally:
                if self.active_job.get("job") is job:
                    self.active_job["job"] = None
            yield self.ui_update(start_enabled=False, abort_enabled=False)
            if not result.success:
                if result.cancelled:
                    progress.write_state.stopped = True
                    break
                errors = list(result.errors or [])
                raise gr.Error(str(errors[0] if errors else f"Chunk {chunk_index} failed."))

            progress.chunk_output_paths.extend(
                str(Path(path).resolve())
                for path in result.generated_files
                if isinstance(path, str) and len(path.strip()) > 0 and str(Path(path).resolve()) not in progress.chunk_output_paths
            )
            progress.last_segment_path = media.get_last_generated_video_path(list(result.generated_files)) or progress.last_segment_path
            returned_video_item = next((item for item in result.artifacts if item.video_tensor_hdr is not None), None) if context.process_is_hdr else next((item for item in result.artifacts if item.video_tensor_uint8 is not None), None)
            returned_tensor = None if returned_video_item is None else (returned_video_item.video_tensor_hdr if context.process_is_hdr else returned_video_item.video_tensor_uint8)
            if returned_video_item is None or not torch.is_tensor(returned_tensor):
                raise gr.Error(f"Chunk {chunk_index} completed without returned video tensor data.")
            video_tensor_hdr = returned_tensor.detach().cpu() if context.process_is_hdr else None
            video_tensor_uint8 = tonemap_hdr_tensor_to_uint8(video_tensor_hdr) if context.process_is_hdr else returned_tensor.detach().cpu()
            returned_frame_count = int(video_tensor_uint8.shape[1])
            expected_frame_count = plan_requested_frames
            minimum_returned_frames = expected_frame_count - 1 if expected_frame_count > 1 else 1
            if not context.process_is_hdr and returned_frame_count < minimum_returned_frames:
                video_candidates = [path for path in result.generated_files if isinstance(path, str) and os.path.isfile(path) and str(Path(path).suffix).lower() in {".mp4", ".mkv", ".mov", ".avi"}]
                if video_candidates:
                    decoded_tensor = video.load_video_tensor_from_file(video_candidates[0])
                    decoded_frame_count = int(decoded_tensor.shape[1])
                    print(f"[Process Full Video] Chunk {chunk_index}: returned video tensor has {returned_frame_count} frame(s); decoded chunk file has {decoded_frame_count} frame(s)")
                    if decoded_frame_count >= minimum_returned_frames:
                        video_tensor_uint8 = decoded_tensor
                        returned_frame_count = decoded_frame_count
            print(f"[Process Full Video] Chunk {chunk_index}: returned video tensor has {returned_frame_count} frame(s); control video lasts {expected_frame_count} frame(s)")
            chunk_width, chunk_height = video.get_video_tensor_resolution(video_tensor_uint8)
            chunk_resolution = f"{chunk_width}x{chunk_height}"
            print(f"[Process Full Video] Chunk {chunk_index}: generated chunk resolution {chunk_resolution}")
            if len(progress.resolved_resolution) == 0:
                progress.resolved_resolution = chunk_resolution
                progress.resolved_width = chunk_width
                progress.resolved_height = chunk_height
            elif chunk_resolution != progress.resolved_resolution:
                raise gr.Error(f"Chunk {chunk_index} changed output resolution from {progress.resolved_resolution} to {chunk_resolution}.")

            skip_frames = plan_overlap_frames
            remaining_unique_frames = context.requested_unique_frames - (context.resumed_unique_frames + progress.written_unique_frames)
            expected_unique_frames = plan_requested_frames - skip_frames
            if expected_unique_frames <= 0:
                raise gr.Error(f"Chunk {chunk_index} has no writable frame in the computed plan.")
            if expected_unique_frames > remaining_unique_frames:
                raise gr.Error(f"Chunk {chunk_index} would write {expected_unique_frames} frame(s), but only {remaining_unique_frames} frame(s) remain.")
            writable_frame_count = int(video_tensor_uint8.shape[1]) - skip_frames
            if writable_frame_count < expected_unique_frames:
                raise gr.Error(f"Chunk {chunk_index} returned {writable_frame_count} writable frame(s), but {expected_unique_frames} frame(s) were required.")
            frames_to_write = expected_unique_frames

            progress.write_state.ensure_started(
                server_config=self.plugin.server_config,
                ffmpeg_path=context.ffmpeg_path,
                process_is_hdr=context.process_is_hdr,
                use_live_av_mux=context.use_live_av_mux,
                output_container=context.output_container,
                source_path=context.source_path,
                exact_start_seconds=context.exact_start_seconds,
                selected_audio_track=context.selected_audio_track,
                resolved_width=progress.resolved_width,
                resolved_height=progress.resolved_height,
                fps_float=context.fps_float,
            )
            last_frame_tensor = progress.write_state.write_chunk(process_is_hdr=context.process_is_hdr, video_tensor_hdr=video_tensor_hdr, video_tensor_uint8=video_tensor_uint8, start_frame=skip_frames, frame_count=frames_to_write)
            progress.written_unique_frames += frames_to_write
            self.preview_state["image"] = video.frame_to_image(last_frame_tensor)
            overlap_source_tensor = video_tensor_hdr if context.process_is_hdr and video_tensor_hdr is not None else video_tensor_uint8
            next_overlap_tensor = video.update_process_full_video_overlap_buffer(overlap_source_tensor[:, skip_frames:skip_frames + frames_to_write], context.overlap_frames, context.processing_fps, hdr=context.process_is_hdr)
            if next_overlap_tensor is not None and int(next_overlap_tensor.shape[1]) > 0:
                next_overlap_count = int(next_overlap_tensor.shape[1])
                next_overlap_start_frame = context.start_frame + context.resumed_unique_frames + progress.written_unique_frames - next_overlap_count
                print(f"[Process Full Video] Chunk {chunk_index}: next overlap buffer {frames.describe_frame_range(next_overlap_start_frame, next_overlap_count)}")

            progress.completed_chunks += 1
            progress.chunk_output_paths = media.delete_released_chunk_outputs(context.state, progress.chunk_output_paths, preserve_paths=[progress.last_segment_path] if progress.last_segment_path else None)
            if chunk_index < len(context.plans):
                progress.current_chunk_display = progress.completed_chunks + 1
                yield self.ui_update(status_ui.render_chunk_status_html(context.total_chunks_display, progress.completed_chunks, progress.current_chunk_display, "Starting new Chunk", f"Chunk {progress.completed_chunks} finished with {frames_to_write} written frame(s). Preparing next chunk...", continued=context.continued_mode, **context.timing_kwargs(progress.completed_chunks)), self.ui_skip, str(time.time_ns()))
            else:
                progress.current_chunk_display = progress.completed_chunks
                yield self.ui_update(status_ui.render_chunk_status_html(context.total_chunks_display, progress.completed_chunks, progress.current_chunk_display, "Chunk Completed", f"Chunk {progress.completed_chunks} finished with {frames_to_write} written frame(s).", continued=context.continued_mode, **context.timing_kwargs(progress.completed_chunks)), self.ui_skip, str(time.time_ns()))

        return ChunkExecutionResult(
            written_unique_frames=progress.written_unique_frames,
            completed_chunks=progress.completed_chunks,
            current_chunk_display=progress.current_chunk_display,
            resolved_resolution=progress.resolved_resolution,
            resolved_width=progress.resolved_width,
            resolved_height=progress.resolved_height,
            last_segment_path=progress.last_segment_path,
            chunk_output_paths=progress.chunk_output_paths,
        )
