from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from shared.utils.hdr import VIDEO_PROMPT_HDR_OUTPUT_FLAG

from .chunk_executor import ChunkExecutor, ChunkProgress, ProcessContext
from . import common
from . import continuation_recovery
from . import frame_planning as frames
from . import media_io as media
from . import output_paths
from . import process_catalog as catalog
from . import process_metadata
from . import prompt_schedule as prompts
from . import status_ui
from . import video_buffers as video
from .mux_session import MuxSession
from .run_preparation import ProcessInfoExit, prepare_run


@dataclass(frozen=True)
class RunRequest:
    state: dict | None = None
    process_name: str = ""
    user_refs: list[str] = field(default_factory=list)
    source_path: str = ""
    process_strength: object = None
    output_path: str = ""
    prompt_text: str = ""
    continue_enabled: bool = True
    source_audio_track: str = ""
    output_resolution: str = "720p"
    target_ratio: str = ""
    chunk_size_seconds: object = 10.0
    sliding_window_overlap: object = 1
    start_seconds: str = ""
    end_seconds: str = ""

    @classmethod
    def from_gradio(
        cls,
        state=None,
        process_name="",
        user_refs=None,
        source_path="",
        process_strength=None,
        output_path="",
        prompt_text="",
        continue_enabled=True,
        source_audio_track="",
        output_resolution="720p",
        target_ratio="",
        chunk_size_seconds=10.0,
        sliding_window_overlap=1,
        start_seconds="",
        end_seconds="",
    ) -> "RunRequest":
        return cls(
            state=state,
            process_name=str(process_name or "").strip(),
            user_refs=list(user_refs or []),
            source_path=str(source_path or "").strip(),
            process_strength=process_strength,
            output_path=str(output_path or "").strip(),
            prompt_text=str(prompt_text or ""),
            continue_enabled=bool(continue_enabled),
            source_audio_track=str(source_audio_track or "").strip(),
            output_resolution=str(output_resolution or "720p").strip(),
            target_ratio=str(target_ratio or "").strip(),
            chunk_size_seconds=chunk_size_seconds,
            sliding_window_overlap=sliding_window_overlap,
            start_seconds="" if start_seconds in (None, "") else str(start_seconds),
            end_seconds="" if end_seconds in (None, "") else str(end_seconds),
        )


class ProcessRunner:
    def __init__(self, *, plugin, api_session, library, get_model_def, active_job: dict, preview_state: dict, ui_skip, ui_update, info_exit, reset_live_chunk_status) -> None:
        self.plugin = plugin
        self.api_session = api_session
        self.library = library
        self.get_model_def = get_model_def
        self.active_job = active_job
        self.preview_state = preview_state
        self.ui_skip = ui_skip
        self.ui_update = ui_update
        self.info_exit = info_exit
        self.reset_live_chunk_status = reset_live_chunk_status

    def start_process(self, state=None, process_name="", user_refs=None, source_path="", process_strength=None, output_path="", prompt_text="", continue_enabled=True, source_audio_track="", output_resolution="720p", target_ratio="", chunk_size_seconds=10.0, sliding_window_overlap=1, start_seconds="", end_seconds=""):
        if self.active_job.get("running"):
            yield self.info_exit("A process is already running.")
            return
        request = RunRequest.from_gradio(state, process_name, user_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds)
        process_definition = self.library.process_definition(request.process_name, request.state, request.user_refs)
        if process_definition is None:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            return
        if process_definition.get("source") == "user":
            problems = self.library.validate_user_process_definition(process_definition)
            if len(problems) > 0:
                yield self.info_exit(self.library.format_user_process_validation_error(process_definition, problems))
                return
        process_settings = process_definition["settings"]
        model_type = str(process_settings.get("model_type") or "")
        if len(model_type) == 0:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            return
        process_display_name = str(process_definition.get("name") or request.process_name or "").strip()
        process_is_hdr = VIDEO_PROMPT_HDR_OUTPUT_FLAG in str(process_settings.get("video_prompt_type") or "")
        is_user_process = process_definition.get("source") == "user"
        has_outpaint_setting = "video_guide_outpainting" in process_settings
        uses_builtin_outpaint_ui = self.library.uses_builtin_outpaint_ui(process_definition)
        user_lora_strength_override_default = self.library.user_lora_strength_override_default(process_definition)
        use_lora_strength_override = not uses_builtin_outpaint_ui and (not is_user_process or user_lora_strength_override_default is not None)
        process_strength_default = user_lora_strength_override_default if user_lora_strength_override_default is not None else common.get_default_process_strength(process_settings)
        active_process_strength = 1.0 if uses_builtin_outpaint_ui else (common.coerce_float(request.process_strength, process_strength_default) if use_lora_strength_override else process_strength_default)
        source_path = request.source_path
        output_path = request.output_path
        source_audio_track = request.source_audio_track
        output_resolution = request.output_resolution
        target_ratio = request.target_ratio
        prompt_text = request.prompt_text
        start_seconds = request.start_seconds
        end_seconds = request.end_seconds
        try:
            chunk_size_seconds = common.require_float(request.chunk_size_seconds, "Chunk Size", minimum=0.1)
            sliding_window_overlap = common.require_int(request.sliding_window_overlap, "Sliding Window Overlap", minimum=1)
        except gr.Error as exc:
            yield self.info_exit(common.get_error_message(exc) or "Invalid processing settings.")
            return
        try:
            catalog.save_process_full_video_ui_settings({
                "process_model_type": model_type,
                "process_name": request.process_name,
                "source_path": source_path,
                "process_strength": active_process_strength,
                "output_path": output_path,
                "prompt": prompt_text,
                "continue_enabled": request.continue_enabled,
                "source_audio_track": source_audio_track,
                "output_resolution": output_resolution,
                "target_ratio": target_ratio,
                "chunk_size_seconds": chunk_size_seconds,
                "sliding_window_overlap": sliding_window_overlap,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            })
        except OSError as exc:
            yield self.info_exit(f"Unable to save plugin settings to {catalog.PROCESS_FULL_VIDEO_SETTINGS_FILE}: {exc}")
            return
        active_target_ratio = target_ratio if uses_builtin_outpaint_ui else ""
        default_prompt_text = str(process_settings.get("prompt") or "")
        if len(prompt_text.strip()) == 0:
            prompt_text = default_prompt_text
        if not os.path.isfile(source_path):
            yield self.info_exit(f"Source video not found: {source_path}")
            return
        try:
            start_seconds = prompts.parse_time_input(start_seconds, label="Start", allow_empty=False)
            end_seconds = prompts.parse_time_input(end_seconds, label="End", allow_empty=True)
        except gr.Error as exc:
            yield self.info_exit(common.get_error_message(exc) or "Invalid start/end selection.")
            return
        try:
            prompt_schedule = prompts.parse_prompt_schedule(prompt_text)
        except gr.Error as exc:
            yield self.info_exit(common.get_error_message(exc) or f"Invalid prompt syntax.\n\nExample:\n{prompts.TIMED_PROMPT_EXAMPLE}")
            return
        started_ui = False
        preflight_stage = True
        write_state = MuxSession()
        total_chunks_display = 1
        completed_chunks = 0
        resumed_unique_frames = 0
        self.active_job["cancel_requested"] = False
        self.active_job["write_state"] = write_state
        self.active_job["running"] = True
        try:
            video.clear_process_full_video_source()
            yield self.ui_update(status_ui.render_chunk_status_html(1, 0, 1, "Initializing", "Preparing processing job..."), self.ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            started_ui = True
            try:
                prepared_run = prepare_run(
                    plugin=self.plugin,
                    get_model_def=self.get_model_def,
                    process_name=request.process_name,
                    process_display_name=process_display_name,
                    source_path=source_path,
                    output_path=output_path,
                    output_resolution=output_resolution,
                    active_target_ratio=active_target_ratio,
                    continue_enabled=request.continue_enabled,
                    source_audio_track=source_audio_track,
                    chunk_size_seconds=chunk_size_seconds,
                    sliding_window_overlap=sliding_window_overlap,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    model_type=model_type,
                    uses_builtin_outpaint_ui=uses_builtin_outpaint_ui,
                )
                verbose_level = prepared_run.verbose_level
                start_frame = prepared_run.start_frame
                end_frame_exclusive = prepared_run.end_frame_exclusive
                fps_float = prepared_run.fps_float
                total_source_frames = prepared_run.total_source_frames
                processing_fps = prepared_run.processing_fps
                selected_audio_track = prepared_run.selected_audio_track
                frame_plan_rules = prepared_run.frame_plan_rules
                budget_resolution = prepared_run.budget_resolution
                chunk_frames = prepared_run.chunk_frames
                overlap_frames = prepared_run.overlap_frames
                full_plans = prepared_run.full_plans
                requested_unique_frames = prepared_run.requested_unique_frames
                requested_source_segment = prepared_run.requested_source_segment
                output_path = prepared_run.output_path
                resume_existing_output = prepared_run.resume_existing_output
                ffmpeg_path = prepared_run.ffmpeg_path
                ffprobe_path = prepared_run.ffprobe_path
                output_container = prepared_run.output_container
                merged_continuation_signatures = prepared_run.merged_continuation_signatures
            except ProcessInfoExit as exc:
                yield self.info_exit(exc.message, output=exc.output_path or self.ui_skip)
                return
            except gr.Error as exc:
                yield self.info_exit(common.get_error_message(exc) or "Invalid process settings.", output=output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip)
                return
            preflight_stage = False
            if resume_existing_output:
                recovery_result = yield from continuation_recovery.recover_residual_continuations(
                    plugin=self.plugin,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    output_path=output_path,
                    full_plans=full_plans,
                    output_container=output_container,
                    fps_float=fps_float,
                    selected_audio_track=selected_audio_track,
                    source_path=source_path,
                    start_frame=start_frame,
                    merged_signatures=merged_continuation_signatures,
                    verbose_level=verbose_level,
                    ui_update=self.ui_update,
                    ui_skip=self.ui_skip,
                )
                merged_continuation_signatures = recovery_result.merged_signatures
                if recovery_result.blocked:
                    return
                if self.active_job.get("cancel_requested"):
                    write_state.stopped = True
                    yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans) if len(full_plans) > 0 else 1, 0, 1, "Stopped", "Stopped before processing a new chunk."), output_path if os.path.isfile(output_path) else self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
                    return
            last_frame_image = None
            last_segment_path = None
            continuation_output_path = ""
            chunk_output_paths: list[str] = []
            written_unique_frames = 0
            resumed_unique_frames = 0
            resume_overlap_frames = 0
            completed_chunks = 0
            resolved_resolution = ""
            resolved_width = 0
            resolved_height = 0
            merged_continuation = False
            resume_audio_trim_seconds = 0.0
            self.preview_state["image"] = None
            write_state.set_output_path(output_path)
            exact_start_seconds = start_frame / fps_float
            if resume_existing_output:
                yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans), 0, 1, "Inspecting Existing Output", f"Inspecting existing output to continue: {output_path}"), output_path, str(time.time_ns()))
                process_metadata.log_existing_output_metadata(output_path, verbose_level)
                resumed_unique_frames, resume_reason = media.probe_resume_frame_count(ffprobe_path, output_path, fps_float)
                recorded_written_unique_frames = process_metadata.read_recorded_written_unique_frames(output_path)
                if 0 < recorded_written_unique_frames < resumed_unique_frames:
                    common.plugin_info(f"Output contains {resumed_unique_frames} readable frame(s), but metadata recorded {recorded_written_unique_frames}. Using the real frame count for continuation.")
                    process_metadata.store_process_progress(output_path, written_unique_frames=resumed_unique_frames, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
                elif recorded_written_unique_frames > resumed_unique_frames:
                    common.plugin_info(f"Ignoring recorded output progress of {recorded_written_unique_frames} frame(s) because the output only probes as {resumed_unique_frames} frame(s).")
                if resumed_unique_frames < 0:
                    common.plugin_info(f"Ignoring negative probed output progress from {output_path}.")
                    resumed_unique_frames = 0
                elif resumed_unique_frames > requested_unique_frames:
                    common.plugin_info(f"Existing output already covers the requested {requested_unique_frames} frame(s).")
                    resumed_unique_frames = requested_unique_frames
                if resumed_unique_frames <= 0:
                    common.plugin_info(f"Unable to continue from existing output: {output_path}. {resume_reason or 'Starting a new file instead.'}")
                    output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                    write_state.set_output_path(output_path)
                    resumed_unique_frames = 0
                    resume_overlap_frames = 0
                    completed_chunks = 0
                    exact_start_seconds = start_frame / fps_float
                    resume_existing_output = False
                else:
                    common.plugin_info(f"Continuing existing output: {output_path}")
                    resolved_resolution, resolved_width, resolved_height = video.probe_existing_output_resolution(output_path)
                    print(f"[Process Full Video] Continuing with locked output resolution {resolved_resolution}")
                    completed_chunks, _ = frames.count_completed_chunks(full_plans, resumed_unique_frames)
                    exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                    if resumed_unique_frames < requested_unique_frames:
                        yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans), 0, 1, "Loading Overlap Frames", f"Continuing existing output with {resumed_unique_frames} frame(s) already written."), output_path, str(time.time_ns()))
                        resumed_unique_frames, last_frame_image, tail_reason = video.resolve_resume_last_frame(output_path, resumed_unique_frames)
                        if resumed_unique_frames <= 0 or last_frame_image is None:
                            common.plugin_info(f"Unable to continue from existing output: {output_path}. {tail_reason or 'Starting a new file instead.'}")
                            output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                            write_state.set_output_path(output_path)
                            resumed_unique_frames = 0
                            resume_overlap_frames = 0
                            completed_chunks = 0
                            self.preview_state["image"] = None
                            exact_start_seconds = start_frame / fps_float
                            resume_existing_output = False
                        else:
                            if tail_reason:
                                common.plugin_info(tail_reason)
                            self.preview_state["image"] = last_frame_image
                            completed_chunks, _ = frames.count_completed_chunks(full_plans, resumed_unique_frames)
                            exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                            resume_overlap_frames = overlap_frames
                            if resumed_unique_frames < resume_overlap_frames:
                                resume_overlap_frames = resumed_unique_frames
                            overlap_tensor = video.load_process_full_video_hdr_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames) if process_is_hdr else video.load_process_full_video_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames)
                            if overlap_tensor is None:
                                common.plugin_info(f"Unable to continue from existing output: {output_path}. Failed to load the overlap frames from the recorded output.")
                                output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                                write_state.set_output_path(output_path)
                                resumed_unique_frames = 0
                                resume_overlap_frames = 0
                                completed_chunks = 0
                                self.preview_state["image"] = None
                                exact_start_seconds = start_frame / fps_float
                                resume_existing_output = False
                            else:
                                resume_overlap_frames = int(overlap_tensor.shape[1])
                                video.set_process_full_video_overlap_buffer(overlap_tensor, processing_fps, hdr=process_is_hdr)
                                print(f"[Process Full Video] Loaded overlap buffer from existing output: {frames.describe_frame_range(start_frame + resumed_unique_frames - resume_overlap_frames, resume_overlap_frames)}")
                    if resume_existing_output and resumed_unique_frames < requested_unique_frames:
                        if not continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE:
                            resume_audio_trim_seconds = media.probe_selected_audio_overhang(ffprobe_path, output_path, selected_audio_track, resumed_unique_frames / fps_float)
                        if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None:
                            print(f"[Process Full Video] Final merge: rebuilding audio from source starting at {start_frame / fps_float:.6f}s")
                        elif resume_audio_trim_seconds > 0.0:
                            print(f"[Process Full Video] Final merge: trimming {resume_audio_trim_seconds:.6f}s from continuation audio and clamping segment audio to visible video duration")
                        remaining_resume_unique_frames = frames.align_total_unique_frames(
                            end_frame_exclusive - (start_frame + resumed_unique_frames),
                            frame_step=frame_plan_rules.frame_step,
                            minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                            initial_overlap_frames=resume_overlap_frames,
                        )
                        if remaining_resume_unique_frames <= 0:
                            trailing_frames = requested_unique_frames - resumed_unique_frames
                            common.plugin_info(f"Existing output has {resumed_unique_frames} frame(s). The remaining {trailing_frames} frame(s) are too short to build another continuation chunk for the current model, so the existing output is treated as complete.")
                            resumed_unique_frames = requested_unique_frames
                            completed_chunks = len(full_plans)
                            plans = []
                        else:
                            continuation_output_path = output_paths.make_continuation_output_path(output_path)
                            write_state.set_output_path(continuation_output_path)
                            try:
                                plans = frames.build_chunk_plan(
                                    start_frame + resumed_unique_frames,
                                    end_frame_exclusive,
                                    total_source_frames,
                                    chunk_frames,
                                    frame_step=frame_plan_rules.frame_step,
                                    minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                                    overlap_frames=overlap_frames,
                                    initial_overlap_frames=resume_overlap_frames,
                                )
                            except frames.FramePlanningError as exc:
                                raise gr.Error(str(exc)) from exc
                    elif resume_existing_output:
                        plans = []
            if not resume_existing_output:
                plans = full_plans
            continued_mode = resumed_unique_frames > 0
            use_live_av_mux = selected_audio_track is not None
            total_chunks_display = completed_chunks + len(plans)
            current_chunk_display = completed_chunks + 1
            run_started_at = time.time()
            initial_completed_chunks = completed_chunks

            def _timing_kwargs(current_completed_chunks=None, phase_current_step=None, phase_total_steps=None):
                elapsed_seconds = time.time() - run_started_at
                if elapsed_seconds < 0.0:
                    elapsed_seconds = 0.0
                if len(plans) == 0:
                    return {"elapsed_seconds": elapsed_seconds, "eta_seconds": None}
                completed_for_eta = completed_chunks if current_completed_chunks is None else current_completed_chunks
                run_completed_chunks = completed_for_eta - initial_completed_chunks
                phase_ratio = 0.0
                if phase_current_step is not None and phase_total_steps is not None and phase_total_steps > 0:
                    phase_ratio = float(phase_current_step) / float(phase_total_steps)
                overall_ratio = (run_completed_chunks + phase_ratio) / float(len(plans))
                eta_seconds = None if overall_ratio <= 0.0 or overall_ratio >= 1.0 else elapsed_seconds * (1.0 - overall_ratio) / overall_ratio
                return {"elapsed_seconds": elapsed_seconds, "eta_seconds": eta_seconds}

            if len(plans) == 0:
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, completed_chunks, "Completed", "Existing output already covers the requested range.", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                return
            planning_text = f"Resuming from {resumed_unique_frames} frame(s) already written." if resumed_unique_frames > 0 else f"Preparing {len(plans)} chunk(s)..."
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Planning", planning_text, continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()))

            chunk_result = yield from ChunkExecutor(
                plugin=self.plugin,
                api_session=self.api_session,
                active_job=self.active_job,
                preview_state=self.preview_state,
                ui_update=self.ui_update,
                ui_skip=self.ui_skip,
                reset_live_chunk_status=self.reset_live_chunk_status,
            ).run(ProcessContext(
                state=request.state,
                process_settings=process_definition["settings"],
                model_type=model_type,
                process_is_hdr=process_is_hdr,
                is_user_process=is_user_process,
                has_outpaint_setting=has_outpaint_setting,
                uses_builtin_outpaint_ui=uses_builtin_outpaint_ui,
                use_lora_strength_override=use_lora_strength_override,
                active_process_strength=active_process_strength,
                active_target_ratio=active_target_ratio,
                source_path=source_path,
                selected_audio_track=selected_audio_track,
                prompt_schedule=prompt_schedule,
                default_prompt_text=default_prompt_text,
                budget_resolution=budget_resolution,
                start_frame=start_frame,
                resumed_unique_frames=resumed_unique_frames,
                requested_unique_frames=requested_unique_frames,
                overlap_frames=overlap_frames,
                processing_fps=processing_fps,
                fps_float=fps_float,
                continued_mode=continued_mode,
                plans=plans,
                total_chunks_display=total_chunks_display,
                ffmpeg_path=ffmpeg_path,
                use_live_av_mux=use_live_av_mux,
                output_container=output_container,
                exact_start_seconds=exact_start_seconds,
                timing_kwargs=_timing_kwargs,
            ), ChunkProgress(
                completed_chunks=completed_chunks,
                current_chunk_display=current_chunk_display,
                chunk_output_paths=chunk_output_paths,
                last_segment_path=last_segment_path,
                write_state=write_state,
                resolved_resolution=resolved_resolution,
                resolved_width=resolved_width,
                resolved_height=resolved_height,
            ))
            written_unique_frames = chunk_result.written_unique_frames
            completed_chunks = chunk_result.completed_chunks
            current_chunk_display = chunk_result.current_chunk_display
            resolved_resolution = chunk_result.resolved_resolution
            resolved_width = chunk_result.resolved_width
            resolved_height = chunk_result.resolved_height
            last_segment_path = chunk_result.last_segment_path
            chunk_output_paths = chunk_result.chunk_output_paths
            if write_state.mux_process is None:
                if write_state.stopped and resumed_unique_frames > 0:
                    common.plugin_info(f"Processing was stopped before writing a new chunk. Kept existing output at {output_path}")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", "Stopped before a new chunk was written. Existing output kept.", continued=continued_mode, **_timing_kwargs()), output_path, self.ui_skip, start_enabled=True, abort_enabled=False)
                    return
                if write_state.stopped:
                    common.plugin_info("Processing was stopped before any output chunk was written.")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", "Stopped before any output chunk was written.", continued=continued_mode, **_timing_kwargs()), self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
                    return
                raise gr.Error("Processing completed without creating an output file.")
            if self.active_job.get("cancel_requested"):
                write_state.stopped = True
            finalizing_message = "Finalizing written output before merge..." if continuation_output_path and os.path.isfile(write_state.output_path_for_write) else "Finalizing written output..."
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Finalizing Output", finalizing_message, continued=continued_mode, **_timing_kwargs()), output_path if os.path.isfile(output_path) else self.ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            return_code, stderr, forced_termination = write_state.finalize()
            if self.active_job.get("cancel_requested"):
                write_state.stopped = True
            if forced_termination:
                raise gr.Error("ffmpeg did not finalize the partial output in time.")
            if return_code != 0 and not (write_state.stopped and os.path.isfile(write_state.output_path_for_write if use_live_av_mux else write_state.video_only_output_path)):
                raise gr.Error(stderr or "ffmpeg failed while assembling the processed video.")
            if use_live_av_mux and os.path.isfile(write_state.output_path_for_write) and os.path.getsize(write_state.output_path_for_write) <= 0:
                media.delete_file_if_exists(write_state.output_path_for_write, label="continuation output")
                raise gr.Error("ffmpeg created an empty continuation file.")
            if not use_live_av_mux and os.path.isfile(write_state.video_only_output_path):
                try:
                    os.replace(write_state.video_only_output_path, write_state.output_path_for_write)
                except OSError as exc:
                    raise gr.Error(f"Unable to finalize the written video-only segment: {write_state.output_path_for_write}") from exc
            undeleted_merged_continuation_paths: list[str] = []
            if continuation_output_path and os.path.isfile(write_state.output_path_for_write):
                continuation_signature = process_metadata.make_continuation_signature(write_state.output_path_for_write)
                try:
                    existing_output_generation_time = process_metadata.read_metadata_generation_time(output_path)
                    merged_duration_seconds = float(resumed_unique_frames + written_unique_frames) / fps_float
                    committed_signatures = process_metadata.append_merged_continuation_signature(merged_continuation_signatures, continuation_signature)
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Merging Continuation", "Merging the continued segment into the main output...", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                    media.concat_video_segments(
                        ffmpeg_path,
                        [output_path, write_state.output_path_for_write],
                        output_path,
                        self.plugin.server_config.get("video_output_codec", "libx264_8"),
                        output_container,
                        self.plugin.server_config.get("audio_output_codec", "aac_128"),
                        segment_audio_trim_seconds=[0.0, resume_audio_trim_seconds],
                        segment_audio_duration_seconds=[(float(resumed_unique_frames) / fps_float) if resumed_unique_frames > 0 else None, (float(written_unique_frames) / fps_float) if written_unique_frames > 0 else None],
                        fps_float=fps_float,
                        selected_audio_track_no=selected_audio_track,
                        reserved_metadata_path=write_state.reserved_metadata_path,
                        source_audio_path=source_path if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_start_seconds=(start_frame / fps_float) if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_duration_seconds=merged_duration_seconds if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_track_no=selected_audio_track if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                    )
                    merged_continuation = True
                    merged_continuation_signatures = committed_signatures
                    process_metadata.store_process_progress(output_path, written_unique_frames=resumed_unique_frames + written_unique_frames, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
                except media.ContinuationMergeOutputLockedError:
                    locked_message = f"{Path(output_path).name} is open, so the continuation merge could not replace it. Existing output was kept and {Path(write_state.output_path_for_write).name} was preserved. Release the base file and start a process again."
                    gr.Info(locked_message)
                    media.delete_file_if_exists(write_state.reserved_metadata_path, label="reserved metadata file")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Merge Pending", locked_message, continued=continued_mode, **_timing_kwargs()), write_state.output_path_for_write, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                    return
                except Exception as exc:
                    raise gr.Error(f"Failed to finalize continued output. Existing output kept, and continuation was preserved at {continuation_output_path}. {exc}") from exc
                if os.path.isfile(write_state.output_path_for_write):
                    try:
                        os.remove(write_state.output_path_for_write)
                    except OSError:
                        undeleted_merged_continuation_paths.append(write_state.output_path_for_write)
                        common.plugin_info(f"Merged continuation progress into {Path(output_path).name}, but {Path(write_state.output_path_for_write).name} could not be deleted because it is still open. Delete it manually when released.")
            else:
                existing_output_generation_time = 0.0
            media.delete_file_if_exists(write_state.reserved_metadata_path, label="reserved metadata file")
            total_written_unique_frames = resumed_unique_frames + written_unique_frames
            if not write_state.stopped and total_written_unique_frames < requested_unique_frames:
                raise gr.Error(f"Processing wrote {total_written_unique_frames} frame(s), but {requested_unique_frames} frame(s) were required.")
            metadata_target_path = output_path if merged_continuation or not continuation_output_path else write_state.output_path_for_write
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Writing Metadata", "Writing final output metadata...", continued=continued_mode, **_timing_kwargs()), metadata_target_path if os.path.isfile(metadata_target_path) else output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            metadata_source_path = last_segment_path or media.get_last_generated_video_path(chunk_output_paths)
            actual_output_frames = media.probe_resume_frame_count(ffprobe_path, metadata_target_path, fps_float)[0] if os.path.isfile(metadata_target_path) else 0
            if actual_output_frames <= 0:
                if metadata_target_path != output_path or not resume_existing_output:
                    media.delete_file_if_exists(metadata_target_path, label="invalid output")
                raise gr.Error("Final output does not contain a readable video frame.")
            if actual_output_frames < total_written_unique_frames:
                if write_state.stopped:
                    common.plugin_info(f"Stopped output contains {actual_output_frames} readable frame(s), lower than the {total_written_unique_frames} frame(s) attempted. Recording the probed frame count.")
                else:
                    common.plugin_info(f"Final output contains {actual_output_frames} readable frame(s), lower than the {total_written_unique_frames} frame(s) written. Recording the probed frame count.")
                total_written_unique_frames = actual_output_frames
            total_generation_time = existing_output_generation_time + process_metadata.read_metadata_generation_time(metadata_source_path) if merged_continuation else process_metadata.read_metadata_generation_time(metadata_source_path)
            output_process_metadata = {
                "process": process_display_name,
                "written_unique_frames": int(total_written_unique_frames),
                "chunks": int(total_chunks_display),
                "sliding_window_overlap": int(overlap_frames),
                "start_seconds": float(start_seconds),
                "end_seconds": float(start_seconds + (total_written_unique_frames / float(fps_float))),
                "source_video": source_path,
                "source_segment": requested_source_segment,
                "merged_continuations": process_metadata.normalize_merged_continuation_signatures(merged_continuation_signatures),
            }
            if process_is_hdr:
                output_process_metadata["hdr"] = True
            process_metadata.store_output_metadata(metadata_target_path, metadata_source_path, source_path=source_path, process_name=process_display_name, source_start_seconds=start_seconds, start_frame=start_frame, fps_float=fps_float, selected_audio_track=selected_audio_track, total_generation_time=total_generation_time, actual_frame_count=actual_output_frames, process_metadata=output_process_metadata, verbose_level=verbose_level)
            chunk_output_paths = media.delete_released_chunk_outputs(request.state, chunk_output_paths)
            if write_state.stopped:
                stopped_output_path = output_path
                if merged_continuation:
                    common.plugin_info(f"Processing was stopped. Merged continued progress into {output_path}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Continued progress was merged into the output."
                elif continuation_output_path and os.path.isfile(write_state.output_path_for_write):
                    stopped_output_path = write_state.output_path_for_write
                    common.plugin_info(f"Processing was stopped. Kept existing output at {output_path} and preserved continuation clip at {write_state.output_path_for_write}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Existing output kept and continuation clip preserved."
                else:
                    common.plugin_info(f"Processing was stopped. Kept partial output at {output_path}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Partial output kept."
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", stop_message, continued=continued_mode, **_timing_kwargs()), stopped_output_path, self.ui_skip, start_enabled=True, abort_enabled=False)
                return
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, total_chunks_display, total_chunks_display, "Completed", f"Completed {total_chunks_display} chunk(s).", continued=continued_mode, **_timing_kwargs()), output_path, self.ui_skip, start_enabled=True, abort_enabled=False)
            self.active_job["job"] = None
            write_state.cleanup_partial_outputs()
        except gr.Error as exc:
            self.active_job["job"] = None
            write_state.cleanup_partial_outputs()
            status_message = common.get_error_message(exc) or "Processing failed."
            if not started_ui:
                gr.Info(status_message)
                return
            if started_ui:
                total_chunks_value = total_chunks_display
                completed_value = completed_chunks
                current_value = completed_chunks + 1 if completed_chunks < total_chunks_display else total_chunks_display
                continued_value = resumed_unique_frames > 0
                output_value = output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip
                if preflight_stage:
                    gr.Info(status_message)
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_value, completed_value, current_value, "Info" if preflight_stage else "Error", status_message, continued=continued_value), output_value, self.ui_skip, start_enabled=True, abort_enabled=False)
            return
        except BaseException as exc:
            self.active_job["job"] = None
            write_state.cleanup_partial_outputs()
            if started_ui:
                total_chunks_value = total_chunks_display
                completed_value = completed_chunks
                current_value = completed_chunks + 1 if completed_chunks < total_chunks_display else total_chunks_display
                continued_value = resumed_unique_frames > 0
                status_message = common.get_error_message(exc) or exc.__class__.__name__
                output_value = output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_value, completed_value, current_value, "Error", status_message, continued=continued_value), output_value, self.ui_skip, start_enabled=True, abort_enabled=False)
            raise
        finally:
            self.active_job["running"] = False
            self.active_job["write_state"] = None
            video.clear_process_full_video_source()

