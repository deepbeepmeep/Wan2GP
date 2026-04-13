import contextlib
import inspect
import sys
import threading
import time
from typing import Any

from shared.api import GenerationError, GenerationResult, SessionJob, _GENERATION_LOCK, _OutputCapture, _pushd
from shared.utils.thread_utils import AsyncStream


def run_cli_job(session, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
    stream = AsyncStream()
    gen = session._state["gen"]
    worker_done = threading.Event()
    base_file_count = len(gen["file_list"])
    base_audio_count = len(gen["audio_file_list"])
    total_tasks = len(tasks)
    runtime = None
    task_summary: dict[str, Any] = {
        "errors": [],
        "successful_tasks": 0,
        "failed_tasks": 0,
        "total_tasks": total_tasks,
    }

    try:
        runtime = session._ensure_runtime()
        with _GENERATION_LOCK, _pushd(runtime.root):
            session._configure_runtime(runtime)
            session._prepare_state_for_run(tasks)
            job.events.put("started", {"tasks": len(tasks)})

            def worker() -> None:
                stdout_capture = _OutputCapture(
                    "stdout",
                    lambda stream_name, line: session._emit_stream(job, stream_name, line),
                    console=sys.__stdout__ if session._console_output else None,
                    console_isatty=session._console_isatty,
                )
                stderr_capture = _OutputCapture(
                    "stderr",
                    lambda stream_name, line: session._emit_stream(job, stream_name, line),
                    console=sys.__stderr__ if session._console_output else None,
                    console_isatty=session._console_isatty,
                )
                try:
                    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                        _run_tasks_worker(session, runtime.module, tasks, stream, job, task_summary)
                except BaseException as exc:
                    failure = session._make_generation_error(exc, task_index=None, task_id=None, stage="runtime")
                    task_summary["errors"].append(failure)
                    stream.output_queue.push("error", failure)
                finally:
                    stdout_capture.flush()
                    stderr_capture.flush()
                    stream.output_queue.push("worker_exit", None)
                    worker_done.set()

            worker_thread = threading.Thread(target=worker, daemon=True, name="wangp-session-worker")
            worker_thread.start()

            while True:
                if job.cancel_requested:
                    session._request_cancel_unlocked(runtime.module)
                item = stream.output_queue.pop()
                if item is None:
                    if worker_done.is_set() and not worker_thread.is_alive():
                        break
                    time.sleep(0.01)
                    continue
                command, data = item
                if command == "worker_exit":
                    break
                _handle_command(session, job, runtime.module, tasks, command, data)

            worker_thread.join(timeout=0.1)
            outputs = session._collect_outputs(base_file_count, base_audio_count)
            artifacts = session._consume_output_artifacts(tasks)
            if job.cancel_requested and not task_summary["errors"]:
                task_summary["errors"].append(GenerationError(message="Generation was cancelled", stage="cancelled"))
                task_summary["failed_tasks"] = max(task_summary["failed_tasks"], 1)
            result = GenerationResult(
                success=not task_summary["errors"],
                generated_files=outputs,
                errors=list(task_summary["errors"]),
                total_tasks=task_summary["total_tasks"],
                successful_tasks=task_summary["successful_tasks"],
                failed_tasks=task_summary["failed_tasks"],
                artifacts=artifacts,
            )
            job.events.put("completed", result)
            session._emit_callback("on_complete", result, job=job)
            job._set_result(result)
    except BaseException as exc:
        failure = session._make_generation_error(exc, task_index=None, task_id=None, stage="runtime")
        result = GenerationResult(
            success=False,
            generated_files=[],
            errors=[failure],
            total_tasks=total_tasks,
            successful_tasks=task_summary["successful_tasks"],
            failed_tasks=max(task_summary["failed_tasks"], 1 if total_tasks > 0 else 0),
            artifacts=(),
        )
        job.events.put("error", failure)
        session._emit_callback("on_error", failure, job=job)
        job.events.put("completed", result)
        session._emit_callback("on_complete", result, job=job)
        job._set_result(result)
    finally:
        job.events.close()
        if runtime is not None:
            session._reset_state_after_run()
        with session._job_lock:
            if session._active_job is job:
                session._active_job = None


def _run_tasks_worker(session, wgp, tasks: list[dict[str, Any]], stream: AsyncStream, job: SessionJob, task_summary: dict[str, Any]) -> None:
    expected_args = set(inspect.signature(wgp.generate_video).parameters.keys())
    total_tasks = len(tasks)

    for task_index, task in enumerate(tasks, start=1):
        if job.cancel_requested:
            break
        session._state["gen"]["prompt_no"] = task_index
        session._state["gen"]["prompts_max"] = total_tasks
        session._state["gen"]["queue"] = tasks
        task_id = task.get("id")
        task_errors: list[GenerationError] = []

        def send_cmd(command: str, data: Any = None) -> None:
            if command == "error":
                failure = session._make_generation_error(data, task_index=task_index, task_id=task_id, stage="generation")
                task_errors.append(failure)
                stream.output_queue.push("error", failure)
                return
            stream.output_queue.push(command, data)

        validated_settings, validation_error = wgp.validate_task(task, session._state)
        if validated_settings is None:
            failure = GenerationError(
                message=validation_error or f"Task {task_index} failed validation",
                task_index=task_index,
                task_id=task_id,
                stage="validation",
            )
            task_summary["errors"].append(failure)
            task_summary["failed_tasks"] += 1
            stream.output_queue.push("error", failure)
            continue

        task_settings = validated_settings.copy()
        task_settings["state"] = session._state
        filtered_params = {key: value for key, value in task_settings.items() if key in expected_args}
        plugin_data = task.get("plugin_data", {})
        try:
            success = wgp.generate_video(task, send_cmd, plugin_data=plugin_data, **filtered_params)
        except BaseException as exc:
            if not task_errors:
                task_errors.append(session._make_generation_error(exc, task_index=task_index, task_id=task_id, stage="generation"))
                stream.output_queue.push("error", task_errors[-1])
            success = False

        if session._state["gen"].get("abort", False) or job.cancel_requested:
            task_errors.append(GenerationError(message="Generation was cancelled", task_index=task_index, task_id=task_id, stage="cancelled"))
            stream.output_queue.push("error", task_errors[-1])
            task_summary["errors"].extend(task_errors)
            task_summary["failed_tasks"] += 1
            break

        if task_errors:
            task_summary["errors"].extend(task_errors)
            task_summary["failed_tasks"] += 1
            continue
        if not success:
            failure = GenerationError(
                message=f"Task {task_index} did not complete successfully",
                task_index=task_index,
                task_id=task_id,
                stage="generation",
            )
            task_summary["errors"].append(failure)
            task_summary["failed_tasks"] += 1
            stream.output_queue.push("error", failure)
            continue
        task_summary["successful_tasks"] += 1


def _handle_command(session, job: SessionJob, wgp, tasks: list[dict[str, Any]], command: str, data: Any) -> None:
    if command == "progress":
        progress = session._build_progress_update(data)
        job.events.put("progress", progress)
        session._emit_callback("on_progress", progress, job=job)
        return
    if command == "preview":
        preview = session._build_preview_update(wgp, tasks, data)
        if preview is not None:
            job.events.put("preview", preview)
            session._emit_callback("on_preview", preview, job=job)
        return
    if command == "status":
        text = str(data or "")
        job.events.put("status", text)
        session._emit_callback("on_status", text, job=job)
        return
    if command == "info":
        text = str(data or "")
        job.events.put("info", text)
        session._emit_callback("on_info", text, job=job)
        return
    if command == "output":
        job.events.put("output", data)
        session._emit_callback("on_output", data, job=job)
        return
    if command == "refresh_models":
        job.events.put("refresh_models", data)
        return
    if command == "error":
        error = data if isinstance(data, GenerationError) else session._make_generation_error(data)
        job.events.put("error", error)
        session._emit_callback("on_error", error, job=job)
        return
