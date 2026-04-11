from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

FALLBACK_PYTHON = Path("/usr/local/bin/python3.12")


def _ensure_supported_python() -> None:
    if sys.version_info >= (3, 10):
        return
    if FALLBACK_PYTHON.exists() and Path(sys.executable).resolve() != FALLBACK_PYTHON.resolve():
        os.execv(str(FALLBACK_PYTHON), [str(FALLBACK_PYTHON), __file__, *sys.argv[1:]])
    raise SystemExit(
        "Preview harness requires Python 3.10+; "
        "re-run with /usr/local/bin/python3.12 scripts/preview/run_preview.py"
    )


_ensure_supported_python()

WORKER_ROOT = Path(__file__).resolve().parents[2]
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from scripts.preview.assets import ensure_assets
from scripts.preview.db_spoof import SpoofDbRuntime, SpoofTaskFeed, inject_db_config
from scripts.preview.fixtures import get_fixtures
from scripts.preview.network_spoof import build_network_spoofs
from scripts.preview.wgp_spoof import SpoofQueue, ensure_samples_dir, inject_module_stubs


class TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Preview harness runner")
    parser.add_argument("--task-types", type=str, default=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--debug", dest="debug", action="store_true")
    mode_group.add_argument("--normal", dest="debug", action="store_false")
    parser.set_defaults(debug=False)
    parser.add_argument("--output-log", type=str, default=None)
    parser.add_argument("--idle-polls", type=int, default=1)
    return parser.parse_args(argv)


def _prepare_environment() -> dict[str, str]:
    os.environ.setdefault("HEADLESS_WAN2GP_SMOKE", "1")
    os.environ.setdefault("HEADLESS_WAN2GP_FORCE_CPU", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/preview/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/preview/cache")
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    inject_module_stubs()
    inject_db_config()
    ensure_samples_dir()
    return ensure_assets(Path(__file__).resolve().parent)


def _set_dotted_attr(
    dotted_path: str,
    value,
    patch_log: dict[str, tuple[object, str, bool, Any]] | None = None,
) -> None:
    module_name, _, attribute = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        if patch_log is not None and dotted_path not in patch_log:
            existed = hasattr(module, attribute)
            old_value = getattr(module, attribute) if existed else None
            patch_log[dotted_path] = (module, attribute, existed, old_value)
        setattr(module, attribute, value)
    except (ImportError, AttributeError):
        pass  # Module not present in this codebase version — skip gracefully


def _restore_patches(patch_log: dict[str, tuple[object, str, bool, Any]]) -> None:
    for dotted_path in reversed(list(patch_log.keys())):
        module, attribute, existed, old_value = patch_log[dotted_path]
        if existed:
            setattr(module, attribute, old_value)
        elif hasattr(module, attribute):
            delattr(module, attribute)


def _apply_patch_targets(
    targets: Iterable[str],
    value,
    patch_log: dict[str, tuple[object, str, bool, Any]],
) -> None:
    for dotted_path in targets:
        _set_dotted_attr(dotted_path, value, patch_log)


def _install_spoofs(
    *,
    spoof_db: SpoofDbRuntime,
    feed: SpoofTaskFeed,
    asset_paths: dict[str, str],
) -> dict[str, tuple[object, str, bool, Any]]:
    patch_log: dict[str, tuple[object, str, bool, Any]] = {}
    network_spoofs = build_network_spoofs(asset_paths)

    def preview_transition_prompt(start_image_path, end_image_path, base_prompt=None, device="cuda", task_id=None, upload_debug_images=True):
        _ = start_image_path, end_image_path, device, task_id, upload_debug_images
        return (base_prompt or "preview transition motion").strip()

    def preview_transition_prompts_batch(image_pairs, base_prompts, device="cuda", task_id=None, upload_debug_images=True):
        _ = image_pairs, device, task_id, upload_debug_images
        return [(prompt or "preview transition motion").strip() for prompt in base_prompts]

    def preview_single_image_prompts_batch(image_paths, base_prompts, device="cuda"):
        _ = image_paths, device
        return [(prompt or "preview cinematic motion").strip() for prompt in base_prompts]

    def preview_upload_intermediate_file_to_storage(local_file_path, task_id, filename, runtime_config=None):
        _ = task_id, filename, runtime_config
        path = Path(local_file_path)
        return str(path.resolve()) if path.exists() else None

    def preview_upload_and_get_final_output_location(local_file_path, initial_db_location):
        _ = initial_db_location
        path = Path(local_file_path)
        return str(path.resolve()) if path.exists() else str(path)

    def preview_httpx_post(*_args, **_kwargs):
        class _PreviewResponse:
            status_code = 200
            text = '{"success": true}'

            @staticmethod
            def json():
                return {"success": True}

        return _PreviewResponse()

    def preview_handle_join_clips_task(task_params_from_db, main_output_dir_base, task_id, task_queue=None):
        _ = task_queue
        from source.core.log import orchestrator_logger

        output_dir = Path(main_output_dir_base) / "join_clips_segment"
        output_dir.mkdir(parents=True, exist_ok=True)
        transition_path = output_dir / f"{task_id}_transition.mp4"
        shutil.copy2(asset_paths["video"], transition_path)
        transition_index = int(task_params_from_db.get("transition_index", 0))
        transition_payload = {
            "transition_url": str(transition_path.resolve()),
            "transition_index": transition_index,
            "frames": 1,
            "fps": task_params_from_db.get("fps", 16),
            "gap_frames": 0,
            "blend_frames": 0,
            "context_from_clip1": 0,
            "context_from_clip2": 0,
            "gap_from_clip1": 0,
            "gap_from_clip2": 0,
        }
        orchestrator_logger.debug(
            f"[JOIN_CLIPS] Task {task_id}: preview transition_only output ready at {transition_path}"
        )
        return True, json.dumps(transition_payload)

    def preview_handle_join_final_stitch(task_params_from_db, main_output_dir_base, task_id):
        from source.core.log import orchestrator_logger

        output_dir = Path(main_output_dir_base) / "join_final_stitch"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{task_id}_joined.mp4"
        shutil.copy2(asset_paths["video"], final_path)
        orchestrator_logger.debug(
            f"[FINAL_STITCH] Task {task_id}: preview final stitch output ready at {final_path}"
        )
        return True, str(final_path.resolve())

    _apply_patch_targets(
        [
            "source.core.db.task_claim.poll_next_task",
        ],
        feed.poll,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_status.update_task_status_supabase",
            "source.runtime.worker.server.update_task_status_supabase",
        ],
        spoof_db.update_status_complete,
        patch_log,
    )

    _apply_patch_targets(
        [
            "source.core.db.task_status.update_task_status",
            "source.utils.orchestrator_utils.update_task_status",
            "source.core.db.task_dependencies.update_task_status",
        ],
        spoof_db.update_status,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_status.reset_generation_started_at",
        ],
        spoof_db.reset_generation_started_at,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.lifecycle.task_status_retry.requeue_task_for_retry",
        ],
        spoof_db.requeue,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_completion.add_task_to_db",
            "source.core.db.lifecycle.task_completion.add_task_to_db",
            "source.task_handlers.join.task_builder.add_task_to_db",
        ],
        spoof_db.add_task,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_dependencies.get_orchestrator_child_tasks",
            "source.task_handlers.travel.orchestrator.get_orchestrator_child_tasks",
            "source.task_handlers.join.shared.get_orchestrator_child_tasks",
            "source.task_handlers.travel.orchestration.orchestrator.get_orchestrator_child_tasks",
        ],
        spoof_db.get_children,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_dependencies.get_task_dependency",
        ],
        spoof_db.get_dependency,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_dependencies.get_task_current_status",
            "source.core.db.dependencies.task_dependencies_children.get_task_current_status",
            "source.task_handlers.join.shared.get_task_current_status",
        ],
        spoof_db.get_status,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_dependencies.cancel_orchestrator_children",
            "source.core.db.dependencies.task_dependencies_children.cancel_orchestrator_children",
            "source.task_handlers.join.shared.cancel_orchestrator_children",
        ],
        spoof_db.cancel_children,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_dependencies.cleanup_duplicate_child_tasks",
            "source.task_handlers.travel.orchestrator.cleanup_duplicate_child_tasks",
            "source.task_handlers.travel.orchestration.orchestrator.cleanup_duplicate_child_tasks",
        ],
        spoof_db.cleanup_duplicate_child_tasks,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.core.db.task_polling.get_task_output_location_from_db",
            "source.core.db.task_dependencies.get_task_output_location_from_db",
            "source.task_handlers.extract_frame.get_task_output_location_from_db",
            "source.task_handlers.join.orchestrator.get_task_output_location_from_db",
        ],
        spoof_db.get_output,
        patch_log,
    )
    _apply_patch_targets(
        network_spoofs["import_site_targets"]["download_lora_from_url"],
        network_spoofs["download_lora_from_url"],
        patch_log,
    )
    _apply_patch_targets(
        network_spoofs["import_site_targets"]["download_file"],
        network_spoofs["download_file"],
        patch_log,
    )
    _apply_patch_targets(
        network_spoofs["import_site_targets"]["download_image_if_url"],
        network_spoofs["download_image_if_url"],
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.runtime.worker.guardian.send_heartbeat_with_logs",
        ],
        lambda *_args, **_kwargs: True,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.task_handlers.worker.heartbeat_utils.start_heartbeat_guardian_process",
        ],
        lambda *_args, **_kwargs: (None, None),
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.task_handlers.worker.worker_utils.cleanup_generated_files",
        ],
        lambda *_args, **_kwargs: None,
        patch_log,
    )
    _apply_patch_targets(
        [
            "httpx.post",
        ],
        preview_httpx_post,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.utils.output_paths.upload_intermediate_file_to_storage",
            "source.utils.upload_intermediate_file_to_storage",
            "source.task_handlers.join.generation.upload_intermediate_file_to_storage",
            "source.task_handlers.join.orchestrator.upload_intermediate_file_to_storage",
            "source.task_handlers.join.clip_preprocessor.upload_intermediate_file_to_storage",
            "source.task_handlers.tasks.task_registry.upload_intermediate_file_to_storage",
        ],
        preview_upload_intermediate_file_to_storage,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.utils.output_paths.upload_and_get_final_output_location",
            "source.utils.upload_and_get_final_output_location",
            "source.task_handlers.join.generation.upload_and_get_final_output_location",
            "source.task_handlers.join.final_stitch.upload_and_get_final_output_location",
            "source.task_handlers.travel.orchestrator.upload_and_get_final_output_location",
            "source.task_handlers.travel.stitch.upload_and_get_final_output_location",
        ],
        preview_upload_and_get_final_output_location,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.media.vlm.generate_transition_prompts_batch",
            "source.media.vlm.api.generate_transition_prompts_batch",
        ],
        preview_transition_prompts_batch,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.task_handlers.join.generation.handle_join_clips_task",
            "source.task_handlers.tasks.task_registry.handle_join_clips_task",
        ],
        preview_handle_join_clips_task,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.task_handlers.join.final_stitch.handle_join_final_stitch",
            "source.task_handlers.tasks.task_registry.handle_join_final_stitch",
        ],
        preview_handle_join_final_stitch,
        patch_log,
    )
    _apply_patch_targets(
        [
            "source.media.vlm.generate_single_image_prompts_batch",
            "source.media.vlm.single_image_prompts.generate_single_image_prompts_batch",
        ],
        preview_single_image_prompts_batch,
        patch_log,
    )
    _set_dotted_attr("source.media.vlm.generate_transition_prompt", preview_transition_prompt, patch_log)
    _set_dotted_attr("source.media.vlm.api.generate_transition_prompt", preview_transition_prompt, patch_log)
    _set_dotted_attr("source.media.vlm.generate_single_image_prompts_batch", preview_single_image_prompts_batch, patch_log)
    _set_dotted_attr(
        "source.media.vlm.single_image_prompts.generate_single_image_prompts_batch",
        preview_single_image_prompts_batch,
        patch_log,
    )
    return patch_log


def _normalize_raw_result(raw_result, TaskResult):
    if isinstance(raw_result, TaskResult):
        result = raw_result
        task_succeeded, output_location = raw_result
    else:
        task_succeeded, output_location = raw_result
        result = None
    return result, task_succeeded, output_location


def _summarize_statuses(spoof_db: SpoofDbRuntime) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in spoof_db._records.values():
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def run_preview(argv: Iterable[str] | None = None) -> int:
    from source.core.log.core import _is_env_debug, disable_debug_mode, enable_debug_mode, suppress_library_logging

    args = parse_args(argv)
    debug_mode = args.debug or _is_env_debug()
    asset_paths = _prepare_environment()

    from source.core.log import headless_logger, set_current_task_context
    from source.core.params.task_result import TaskOutcome, TaskResult
    from source.runtime.worker import server as worker_server
    from source.runtime.worker.server import bootstrap_runtime_environment, process_single_task
    from source.runtime.worker.status_display import WorkerStatusDisplay
    from source.task_handlers.worker.fatal_error_handler import FatalWorkerError, is_retryable_error, reset_fatal_error_counter
    from source.task_handlers.worker import worker_utils
    from source.core.db.task_claim import ClaimPollOutcome

    fixtures = get_fixtures(args.task_types)
    spoof_db = SpoofDbRuntime(fixtures)
    feed = SpoofTaskFeed(spoof_db, idle_polls_between_tasks=args.idle_polls)
    spoof_queue = SpoofQueue(wan_dir=worker_server.wan2gp_path, db_runtime=spoof_db)
    patch_log = _install_spoofs(spoof_db=spoof_db, feed=feed, asset_paths=asset_paths)

    bootstrap_runtime_environment()
    if debug_mode:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        enable_debug_mode()
    else:
        disable_debug_mode()
        suppress_library_logging()

    display = WorkerStatusDisplay("Preview GPU (spoofed)", "Preview Profile")
    display.show_banner()
    display.show_idle()

    main_output_dir = (WORKER_ROOT / "outputs").resolve()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    STATUS_COMPLETE = "Complete"
    STATUS_FAILED = "Failed"
    STATUS_IN_PROGRESS = "In Progress"
    orchestrator_types = {"travel_orchestrator", "join_clips_orchestrator", "edit_video_orchestrator"}

    headless_logger.essential(
        f"[PREVIEW] Starting preview harness with {len(fixtures)} fixture(s): "
        f"{', '.join(item['task_type'] for item in fixtures)}"
    )

    try:
        while True:
            poll_outcome, task_info = feed.poll(
                worker_id="preview-worker",
                same_model_only=True,
                max_task_wait_minutes=5,
            )
            if poll_outcome is None:
                break
            if poll_outcome == ClaimPollOutcome.EMPTY:
                display.show_idle()
                time.sleep(0.5)
                continue

            current_task_params = copy.deepcopy(task_info["params"])
            current_task_type = task_info["task_type"]
            current_project_id = task_info.get("project_id")
            current_task_id = task_info["task_id"]

            display.on_task_start()
            headless_logger.essential(f"[PREVIEW] Processing {current_task_type} ({current_task_id})")
            set_current_task_context(current_task_id)

            try:
                if current_project_id is None and current_task_type in {"travel_orchestrator", "edit_video_orchestrator"}:
                    spoof_db.update_status_complete(current_task_id, STATUS_FAILED, "Orchestrator missing project_id")
                    continue

                current_task_params["task_id"] = current_task_id
                if isinstance(current_task_params.get("orchestrator_details"), dict):
                    current_task_params["orchestrator_details"]["orchestrator_task_id"] = current_task_id

                raw_result = process_single_task(
                    task_params_dict=current_task_params,
                    main_output_dir_base=main_output_dir,
                    task_type=current_task_type,
                    project_id_for_task=current_project_id,
                    image_download_dir=current_task_params.get("segment_image_download_dir"),
                    colour_match_videos=False,
                    mask_active_frames=True,
                    task_queue=spoof_queue,
                )
                result, task_succeeded, output_location = _normalize_raw_result(raw_result, TaskResult)

                if task_succeeded:
                    reset_fatal_error_counter()
                    if current_task_type in orchestrator_types:
                        if result and result.outcome == TaskOutcome.ORCHESTRATOR_COMPLETE:
                            spoof_db.update_status_complete(
                                current_task_id,
                                STATUS_COMPLETE,
                                result.output_path,
                                result.thumbnail_url,
                            )
                        elif result and result.outcome == TaskOutcome.ORCHESTRATING:
                            spoof_db.update_status(
                                current_task_id,
                                STATUS_IN_PROGRESS,
                                result.output_path or result.progress_message,
                            )
                        elif isinstance(output_location, str) and output_location.startswith("[ORCHESTRATOR_COMPLETE]"):
                            actual_output = output_location.replace("[ORCHESTRATOR_COMPLETE]", "")
                            spoof_db.update_status_complete(current_task_id, STATUS_COMPLETE, actual_output)
                        else:
                            spoof_db.update_status(current_task_id, STATUS_IN_PROGRESS, output_location)
                    else:
                        spoof_db.update_status_complete(current_task_id, STATUS_COMPLETE, output_location)
                        worker_utils.cleanup_generated_files(output_location, current_task_id, debug_mode)
                else:
                    error_message = (result.error_message if result else output_location) or "Unknown error"
                    is_retryable, error_category, max_attempts = is_retryable_error(error_message)
                    current_attempts = task_info.get("attempts", 0)
                    if is_retryable and current_attempts < max_attempts:
                        headless_logger.warning(
                            f"Task {current_task_id} failed with retryable error ({error_category}), "
                            f"requeuing for retry (attempt {current_attempts + 1}/{max_attempts})"
                        )
                        spoof_db.requeue(current_task_id, error_message, current_attempts, error_category)
                    else:
                        if is_retryable and current_attempts >= max_attempts:
                            headless_logger.error(
                                f"Task {current_task_id} exhausted {max_attempts} retry attempts for {error_category}"
                            )
                        spoof_db.update_status_complete(current_task_id, STATUS_FAILED, output_location)
            except FatalWorkerError:
                raise
            except Exception as exc:
                headless_logger.error(
                    f"Unhandled exception while processing task {current_task_id}: {exc}",
                    task_id=current_task_id,
                    exc_info=True,
                )
                error_message = str(exc) or exc.__class__.__name__
                is_retryable, error_category, max_attempts = is_retryable_error(error_message)
                current_attempts = task_info.get("attempts", 0)
                if is_retryable and current_attempts < max_attempts:
                    spoof_db.requeue(current_task_id, error_message, current_attempts, error_category)
                else:
                    spoof_db.update_status_complete(current_task_id, STATUS_FAILED, error_message)
            finally:
                set_current_task_context(None)
                display.on_task_done()

        summary = ", ".join(
            f"{status}={count}" for status, count in sorted(_summarize_statuses(spoof_db).items())
        ) or "no tasks"
        headless_logger.essential(f"[PREVIEW] Preview harness complete: {summary}")
        return 0
    except FatalWorkerError as exc:
        headless_logger.critical(f"[PREVIEW] Fatal worker error: {exc}")
        return 1
    finally:
        _restore_patches(patch_log)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.output_log:
        return run_preview(argv)

    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            return run_preview(argv)


if __name__ == "__main__":
    raise SystemExit(main())
