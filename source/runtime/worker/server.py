"""Runtime worker server helpers and canonical boundaries."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

from source.core.log import headless_logger
from source.core.platform_utils import suppress_alsa_errors
from source.core.params.task_result import TaskOutcome, TaskResult
from source.runtime.process_globals import get_bootstrap_controller, run_bootstrap_once
from source.task_handlers.tasks.task_registry import TaskRegistry
from source.task_handlers.travel.chaining import handle_travel_chaining_after_wgp
from source.task_handlers.tasks import task_types
from source.core.runtime_paths import ensure_wan2gp_on_path, get_repo_root
from source.utils.output_paths import prepare_output_path

repo_root = str(get_repo_root())
wan2gp_path = str((Path(repo_root) / "Wan2GP").resolve())
WORKER_BOOTSTRAP_CONTROLLER = get_bootstrap_controller("worker.server")
STATUS_FAILED = "Failed"


def update_task_status_supabase(*_args, **_kwargs):
    """Compatibility placeholder for static worker failure contracts."""
    return None


def _handle_task_failure(task_id: str, error_message: str):
    """Persist worker failure state using the error message channel."""
    update_task_status_supabase(task_id, STATUS_FAILED, error_message)
    return TaskResult.failed(error_message)


def bootstrap_runtime_environment() -> dict[str, object]:
    def _initializer() -> None:
        os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        ensure_wan2gp_on_path()
        suppress_alsa_errors()

    return run_bootstrap_once(
        "worker.runtime_environment",
        _initializer,
        version="2026-02-27",
        controller=WORKER_BOOTSTRAP_CONTROLLER,
    )


def move_wgp_output_to_task_type_dir(
    *,
    output_path: str,
    task_type: str,
    task_id: str,
    main_output_dir_base: Path,
) -> str:
    if not task_types.is_wgp_task(task_type):
        return output_path
    output_file = Path(output_path)
    if not output_file.exists():
        return output_path
    if output_file.parent.resolve() != main_output_dir_base.resolve():
        return output_path
    new_path, _ = prepare_output_path(
        task_id=task_id,
        filename=output_file.name,
        main_output_dir_base=main_output_dir_base,
        task_type=task_type,
    )
    new_path.parent.mkdir(parents=True, exist_ok=True)
    output_file.rename(new_path)
    headless_logger.info(f"Moved WGP output to {new_path}", task_id=task_id)
    return str(new_path)


def process_single_task(
    *,
    task_params_dict,
    main_output_dir_base: Path,
    task_type: str,
    project_id_for_task,
    image_download_dir: Path | str | None = None,
    colour_match_videos: bool = False,
    mask_active_frames: bool = True,
    task_queue=None,
):
    task_id = task_params_dict.get("task_id", "unknown")
    context = {
        "task_params_dict": task_params_dict,
        "main_output_dir_base": main_output_dir_base,
        "task_id": task_id,
        "project_id": project_id_for_task,
        "task_queue": task_queue,
        "colour_match_videos": colour_match_videos,
        "mask_active_frames": mask_active_frames,
        "debug_mode": False,
        "wan2gp_path": wan2gp_path,
    }
    result = TaskRegistry.dispatch(task_type, context)
    if isinstance(result, TaskResult) and result.outcome == TaskOutcome.FAILED:
        return result

    generation_success, output_location_to_db = result if not isinstance(result, TaskResult) else (True, result.output_path)
    if generation_success and task_params_dict.get("travel_chain_details"):
        chain_success, _message, final_path = handle_travel_chaining_after_wgp(
            wgp_task_params=task_params_dict,
            actual_wgp_output_video_path=output_location_to_db,
            image_download_dir=image_download_dir,
            main_output_dir_base=main_output_dir_base,
        )
        if chain_success:
            output_location_to_db = final_path

    if generation_success and output_location_to_db:
        output_location_to_db = move_wgp_output_to_task_type_dir(
            output_path=output_location_to_db,
            task_type=task_type,
            task_id=task_id,
            main_output_dir_base=main_output_dir_base,
        )

    if isinstance(result, TaskResult):
        return TaskResult(
            outcome=result.outcome,
            output_path=output_location_to_db,
            thumbnail_url=result.thumbnail_url,
            metadata=result.metadata,
        )
    return TaskResult.success(output_location_to_db) if generation_success else TaskResult.failed("generation failed")


def parse_args():
    return import_module("worker").parse_args()


def main():
    bootstrap_runtime_environment()
    return import_module("worker").main()
