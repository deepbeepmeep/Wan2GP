"""Runtime worker server helpers and canonical boundaries."""

from __future__ import annotations

import os
import argparse
from importlib import import_module
from pathlib import Path

try:
    from supabase import create_client
except ImportError:  # pragma: no cover - tests monkeypatch this seam
    def create_client(*_args, **_kwargs):
        raise RuntimeError("supabase client is unavailable")

from source.core.db import config as db_config
from source.core.log import headless_logger
from source.core.platform_utils import suppress_alsa_errors
from source.core.params.task_result import TaskOutcome, TaskResult
from source.runtime.process_globals import get_bootstrap_controller, run_bootstrap_once
from source.task_handlers.tasks.task_registry import TaskRegistry
from source.task_handlers.orchestration.finalization_service import (
    WorkerPostGenerationRequest,
    apply_worker_post_generation_policy,
)
from source.task_handlers.travel.chaining import handle_travel_chaining_after_wgp
from source.task_handlers.tasks import task_types
from source.core.runtime_paths import get_repo_root
from source.runtime import wgp_bridge
from source.utils.output_paths import prepare_output_path

repo_root = str(get_repo_root())
wan2gp_path = str((Path(repo_root) / "Wan2GP").resolve())
WORKER_BOOTSTRAP_CONTROLLER = get_bootstrap_controller("worker.server")
STATUS_FAILED = "Failed"
_ensure_runtime_bridge_path = wgp_bridge.ensure_wan2gp_on_path
ensure_wan2gp_on_path = wgp_bridge.ensure_wan2gp_on_path


def update_task_status_supabase(*_args, **_kwargs):
    """Compatibility placeholder for static worker failure contracts."""
    return None


def _resolve_worker_db_client_key(cli_args, *, access_token: str | None) -> str:
    auth_mode = os.environ.get("WORKER_DB_CLIENT_AUTH_MODE", "anon").strip().lower()
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    anon_key = getattr(cli_args, "supabase_anon_key", None) or os.environ.get("SUPABASE_ANON_KEY")

    if auth_mode == "service":
        if not service_key:
            raise ValueError("SERVICE_ROLE_KEY is required when WORKER_DB_CLIENT_AUTH_MODE=service")
        return service_key
    if auth_mode == "worker":
        if not access_token:
            raise ValueError("WORKER_DB_CLIENT_AUTH_MODE=worker requires --reigh-access-token")
        return access_token
    if not anon_key:
        raise ValueError("SUPABASE_ANON_KEY is required for worker DB client initialization")
    return anon_key


def _initialize_db_runtime(cli_args, *, access_token: str | None, debug_mode_enabled: bool):
    supabase_url = getattr(cli_args, "supabase_url", None) or os.environ.get("SUPABASE_URL")
    client_key = _resolve_worker_db_client_key(cli_args, access_token=access_token)
    client = create_client(supabase_url, client_key)
    runtime_cfg = db_config.initialize_db_runtime(
        db_type="supabase",
        pg_table_name=db_config.PG_TABLE_NAME,
        supabase_url=supabase_url,
        supabase_service_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY"),
        supabase_video_bucket=db_config.SUPABASE_VIDEO_BUCKET,
        supabase_client=client,
        supabase_access_token=access_token,
        supabase_edge_complete_task_url=db_config.SUPABASE_EDGE_COMPLETE_TASK_URL,
        supabase_edge_create_task_url=db_config.SUPABASE_EDGE_CREATE_TASK_URL,
        supabase_edge_claim_task_url=db_config.SUPABASE_EDGE_CLAIM_TASK_URL,
        debug=debug_mode_enabled,
    )
    validation_errors = db_config.validate_config(runtime_config=runtime_cfg)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return runtime_cfg, client_key


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
        globals()["ensure_wan2gp_on_path"]()
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
    if generation_success and output_location_to_db:
        normalized_task_params = dict(task_params_dict)
        chain_details = normalized_task_params.get("travel_chain_details")
        if isinstance(chain_details, dict) and chain_details and "enabled" not in chain_details:
            normalized_task_params["travel_chain_details"] = {
                **chain_details,
                "enabled": True,
            }

        output_location_to_db = apply_worker_post_generation_policy(
            request=WorkerPostGenerationRequest(
                task_id=task_id,
                task_type=task_type,
                normalized_task_params=normalized_task_params,
                output_location_to_db=output_location_to_db,
                image_download_dir=str(image_download_dir) if image_download_dir is not None else None,
                main_output_dir_base=main_output_dir_base,
            ),
            chain_handler=lambda **kwargs: handle_travel_chaining_after_wgp(
                wgp_task_params=kwargs["normalized_task_params"],
                actual_wgp_output_video_path=kwargs["actual_wgp_output_video_path"],
                image_download_dir=kwargs["image_download_dir"],
                main_output_dir_base=main_output_dir_base,
            ),
            relocate_output=move_wgp_output_to_task_type_dir,
            log_error=lambda message: headless_logger.error(message, task_id=task_id),
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
    parser = argparse.ArgumentParser(description="Runtime worker")
    parser.add_argument("--poll-interval", type=float, default=15.0, dest="poll_interval")
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--queue-workers", type=int, default=1, dest="queue_workers")
    return parser.parse_args()


def main():
    bootstrap_runtime_environment()
    return import_module("worker").main()
