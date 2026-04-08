"""Runtime worker server helpers and canonical boundaries."""

from __future__ import annotations

import os
import argparse
import signal
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
from source.runtime.worker_protocol import IDLE_RELEASE_EXIT_CODE
from source.runtime.worker import idle_release
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
    auth_mode = os.environ.get("WORKER_DB_CLIENT_AUTH_MODE", "").strip().lower()
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
    # Default: service key > access token > anon key (matches old worker.py behavior)
    resolved = service_key or access_token or anon_key
    if not resolved:
        raise ValueError("No Supabase key found. Provide --reigh-access-token, SUPABASE_SERVICE_ROLE_KEY, or SUPABASE_ANON_KEY")
    return resolved


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
        # With access-token auth, missing service key is non-fatal (old behavior: warn only)
        fatal_errors = [e for e in validation_errors if "SERVICE_KEY" not in e]
        for err in validation_errors:
            if "SERVICE_KEY" in err:
                headless_logger.warning(f"[CONFIG] {err} (non-fatal with access token auth)")
        if fatal_errors:
            raise ValueError("; ".join(fatal_errors))
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
    parser = argparse.ArgumentParser("WanGP Worker Server")
    parser.add_argument("--main-output-dir", type=str, default="./outputs")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--save-logging", type=str, nargs='?', const='logs/worker.log', default=None)
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--colour-match-videos", action="store_true")
    parser.add_argument("--mask-active-frames", dest="mask_active_frames", action="store_true", default=True)
    parser.add_argument("--no-mask-active-frames", dest="mask_active_frames", action="store_false")
    parser.add_argument("--queue-workers", type=int, default=1, dest="queue_workers")
    parser.add_argument("--preload-model", type=str, default="")
    parser.add_argument("--db-type", type=str, default="supabase")
    parser.add_argument("--supabase-url", type=str, default="https://wczysqzxlwdndgxitrvc.supabase.co")
    parser.add_argument("--reigh-access-token", type=str, default=None, help="Access token for Reigh API (preferred)")
    parser.add_argument("--supabase-access-token", type=str, default=None, help="Legacy alias for --reigh-access-token")
    parser.add_argument("--supabase-anon-key", type=str, default=None, help="Supabase anon key (set via env SUPABASE_ANON_KEY)")
    idle_release.add_cli_args(parser)

    # WGP Globals
    parser.add_argument("--wgp-attention-mode", type=str, default=None)
    parser.add_argument("--wgp-compile", type=str, default=None)
    parser.add_argument("--wgp-profile", type=int, default=None)
    parser.add_argument("--wgp-vae-config", type=int, default=None)
    parser.add_argument("--wgp-boost", type=int, default=None)
    parser.add_argument("--wgp-transformer-quantization", type=str, default=None)
    parser.add_argument("--wgp-transformer-dtype-policy", type=str, default=None)
    parser.add_argument("--wgp-text-encoder-quantization", type=str, default=None)
    parser.add_argument("--wgp-vae-precision", type=str, default=None)
    parser.add_argument("--wgp-mixed-precision", type=str, default=None)
    parser.add_argument("--wgp-preload-policy", type=str, default=None)
    parser.add_argument("--wgp-preload", type=int, default=None)

    return parser.parse_args()


def main():
    import sys
    import time
    import datetime
    import logging
    from dotenv import load_dotenv

    print("[WORKER] main() entered", flush=True)

    load_dotenv()
    bootstrap_runtime_environment()

    def _request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    # Steady-state SIGTERM maps to the existing KeyboardInterrupt cleanup path.
    # SIGTERM during DB init, WGP import, or task_queue.start() still exits before
    # the later cleanup finally runs; that startup-window limitation is unchanged.
    signal.signal(signal.SIGTERM, _request_shutdown)
    print("[WORKER] bootstrap done", flush=True)

    cli_args = parse_args()
    print(f"[WORKER] args parsed: worker={cli_args.worker}, debug={cli_args.debug}", flush=True)

    # Resolve access token: prefer --reigh-access-token, fall back to --supabase-access-token
    access_token = cli_args.reigh_access_token or cli_args.supabase_access_token
    if not access_token:
        print("Error: Worker authentication credential is required", file=sys.stderr)
        sys.exit(1)

    # Auto-derive worker_id when not explicitly provided
    if not cli_args.worker:
        cli_args.worker = os.environ.get("RUNPOD_POD_ID") or "local-worker"
    os.environ["WORKER_ID"] = cli_args.worker
    os.environ["WAN2GP_WORKER_MODE"] = "true"

    # Suppress httpx INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from source.core.log import enable_debug_mode, disable_debug_mode, set_log_file

    debug_mode = cli_args.debug
    if debug_mode:
        enable_debug_mode()
        try:
            from mmgp import offload
            offload.default_verboseLevel = 2
        except ImportError:
            pass
        if not cli_args.save_logging:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
            set_log_file(log_file)
            headless_logger.essential(f"Debug logging enabled. Saving to {log_file}")
    else:
        disable_debug_mode()

    if cli_args.save_logging:
        set_log_file(cli_args.save_logging)

    # Initialize DB runtime
    print("[WORKER] initializing DB...", flush=True)
    try:
        _, client_key = _initialize_db_runtime(cli_args, access_token=access_token, debug_mode_enabled=debug_mode)
        os.environ["SUPABASE_URL"] = cli_args.supabase_url
        print("[WORKER] DB initialized", flush=True)
    except (ValueError, OSError, KeyError) as e:
        print(f"[WORKER] DB init failed: {e}", flush=True)
        sys.exit(1)

    if cli_args.migrate_only:
        sys.exit(0)

    main_output_dir = Path(cli_args.main_output_dir).resolve()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    # Centralized logging with heartbeat guardian
    from source.core.log import LogBuffer, CustomLogInterceptor, set_log_interceptor
    from source.runtime.worker.guardian import send_heartbeat_with_logs
    from source.task_handlers.worker.heartbeat_utils import start_heartbeat_guardian_process

    _log_interceptor_instance = None
    guardian_process = None
    guardian_config = None
    if cli_args.worker:
        guardian_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or access_token
        guardian_config = {
            "db_url": cli_args.supabase_url,
            "api_key": guardian_key,
        }
        guardian_process, log_queue = start_heartbeat_guardian_process(
            cli_args.worker, cli_args.supabase_url, guardian_key
        )
        _global_log_buffer = LogBuffer(max_size=100, shared_queue=log_queue)
        _log_interceptor_instance = CustomLogInterceptor(_global_log_buffer)
        set_log_interceptor(_log_interceptor_instance)

    # Apply WGP overrides
    original_cwd = os.getcwd()
    original_argv = sys.argv[:]
    try:
        os.chdir(wan2gp_path)
        sys.path.insert(0, wan2gp_path)
        sys.argv = ["worker.py"]
        import wgp as wgp_mod
        sys.argv = original_argv

        if cli_args.wgp_attention_mode: wgp_mod.attention_mode = cli_args.wgp_attention_mode
        if cli_args.wgp_compile: wgp_mod.compile = cli_args.wgp_compile
        if cli_args.wgp_profile:
            wgp_mod.force_profile_no = cli_args.wgp_profile
            wgp_mod.default_profile = cli_args.wgp_profile
        if cli_args.wgp_vae_config: wgp_mod.vae_config = cli_args.wgp_vae_config
        if cli_args.wgp_boost: wgp_mod.boost = cli_args.wgp_boost
        if cli_args.wgp_transformer_quantization: wgp_mod.transformer_quantization = cli_args.wgp_transformer_quantization
        if cli_args.wgp_transformer_dtype_policy: wgp_mod.transformer_dtype_policy = cli_args.wgp_transformer_dtype_policy
        if cli_args.wgp_text_encoder_quantization: wgp_mod.text_encoder_quantization = cli_args.wgp_text_encoder_quantization
        if cli_args.wgp_vae_precision: wgp_mod.server_config["vae_precision"] = cli_args.wgp_vae_precision
        if cli_args.wgp_mixed_precision: wgp_mod.server_config["mixed_precision"] = cli_args.wgp_mixed_precision
        if cli_args.wgp_preload_policy: wgp_mod.server_config["preload_model_policy"] = [x.strip() for x in cli_args.wgp_preload_policy.split(',')]
        if cli_args.wgp_preload: wgp_mod.server_config["preload_in_VRAM"] = cli_args.wgp_preload
        if "transformer_types" not in wgp_mod.server_config: wgp_mod.server_config["transformer_types"] = []

        print("[WORKER] WGP imported OK", flush=True)

    except (ImportError, RuntimeError, AttributeError, KeyError) as e:
        print(f"[WORKER] WGP import failed: {e}", flush=True)
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    # Clean up legacy collision-prone LoRA files
    from source.models.lora.lora_utils import cleanup_legacy_lora_collisions
    cleanup_legacy_lora_collisions()

    # Initialize Task Queue
    from headless_model_management import HeadlessTaskQueue
    try:
        task_queue = HeadlessTaskQueue(
            wan_dir=wan2gp_path,
            max_workers=cli_args.queue_workers,
            debug_mode=debug_mode,
            main_output_dir=str(main_output_dir)
        )
        preload_model = cli_args.preload_model if cli_args.preload_model else None
        task_queue.start(preload_model=preload_model)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"[WORKER] Queue init failed: {e}", flush=True)
        sys.exit(1)

    print(f"[WORKER] started. Polling every {cli_args.poll_interval}s.", flush=True)

    # Import task processing dependencies
    from source.core.db.task_claim import ClaimPollOutcome, poll_next_task
    from source.core.db.task_status import (
        update_task_status_supabase as _update_task_complete,
        update_task_status,
    )
    from source.core.db.lifecycle.task_status_retry import requeue_task_for_retry
    from source.task_handlers.worker.fatal_error_handler import FatalWorkerError, reset_fatal_error_counter, is_retryable_error
    from source.task_handlers.worker.worker_utils import cleanup_generated_files

    STATUS_COMPLETE = "Complete"
    STATUS_IN_PROGRESS = "In Progress"
    max_task_wait_minutes = int(os.getenv("MAX_TASK_WAIT_MINUTES", "5"))
    idle_tracker = idle_release.IdleReleaseTracker(
        idle_release.config_from_cli(cli_args, client_key=client_key)
    )
    idle_tracker.mark_onboarded()

    try:
        while True:
            poll_outcome, task_info = poll_next_task(
                worker_id=cli_args.worker,
                same_model_only=True,
                max_task_wait_minutes=max_task_wait_minutes,
            )

            if poll_outcome == ClaimPollOutcome.EMPTY:
                idle_tracker.record_empty_poll()
                if idle_tracker.should_release():
                    headless_logger.essential(
                        f"[WORKER] Idle for >={cli_args.idle_release_minutes:.1f} min — releasing resources (exit {IDLE_RELEASE_EXIT_CODE})"
                    )
                    sys.exit(IDLE_RELEASE_EXIT_CODE)
                time.sleep(cli_args.poll_interval)
                continue
            if poll_outcome == ClaimPollOutcome.ERROR:
                time.sleep(cli_args.poll_interval)
                continue

            idle_tracker.record_claim()

            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"]
            current_project_id = task_info.get("project_id")
            current_task_id = task_info["task_id"]

            try:
                if current_project_id is None and current_task_type in {"travel_orchestrator", "edit_video_orchestrator"}:
                    _update_task_complete(current_task_id, STATUS_FAILED, "Orchestrator missing project_id")
                    continue

                current_task_params["task_id"] = current_task_id
                if "orchestrator_details" in current_task_params:
                    current_task_params["orchestrator_details"]["orchestrator_task_id"] = current_task_id

                if _log_interceptor_instance:
                    _log_interceptor_instance.set_current_task(current_task_id)

                raw_result = process_single_task(
                    task_params_dict=current_task_params,
                    main_output_dir_base=main_output_dir,
                    task_type=current_task_type,
                    project_id_for_task=current_project_id,
                    image_download_dir=current_task_params.get("segment_image_download_dir"),
                    colour_match_videos=cli_args.colour_match_videos,
                    mask_active_frames=cli_args.mask_active_frames,
                    task_queue=task_queue,
                )

                if isinstance(raw_result, TaskResult):
                    result = raw_result
                    task_succeeded, output_location = raw_result  # __iter__ unpacking
                else:
                    task_succeeded, output_location = raw_result
                    result = None

                if task_succeeded:
                    reset_fatal_error_counter()

                    orchestrator_types = {"travel_orchestrator", "join_clips_orchestrator", "edit_video_orchestrator"}

                    if current_task_type in orchestrator_types:
                        if result and result.outcome == TaskOutcome.ORCHESTRATOR_COMPLETE:
                            _update_task_complete(
                                current_task_id, STATUS_COMPLETE,
                                result.output_path, result.thumbnail_url)
                        elif result and result.outcome == TaskOutcome.ORCHESTRATING:
                            update_task_status(current_task_id, STATUS_IN_PROGRESS, result.output_path)
                        elif isinstance(output_location, str) and output_location.startswith("[ORCHESTRATOR_COMPLETE]"):
                            import json as _json
                            actual_output = output_location.replace("[ORCHESTRATOR_COMPLETE]", "")
                            thumbnail_url = None
                            try:
                                data = _json.loads(actual_output)
                                actual_output = data.get("output_location", actual_output)
                                thumbnail_url = data.get("thumbnail_url")
                            except (ValueError, TypeError, KeyError):
                                pass
                            _update_task_complete(current_task_id, STATUS_COMPLETE, actual_output, thumbnail_url)
                        else:
                            update_task_status(current_task_id, STATUS_IN_PROGRESS, output_location)
                    else:
                        _update_task_complete(current_task_id, STATUS_COMPLETE, output_location)
                        cleanup_generated_files(output_location, current_task_id, debug_mode)
                else:
                    error_message = (result.error_message if result else output_location) or "Unknown error"
                    is_retryable, error_category, max_attempts = is_retryable_error(error_message)
                    current_attempts = task_info.get("attempts", 0)

                    if is_retryable and current_attempts < max_attempts:
                        headless_logger.warning(
                            f"Task {current_task_id} failed with retryable error ({error_category}), "
                            f"requeuing for retry (attempt {current_attempts + 1}/{max_attempts})"
                        )
                        requeue_task_for_retry(current_task_id, error_message, current_attempts, error_category)
                    else:
                        if is_retryable and current_attempts >= max_attempts:
                            headless_logger.error(
                                f"Task {current_task_id} exhausted {max_attempts} retry attempts for {error_category}"
                            )
                        _update_task_complete(current_task_id, STATUS_FAILED, output_location)
            except FatalWorkerError:
                raise
            except Exception as e:
                headless_logger.error(
                    f"Unhandled exception while processing task {current_task_id}: {e}",
                    task_id=current_task_id,
                    exc_info=True,
                )
                try:
                    error_message = str(e) or e.__class__.__name__
                    is_retryable, error_category, max_attempts = is_retryable_error(error_message)
                    current_attempts = task_info.get("attempts", 0)

                    if is_retryable and current_attempts < max_attempts:
                        headless_logger.warning(
                            f"Task {current_task_id} failed with retryable error ({error_category}), "
                            f"requeuing for retry (attempt {current_attempts + 1}/{max_attempts})"
                        )
                        requeue_task_for_retry(current_task_id, error_message, current_attempts, error_category)
                    else:
                        if is_retryable and current_attempts >= max_attempts:
                            headless_logger.error(
                                f"Task {current_task_id} exhausted {max_attempts} retry attempts for {error_category}"
                            )
                        _update_task_complete(current_task_id, STATUS_FAILED, error_message)
                except:
                    headless_logger.error(
                        f"Failed to persist task failure for {current_task_id} after unhandled exception",
                        task_id=current_task_id,
                        exc_info=True,
                    )
                continue
            finally:
                if _log_interceptor_instance:
                    _log_interceptor_instance.set_current_task(None)

            time.sleep(1)

    except FatalWorkerError as e:
        headless_logger.critical(f"Fatal Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        headless_logger.essential("Shutting down...")
    finally:
        if guardian_process:
            guardian_process.terminate()
            guardian_process.join(5)
        if cli_args.worker and guardian_config:
            send_heartbeat_with_logs(
                cli_args.worker,
                0,
                0,
                [],
                guardian_config,
                status="terminated",
            )
        if task_queue:
            task_queue.stop()
