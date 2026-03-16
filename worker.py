"""Wan2GP Worker Server.

This long-running process polls the Supabase-backed Postgres `tasks` table,
claims queued tasks, and executes them using the HeadlessTaskQueue system.
"""

# Suppress common warnings/errors from headless environment
import os
import sys
import warnings

# Suppress Python warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")

# Set environment variables for headless operation
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Suppress ALSA errors on Linux (moved to platform_utils for cleaner code)
from source.core.platform_utils import suppress_alsa_errors
suppress_alsa_errors()

import argparse
import time
import datetime
import threading
import logging
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
wan2gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2GP")
if wan2gp_path not in sys.path:
    sys.path.append(wan2gp_path)

from source import db_operations as db_ops
from source.core.db import config as db_config
from source.task_handlers.worker.fatal_error_handler import FatalWorkerError, reset_fatal_error_counter, is_retryable_error
from headless_model_management import HeadlessTaskQueue

from source.core.log import (
    headless_logger, enable_debug_mode, disable_debug_mode,
    LogBuffer, CustomLogInterceptor, set_log_interceptor, set_log_file
)
from source.task_handlers.worker.worker_utils import cleanup_generated_files
from source.task_handlers.worker.heartbeat_utils import start_heartbeat_guardian_process
from source.task_handlers.tasks.task_registry import TaskRegistry
from source.task_handlers.travel.chaining import handle_travel_chaining_after_wgp
from source.models.lora.lora_utils import cleanup_legacy_lora_collisions
from source.utils import prepare_output_path
import shutil

# Global heartbeat control
heartbeat_thread = None
heartbeat_stop_event = threading.Event()
debug_mode = False


def move_wgp_output_to_task_type_dir(output_path: str, task_type: str, task_id: str, main_output_dir_base: Path) -> str:
    """
    Move WGP-generated output from base directory to task-type subdirectory.

    This function handles post-processing of WGP outputs to organize them by task_type.
    WGP generates all files to a globally-configured base directory, so we move them
    to task-type subdirectories after generation completes.

    Args:
        output_path: Path to the WGP-generated file (in base directory)
        task_type: Task type (e.g., "vace", "flux", "t2v")
        task_id: Task ID
        main_output_dir_base: Base output directory

    Returns:
        New path if file was moved, original path otherwise
    """
    # Only process WGP task types
    from source.task_handlers.tasks.task_types import WGP_TASK_TYPES
    if task_type not in WGP_TASK_TYPES:
        return output_path

    try:
        output_file = Path(output_path)

        # Check if file exists
        if not output_file.exists():
            headless_logger.debug(f"Output file doesn't exist, skipping move: {output_path}", task_id=task_id)
            return output_path

        # Check if file is in base directory (not already in a subdirectory)
        # If the parent directory is main_output_dir_base, it's in base directory
        if output_file.parent.resolve() != main_output_dir_base.resolve():
            headless_logger.debug(f"Output already in subdirectory, skipping move: {output_path}", task_id=task_id)
            return output_path

        # Generate new path in task-type subdirectory
        filename = output_file.name
        new_path, _ = prepare_output_path(
            task_id=task_id,
            filename=filename,
            main_output_dir_base=main_output_dir_base,
            task_type=task_type
        )

        # Move the file
        headless_logger.info(
            f"Moving WGP output to task-type directory: {output_file} → {new_path} (task_type={task_type})",
            task_id=task_id
        )

        # Ensure destination directory exists
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(output_file), str(new_path))

        headless_logger.success(f"Moved WGP output to {new_path}", task_id=task_id)
        return str(new_path)

    except (OSError, shutil.Error, ValueError) as e:
        headless_logger.error(f"Failed to move WGP output to task-type directory: {e}", task_id=task_id, exc_info=True)
        # Return original path if move fails
        return output_path

def process_single_task(task_params_dict, main_output_dir_base: Path, task_type: str, project_id_for_task: str | None, image_download_dir: Path | str | None = None, colour_match_videos: bool = False, mask_active_frames: bool = True, task_queue: HeadlessTaskQueue = None):
    from source.core.params.task_result import TaskResult, TaskOutcome

    task_id = task_params_dict.get("task_id", "unknown_task_" + str(time.time()))
    headless_logger.essential(f"Processing {task_type} task", task_id=task_id)

    context = {
        "task_params_dict": task_params_dict,
        "main_output_dir_base": main_output_dir_base,
        "task_id": task_id,
        "project_id": project_id_for_task,
        "task_queue": task_queue,
        "colour_match_videos": colour_match_videos,
        "mask_active_frames": mask_active_frames,
        "debug_mode": debug_mode,
        "wan2gp_path": wan2gp_path,
    }

    result = TaskRegistry.dispatch(task_type, context)
    generation_success, output_location_to_db = result  # backward-compat unpacking

    # Chaining Logic
    if generation_success:
        chaining_result_path_override = None

        if task_params_dict.get("travel_chain_details"):
            chain_success, chain_message, final_path_from_chaining = handle_travel_chaining_after_wgp(
                wgp_task_params=task_params_dict,
                actual_wgp_output_video_path=output_location_to_db,
                image_download_dir=image_download_dir,
                main_output_dir_base=main_output_dir_base,
            )
            if chain_success:
                chaining_result_path_override = final_path_from_chaining
            else:
                headless_logger.error(f"Travel chaining failed: {chain_message}", task_id=task_id)

        if chaining_result_path_override:
            output_location_to_db = chaining_result_path_override

        # Move WGP outputs to task-type subdirectories
        # This post-processes WGP-generated files to organize them by task_type
        if output_location_to_db:
            output_location_to_db = move_wgp_output_to_task_type_dir(
                output_path=output_location_to_db,
                task_type=task_type,
                task_id=task_id,
                main_output_dir_base=main_output_dir_base
            )

    # Ensure orchestrator tasks use their DB row ID
    if task_type in {"travel_orchestrator"}:
        task_params_dict["task_id"] = task_id

    headless_logger.essential(f"Finished task (Success: {generation_success})", task_id=task_id)

    # Return structured TaskResult when the dispatch returned one, preserving
    # thumbnail_url and outcome through the post-processing pipeline.
    if isinstance(result, TaskResult):
        if result.outcome == TaskOutcome.FAILED:
            return result
        # Rebuild with potentially-modified output_location_to_db
        return TaskResult(
            outcome=result.outcome,
            output_path=output_location_to_db,
            thumbnail_url=result.thumbnail_url,
            metadata=result.metadata,
        )
    return generation_success, output_location_to_db


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
    parser.add_argument("--queue-workers", type=int, default=1)
    parser.add_argument("--preload-model", type=str, default="")
    parser.add_argument("--db-type", type=str, default="supabase")
    parser.add_argument("--supabase-url", type=str, default="https://wczysqzxlwdndgxitrvc.supabase.co")
    parser.add_argument("--reigh-access-token", type=str, default=None, help="Access token for Reigh API (preferred)")
    parser.add_argument("--supabase-access-token", type=str, default=None, help="Legacy alias for --reigh-access-token")
    parser.add_argument("--supabase-anon-key", type=str, default=None, help="Supabase anon key (set via env SUPABASE_ANON_KEY)")
    
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
    load_dotenv()

    cli_args = parse_args()

    # Resolve access token: prefer --reigh-access-token, fall back to --supabase-access-token
    access_token = cli_args.reigh_access_token or cli_args.supabase_access_token
    if not access_token:
        print("Error: Worker authentication credential is required", file=sys.stderr)
        sys.exit(1)

    # Auto-derive worker_id when not explicitly provided
    if not cli_args.worker:
        cli_args.worker = os.getenv("RUNPOD_POD_ID") or "local-worker"
    os.environ["WORKER_ID"] = cli_args.worker
    os.environ["WAN2GP_WORKER_MODE"] = "true"

    global debug_mode
    debug_mode = cli_args.debug
    if debug_mode:
        enable_debug_mode()
        # Set Wan2GP verbose level so SVI and other debug logs appear
        try:
            from mmgp import offload
            offload.default_verboseLevel = 2
        except ImportError:
            pass  # mmgp not yet available, will be set when model loads
        if not cli_args.save_logging:
            # Automatically save logs to debug/ directory if debug mode is on
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

    # Supabase Setup
    try:
        # Service key takes priority (RunPod workers), then access token, then anon key (local workers)
        client_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or access_token or cli_args.supabase_anon_key or os.getenv("SUPABASE_ANON_KEY")
        if not client_key: raise ValueError("No Supabase key found")
        
        db_config.DB_TYPE = "supabase"
        db_config.PG_TABLE_NAME = os.getenv("POSTGRES_TABLE_NAME", "tasks")
        db_config.SUPABASE_URL = cli_args.supabase_url
        db_config.SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        db_config.SUPABASE_VIDEO_BUCKET = os.getenv("SUPABASE_VIDEO_BUCKET", "image_uploads")
        db_config.SUPABASE_CLIENT = create_client(cli_args.supabase_url, client_key)
        db_config.SUPABASE_ACCESS_TOKEN = access_token
        db_config.debug_mode = debug_mode

        # Propagate to os.environ for code that can't import db_operations
        # (fatal_error_handler during crashes, headless_wgp notify_model_switch)
        os.environ["SUPABASE_URL"] = cli_args.supabase_url

        # Validate config after initialization
        config_errors = db_config.validate_config()
        if config_errors:
            for err in config_errors:
                headless_logger.warning(f"[CONFIG] {err}")

    except (ValueError, OSError, KeyError) as e:
        headless_logger.critical(f"Supabase init failed: {e}")
        sys.exit(1)

    if cli_args.migrate_only:
        sys.exit(0)

    main_output_dir = Path(cli_args.main_output_dir).resolve()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    # Centralized Logging
    # Global reference to log interceptor for setting current task context
    _log_interceptor_instance = None
    
    if cli_args.worker:
        guardian_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or client_key
        guardian_process, log_queue = start_heartbeat_guardian_process(cli_args.worker, cli_args.supabase_url, guardian_key)
        _global_log_buffer = LogBuffer(max_size=100, shared_queue=log_queue)
        _log_interceptor_instance = CustomLogInterceptor(_global_log_buffer)
        set_log_interceptor(_log_interceptor_instance)

    # Apply WGP Overrides
    wan_dir = str((Path(__file__).parent / "Wan2GP").resolve())
    original_cwd = os.getcwd()
    original_argv = sys.argv[:]
    try:
        os.chdir(wan_dir)
        sys.path.insert(0, wan_dir)
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

    except (ImportError, RuntimeError, AttributeError, KeyError) as e:
        headless_logger.critical(f"WGP import failed: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    # Clean up legacy collision-prone LoRA files
    cleanup_legacy_lora_collisions()

    # Initialize Task Queue
    try:
        task_queue = HeadlessTaskQueue(
            wan_dir=wan_dir,
            max_workers=cli_args.queue_workers,
            debug_mode=debug_mode,
            main_output_dir=str(main_output_dir)
        )
        preload_model = cli_args.preload_model if cli_args.preload_model else None
        task_queue.start(preload_model=preload_model)
    except (RuntimeError, ValueError, OSError) as e:
        headless_logger.critical(f"Queue init failed: {e}")
        sys.exit(1)

    headless_logger.essential(f"Worker {cli_args.worker or 'anonymous'} started. Polling every {cli_args.poll_interval}s.")

    try:
        while True:
            task_info = db_ops.get_oldest_queued_task_supabase(worker_id=cli_args.worker)

            if not task_info:
                time.sleep(cli_args.poll_interval)
                continue

            current_task_params = task_info["params"]
            current_task_type = task_info["task_type"]
            current_project_id = task_info.get("project_id")
            current_task_id = task_info["task_id"]

            if current_project_id is None and current_task_type in {"travel_orchestrator", "edit_video_orchestrator"}:
                db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_FAILED, "Orchestrator missing project_id")
                continue

            # Ensure params["task_id"] is the DB UUID (not the human-readable params.task_id)
            # process_single_task reads task_id from params, so this must be set for ALL task types
            current_task_params["task_id"] = current_task_id
            if "orchestrator_details" in current_task_params:
                current_task_params["orchestrator_details"]["orchestrator_task_id"] = current_task_id

            # Set current task context for log interceptor so all logs are associated with this task
            if _log_interceptor_instance:
                _log_interceptor_instance.set_current_task(current_task_id)

            from source.core.params.task_result import TaskResult, TaskOutcome
            raw_result = process_single_task(
                current_task_params, main_output_dir, current_task_type, current_project_id,
                image_download_dir=current_task_params.get("segment_image_download_dir"),
                colour_match_videos=cli_args.colour_match_videos,
                mask_active_frames=cli_args.mask_active_frames,
                task_queue=task_queue
            )

            # Unpack: raw_result is either a TaskResult or a legacy (bool, str) tuple
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
                        db_ops.update_task_status_supabase(
                            current_task_id, db_ops.STATUS_COMPLETE,
                            result.output_path, result.thumbnail_url)
                    elif result and result.outcome == TaskOutcome.ORCHESTRATING:
                        db_ops.update_task_status(current_task_id, db_ops.STATUS_IN_PROGRESS, result.output_path)
                    elif isinstance(output_location, str) and output_location.startswith("[ORCHESTRATOR_COMPLETE]"):
                        # Legacy string prefix parsing (backward compat during migration)
                        actual_output = output_location.replace("[ORCHESTRATOR_COMPLETE]", "")
                        thumbnail_url = None
                        try:
                            import json
                            data = json.loads(actual_output)
                            actual_output = data.get("output_location", actual_output)
                            thumbnail_url = data.get("thumbnail_url")
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass
                        db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_COMPLETE, actual_output, thumbnail_url)
                    else:
                        db_ops.update_task_status(current_task_id, db_ops.STATUS_IN_PROGRESS, output_location)
                else:
                    db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_COMPLETE, output_location)

                    # Note: Orchestrator completion is handled by the complete-task Edge Function
                    # based on checking if all child tasks are complete.

                    cleanup_generated_files(output_location, current_task_id, debug_mode)
            else:
                # Task failed - check if this is a retryable error
                error_message = (result.error_message if result else output_location) or "Unknown error"
                is_retryable, error_category, max_attempts = is_retryable_error(error_message)
                current_attempts = task_info.get("attempts", 0)
                
                if is_retryable and current_attempts < max_attempts:
                    # Requeue for retry
                    headless_logger.warning(
                        f"Task {current_task_id} failed with retryable error ({error_category}), "
                        f"requeuing for retry (attempt {current_attempts + 1}/{max_attempts})"
                    )
                    db_ops.requeue_task_for_retry(
                        current_task_id, 
                        error_message, 
                        current_attempts, 
                        error_category
                    )
                else:
                    # Permanent failure - either not retryable or exhausted retries
                    if is_retryable and current_attempts >= max_attempts:
                        headless_logger.error(
                            f"Task {current_task_id} exhausted {max_attempts} retry attempts for {error_category}"
                        )
                    db_ops.update_task_status_supabase(current_task_id, db_ops.STATUS_FAILED, output_location)
            
            # Clear task context from log interceptor
            if _log_interceptor_instance:
                _log_interceptor_instance.set_current_task(None)
            
            time.sleep(1)

    except FatalWorkerError as e:
        headless_logger.critical(f"Fatal Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        headless_logger.essential("Shutting down...")
    finally:
        if cli_args.worker:
            heartbeat_stop_event.set()
        if task_queue:
            task_queue.stop()

if __name__ == "__main__":
    main() 
