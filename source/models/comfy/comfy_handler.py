"""
ComfyUI Task Handler

Handles ComfyUI workflow tasks with lazy-loading of the ComfyUI server.
Integrates with the TaskRegistry for seamless task routing.
"""

import json
import asyncio
from pathlib import Path
from typing import Tuple, Optional
import httpx

from source.core.log import headless_logger, model_logger
from source.models.comfy.comfy_utils import ComfyUIManager, ComfyUIClient, COMFY_PATH, COMFY_PORT

# Global ComfyUI manager (lazy-loaded on first comfy task)
_comfy_manager: Optional[ComfyUIManager] = None
_comfy_startup_failed: bool = False
_comfy_startup_lock = asyncio.Lock()

async def _ensure_comfy_running() -> bool:
    """
    Ensure ComfyUI is running. Starts it on first call (lazy-loading).

    Returns:
        True if ComfyUI is ready, False if startup failed
    """
    global _comfy_manager, _comfy_startup_failed

    # If already failed, don't retry
    if _comfy_startup_failed:
        return False

    # If already running, return True
    if _comfy_manager is not None:
        return True

    # First time - start ComfyUI (with lock to prevent races)
    async with _comfy_startup_lock:
        # Double-check after acquiring lock
        if _comfy_manager is not None:
            return True

        if _comfy_startup_failed:
            return False

        try:
            headless_logger.debug("First ComfyUI task detected - starting ComfyUI server...")

            # Check if ComfyUI exists
            comfy_main = Path(COMFY_PATH) / "main.py"
            if not comfy_main.exists():
                raise FileNotFoundError(f"ComfyUI not found at {COMFY_PATH}")

            # Start ComfyUI
            manager = ComfyUIManager(COMFY_PATH, COMFY_PORT)
            manager.start()

            # Wait for ready
            async with httpx.AsyncClient() as client:
                if not await manager.wait_for_ready(client, timeout=120):
                    raise Exception("ComfyUI failed to become ready")

            _comfy_manager = manager
            headless_logger.debug("ComfyUI started successfully and is ready for tasks")
            return True

        except FileNotFoundError as e:
            headless_logger.error(f"ComfyUI not available: {e}")
            _comfy_startup_failed = True
            return False
        except (httpx.HTTPError, OSError, RuntimeError) as e:
            headless_logger.error(f"ComfyUI startup failed: {e}")
            _comfy_startup_failed = True
            return False

def handle_comfy_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    task_id: str) -> Tuple[bool, Optional[str]]:
    """
    Handle ComfyUI workflow tasks.

    Args:
        task_params_from_db: Task parameters from database
        main_output_dir_base: Base output directory
        task_id: Task ID

    Returns:
        (success, output_path) - Standard task handler signature
    """
    model_logger.debug(f"Processing ComfyUI task {task_id}")

    try:
        # Parse params
        params = task_params_from_db
        if isinstance(params, str):
            params = json.loads(params)

        workflow = params.get("workflow")
        if not workflow:
            return False, "Missing required parameter: workflow"

        if isinstance(workflow, str):
            workflow = json.loads(workflow)

        video_url = params.get("video_url")
        video_node_id = params.get("video_node_id")
        video_input_field = params.get("video_input_field", "video")

        # Run async processing
        async def _process():
            # Ensure ComfyUI is running (lazy-start on first use)
            if not await _ensure_comfy_running():
                raise Exception(
                    "ComfyUI is not available on this worker. "
                    "Ensure ComfyUI is installed at COMFY_PATH or use a worker with ComfyUI enabled."
                )

            comfy_client = ComfyUIClient()

            async with httpx.AsyncClient(timeout=300.0) as client:
                # Download and upload video if provided
                if video_url:
                    model_logger.debug(f"Downloading video from {video_url}")

                    response = await client.get(video_url)
                    response.raise_for_status()
                    video_bytes = response.content

                    # Upload to ComfyUI
                    uploaded_filename = await comfy_client.upload_video(
                        client, video_bytes, "input.mp4"
                    )
                    model_logger.debug(f"Uploaded video to ComfyUI: {uploaded_filename}")

                    # Inject into workflow
                    if video_node_id and video_node_id in workflow:
                        workflow[video_node_id]['inputs'][video_input_field] = uploaded_filename

                # Submit workflow
                model_logger.debug(f"Submitting workflow to ComfyUI")
                prompt_id = await comfy_client.queue_workflow(client, workflow)
                model_logger.debug(f"Workflow queued with prompt_id: {prompt_id}")

                # Wait for completion
                model_logger.debug(f"Waiting for workflow to complete...")
                history = await comfy_client.wait_for_completion(client, prompt_id)
                model_logger.debug(f"Workflow completed successfully")

                # Download outputs
                outputs = await comfy_client.download_output(client, history)

                if not outputs:
                    raise Exception("No outputs generated by workflow")

                model_logger.debug(f"Downloaded {len(outputs)} output file(s)")
                return outputs[0], prompt_id

        # Execute async operations
        output, prompt_id = asyncio.run(_process())

        # Save output to file
        output_dir = main_output_dir_base / "comfy"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_id}_{output['filename']}"

        with open(output_path, 'wb') as f:
            f.write(output['content'])

        model_logger.debug(f"Saved output to: {output_path}")

        # Return in standard format (success, path)
        return True, str(output_path)

    except (httpx.HTTPError, OSError, json.JSONDecodeError, ValueError,
            RuntimeError, TimeoutError) as e:
        error_msg = f"ComfyUI task failed: {str(e)}"
        model_logger.debug(error_msg)
        headless_logger.error(error_msg, task_id=task_id, exc_info=True)
        return False, error_msg

def shutdown_comfy():
    """Shutdown ComfyUI server (called on worker exit)."""
    global _comfy_manager

    if _comfy_manager is not None:
        headless_logger.debug("Shutting down ComfyUI server")
        _comfy_manager.stop()
        _comfy_manager = None
