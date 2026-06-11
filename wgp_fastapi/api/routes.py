"""
FastAPI application with routes for text-to-image, image-to-image, and image-to-video generation.
"""

import glob
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
import threading

# Import WanGPSession from shared.api
import sys

from numpy import asarray
from starlette.responses import Response

from wgp_fastapi import upscaler

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

output_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "outputs")
)

from wgp_fastapi.models import (
    ImageToVideoRequest,
    FluxImageRequest,
    FluxImageResponse,
    TaskStatus,
    FluxImageModel,
    MagicMaskResponse,
)

app = FastAPI(
    title="Wan2GP API",
    description="FastAPI wrapper for Wan2GP - Text-to-Image, Image-to-Image, and Image-to-Video generation",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session instance and task tracking
_wgp_session: Optional["WanGPSession"] = None
_tasks: dict = {}  # task_id -> {"status": str, "result": GenerationResult, "job": SessionJob, "settings": dict}

# Async queue system
_async_queue: list = []  # Queue of pending task_id in order
_current_task_id: Optional[str] = None  # Currently executing task
_worker_running: bool = False  # Worker thread flag
_worker_lock = threading.Lock()

# Task states
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_SUCCESS = "success"
TASK_FAILED = "failed"

# Status tracking for /status endpoint
_download_in_progress: bool = False  # Whether a model is currently being downloaded
_model_loading: bool = False         # Whether a model is currently being loaded
_model_load_message: str = ""        # Human-readable status message for model loading


def get_wgp_session():
    """Get or create the WanGPSession instance."""
    global _wgp_session, _model_loading, _model_load_message

    # Import here to avoid issues with module loading order
    from shared.api import WanGPSession

    if _wgp_session is None:
        _model_loading = True
        _model_load_message = "Initializing WanGP runtime..."
        _wgp_session = WanGPSession(
            root=str(project_root),
            console_output=True,
            console_isatty=False,
        ).ensure_ready()
        _model_loading = False
        _model_load_message = ""

    return _wgp_session


def _process_queue_worker():
    """Background worker that processes the async queue one task at a time."""
    global _current_task_id, _worker_running, _model_loading, _model_load_message

    with _worker_lock:
        if _worker_running:
            return
        _worker_running = True

    try:
        while True:
            # Get next task from queue
            with _worker_lock:
                if not _async_queue:
                    _current_task_id = None
                    break
                task_id = _async_queue.pop(0)
                _current_task_id = task_id

            task = _tasks.get(task_id)
            if not task:
                continue

            # Mark as running
            task["status"] = TASK_RUNNING

            # Set model loading state at task start; will be cleared once
            # we receive first progress/preview event (loading → generating)
            task_loading = True
            _model_loading = True
            _model_load_message = "Loading model..."

            try:
                session = get_wgp_session()
                print(
                    f"[QUEUE WORKER] Submitting task {task_id} with settings: {task['settings']}"
                )
                # Submit and store job reference for status polling
                job = session.submit_task(task["settings"])
                print(f"[QUEUE WORKER] Got job: {job}, job.done: {job.done}")
                task["job"] = job

                # Poll for preview events while waiting for completion
                while not job.done:
                    try:
                        event = job.events.get(timeout=0.1)
                        if event:
                            if event.kind == "preview":
                                task["latest_preview"] = event.data
                                if task_loading:
                                    task_loading = False
                                    _model_loading = False
                                    _model_load_message = ""
                            elif event.kind == "progress":
                                task["latest_progress"] = event.data
                                if task_loading:
                                    task_loading = False
                                    _model_loading = False
                                    _model_load_message = ""
                            elif event.kind == "status":
                                # Update model load message from generation status
                                if task_loading and event.data:
                                    _model_load_message = str(event.data)
                    except:
                        pass

                print(f"[QUEUE WORKER] Job done, getting result...")
                result = job.result()
                print(
                    f"[QUEUE WORKER] Got result: success={result.success}, files={result.generated_files}, errors={[e.message for e in result.errors]}"
                )

                task["result"] = result
                task["status"] = TASK_SUCCESS if result.success else TASK_FAILED
            except Exception as e:
                import traceback

                print(f"Queue worker error for task {task_id}: {e}")
                print(traceback.format_exc())
                from shared.api import GenerationResult

                task["result"] = GenerationResult(
                    success=False,
                    generated_files=[],
                    errors=[],
                    total_tasks=0,
                    successful_tasks=0,
                    failed_tasks=1,
                )
                task["status"] = TASK_FAILED
            finally:
                # If the task errored before producing progress/preview, clear loading state
                if task_loading:
                    _model_loading = False
                    _model_load_message = ""
    finally:
        with _worker_lock:
            _worker_running = False
            _current_task_id = None
            _model_loading = False
            _model_load_message = ""


def _queue_task(settings: dict) -> str:
    """Add a task to the queue and start worker if needed. Returns task_id."""
    import uuid

    # Generate task_id and add to settings
    task_id = str(uuid.uuid4())
    settings = dict(settings)  # Don't mutate original
    settings["id"] = task_id

    # Add task to tracking - accept settings dict directly
    _tasks[task_id] = {
        "status": TASK_PENDING,
        "result": None,
        "job": None,
        "settings": settings,
    }

    # Add to queue
    _async_queue.append(task_id)

    # Start worker in background if not running
    with _worker_lock:
        if not _worker_running:
            thread = threading.Thread(
                target=_process_queue_worker,
                daemon=True,
                name="flux-queue-worker",
            )
            thread.start()

    return task_id


# Backwards compatibility wrapper for flux-specific usage
def _queue_flux_task(
    flux_request: "FluxImageRequest", image_start_path: str | None = None
) -> str:
    """Queue a flux image task (backwards compatibility wrapper)."""
    return _queue_task(
        flux_request.to_wgp_settings(image_start_path=image_start_path),
    )


def get_save_path() -> str:
    """Get the save path from the session."""
    session = get_wgp_session()
    # Try to get the actual save_path from the runtime module
    try:
        runtime = session._ensure_runtime()
        module = runtime.module
        save_p = getattr(module, "save_path", None)
        if save_p:
            return str(save_p)
    except Exception:
        pass
    # Fall back to config or default
    config = getattr(session, "_output_dir", None)
    if config:
        return str(config)
    # Default to project root/output
    return str(project_root / "output")


def build_file_url(request: Request, file_path: str) -> str:
    """Build a full URL for a generated file."""
    base_url = str(request.base_url).rstrip("/")
    save_path = get_save_path()

    # Use pathlib for cross-platform path handling
    from pathlib import Path

    file_p = Path(file_path)

    # Get relative path - works cross-platform
    if file_p.is_absolute():
        try:
            rel_path = str(file_p.relative_to(Path(save_path)))
        except ValueError:
            # File is in different directory - just use basename
            rel_path = file_p.name
    else:
        rel_path = file_path

    # Normalize path separators for URL
    return f"{base_url}/files/{rel_path.replace(os.sep, '/')}"


def save_upload_file(upload_file: UploadFile, suffix: str = ".png") -> str:
    """Save an uploaded file to a temp directory and return the path.

    Automatically converts non-JPEG images (HEIC, HEIF, etc.) to JPEG using Pillow.
    """
    # Register HEIF/HEIC opener for Pillow
    from pillow_heif import register_heif_opener

    register_heif_opener()

    from PIL import Image
    import io

    content = upload_file.file.read()

    # Check if image needs conversion to JPEG
    image = Image.open(io.BytesIO(content))
    if image.format not in ("JPEG", "JPG", "JPEG"):
        # Convert to RGB (handles RGBA, palette, etc.) and save as JPEG
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        # Save as JPEG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name, "JPEG")
            return tmp.name

    # Original behavior for JPEG files
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global _wgp_session
    if _wgp_session is not None:
        _wgp_session.close()
        _wgp_session = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status")
async def server_status():
    """Get the current server status, including whether the server is
    downloading models, loading models, or idle/generating."""
    from shared.utils.download import download_in_progress as shared_download_in_progress
    global _download_in_progress, _model_loading, _model_load_message

    # Check if models are being downloaded (either via magic_mask or via download_models/download_file)
    any_download = _download_in_progress or shared_download_in_progress

    # Determine the overall server status
    if any_download:
        status = "downloading"
    elif _model_loading:
        status = "loading"
    elif _current_task_id is not None or _worker_running:
        status = "generating"
    else:
        status = "idle"

    return {
        "status": status,
        "download_in_progress": any_download,
        "loading_in_progress": _model_loading,
        "generation_in_progress": _current_task_id is not None or _worker_running,
        "current_task_id": _current_task_id,
        "message": _model_load_message or "",
    }


@app.post(
    "/api/v1/upscale",
    summary="Text to Image Generation",
    description="Generate images from text prompts. Supports Flux 1 Kontext, Flux 1 Dev, and Flux 2 Klein.",
)
async def upscale(scale: int, image: UploadFile = File(None)):
    from PIL import Image

    # save the image to a temporary path
    image_path = save_upload_file(image, suffix=".png")

    return Response(upscaler.upscale(image_path, scale))


@app.post(
    "/api/v1/i2v",
    summary="Image to Video Generation",
    description="Generate videos from input images and text prompts. Only LTX 2 Video is supported. Returns task_id immediately - poll /api/v1/tasks/{task_id} for status.",
)
async def image_to_video(
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    batch_size: int,
    model: str,
    video_length: int,
    guidance_scale: float,
    fps: int,
    image: UploadFile = File(None),
):
    image_path = None

    try:
        # Handle image upload
        if image is not None:
            image_path = save_upload_file(image, suffix=".png")

        # Create request object
        from wgp_fastapi.models.i2v import I2VVideoModel

        model_enum = I2VVideoModel(model)

        request_obj = ImageToVideoRequest(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=batch_size,
            model=model_enum,
            video_length=video_length,
            guidance_scale=guidance_scale,
            fps=fps
        )

        # Convert request to WanGP settings
        settings = request_obj.to_wgp_settings(image_start_path=image_path)

        # Is this not being setup correctly?
        print(f"Settings: {settings}")

        # Queue the task and return immediately
        task_id = _queue_task(settings)

        # Return just the task_id immediately for async
        return JSONResponse(content={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve generated files."""
    from pathlib import Path

    save_path = get_save_path()
    save_path_p = Path(save_path)

    # Try multiple possible locations
    possible_paths = [
        save_path_p / file_path,
        Path(project_root) / "outputs" / file_path,
        Path("outputs") / file_path,
    ]

    for requested_path in possible_paths:
        requested_path = requested_path.resolve()
        if requested_path.exists():
            return FileResponse(requested_path)

    # Debug: list what directories exist
    debug_info = {
        "save_path": str(save_path),
        "save_path_exists": save_path_p.exists(),
        "project_root": str(project_root),
        "outputs_exists": (Path(project_root) / "outputs").exists(),
        "trying_file": file_path,
    }
    raise HTTPException(status_code=404, detail=f"File not found. Debug: {debug_info}")


@app.post(
    "/api/v1/flux-image",
    response_model=FluxImageResponse,
    summary="Flux Image Generation",
    description="Generate images using Flux 2 Klein 9B model with all available parameters. Supports text-to-image and image-to-image generation. Optionally accepts a mask for inpainting and an inpaint-reference image.",
)
async def flux_image(
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    batch_size: int,
    model: FluxImageModel = FluxImageModel.FLUX_2_KLEIN,
    image_prompt_type: str = "I",
    image: UploadFile = File(None),
    activated_loras: Optional[str] = Form(None),
    mask: UploadFile = File(None),
    inpaint_reference: UploadFile = File(None),
) -> FluxImageResponse:
    # Map input to object
    flux_request = FluxImageRequest(
        prompt=prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        # batch size internally used for queueing more than 1 request
        batch_size=1,
        model=model,
        image_prompt_type=image_prompt_type,
        activated_loras=activated_loras,
    )

    image_path: str | None = None
    mask_path: str | None = None
    inpaint_reference_path: str | None = None

    try:
        try:
            # Handle image upload
            if image is not None:
                image_path = save_upload_file(image, suffix=".png")
        except Exception as e:
            print(f"Unable to process the image: {e}")

        try:
            # Handle mask upload
            if mask is not None:
                mask_path = save_upload_file(mask, suffix=".png")
        except Exception as e:
            print(f"Unable to process the mask: {e}")

        try:
            # Handle inpaint-reference upload
            if inpaint_reference is not None:
                inpaint_reference_path = save_upload_file(inpaint_reference, suffix=".png")
        except Exception as e:
            print(f"Unable to process the inpaint-reference: {e}")

        # Set mask and inpaint-reference paths on the request
        flux_request.mask_path = mask_path
        flux_request.inpaint_reference_path = inpaint_reference_path

        # first to be processed
        task_id = _queue_flux_task(flux_request, image_path)

        # queue up the rest
        for i in range(1, batch_size - 1):
            _queue_flux_task(flux_request, image_path)

        # Return just the task_id immediately for async - bypass Pydantic serialization
        return JSONResponse(content={"task_id": task_id})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Print to console so it shows in run.bat terminal
        print(f"[ERROR] flux-image endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "image_uploaded": image is not None,
                "mask_uploaded": mask is not None,
                "inpaint_reference_uploaded": inpaint_reference is not None,
            }
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.get(
    "/api/v1/flux-image/{task_id}",
    response_model=FluxImageResponse,
    summary="Get task status",
    description="Get the status of an async flux-image generation task.",
)
async def get_flux_image_task(request: Request, task_id: str):
    """Get the status of an async flux-image task."""
    from shared.api import GenerationResult

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    status = task["status"]

    # Task still running
    if status == TASK_PENDING:
        return FluxImageResponse(
            status=TASK_PENDING,
            task_id=task_id,
            progress=0,
        )

    if status == TASK_RUNNING:
        # Estimate progress - check the job
        job = task.get("job")
        if job and hasattr(job, "done") and not job.done:
            # Job in progress - can't get detailed progress easily
            # Could add more detailed progress tracking here
            return FluxImageResponse(
                status=TASK_RUNNING,
                task_id=task_id,
                progress=50,  # Middle of generation
            )
        return FluxImageResponse(
            status=TASK_RUNNING,
            task_id=task_id,
            progress=50,
        )

    # Task not found or still pending/running
    return FluxImageResponse(
        status=TASK_RUNNING,
        task_id=task_id,
        progress=50,
    )


@app.post(
    "/api/v1/magic-mask",
    response_model=MagicMaskResponse,
    summary="Magic Mask Generation",
    description="Generate a segmentation mask from an image using text prompts. Returns the masked image URL directly.",
)
async def magic_mask(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(None),
    no_hole: bool = Form(True),
    negative_mask: bool = Form(False),
) -> MagicMaskResponse:
    """Generate a mask from an image using keyword prompts and return the mask image URL."""
    from PIL import Image

    if image is None:
        raise HTTPException(status_code=400, detail="An image file is required.")

    try:
        # Save uploaded file and open as PIL Image
        image_path = save_upload_file(image, suffix=".png")
        pil_image = Image.open(image_path)

        # Lazy import magic_mask to avoid loading order issues
        from shared import magic_mask as mm

        # Auto-download SAM3 model assets if missing
        global _download_in_progress

        from shared.utils.download import process_files_def

        _download_in_progress = True
        try:
            process_files_def(**mm.query_download_def())
        finally:
            _download_in_progress = False

        # Generate the mask
        background, mask_image, keywords = mm.generate_image_mask(
            pil_image,
            prompt,
            no_hole=no_hole,
            negative_mask=negative_mask,
        )

        # Check if the mask is effectively blank — SAM3 couldn't find anything
        if mask_image.getextrema()[1] <= 5:
            keywords_label = ", ".join(keywords)
            raise HTTPException(
                status_code=422,
                detail=f"SAM3 was unable to generate a mask for '{keywords_label}' on this image. "
                "Try different keywords or a different image.",
            )

        # Save the mask to outputs dir with mask- prefix
        import uuid

        mask_filename = f"mask-{uuid.uuid4()}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        os.makedirs(output_dir, exist_ok=True)
        mask_image.save(mask_path, "PNG")

        # Build the overlay: original image + white mask composited on top
        base_rgba = background.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
        overlay.putalpha(mask_image)
        composited = Image.alpha_composite(base_rgba, overlay)

        overlay_filename = f"mask-overlay-{uuid.uuid4()}.png"
        overlay_path = os.path.join(output_dir, overlay_filename)
        composited.save(overlay_path, "PNG")

        # Build the full URLs
        image_url = build_file_url(request, mask_path)
        overlay_url = build_file_url(request, overlay_path)

        return MagicMaskResponse(
            image_url=image_url,
            maskOverlay=overlay_url,
            keywords=keywords,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] magic-mask endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "image_uploaded": image is not None,
            },
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.get(
    "/api/v1/queue",
    summary="Get queue info",
    description="Get the current queue information.",
)
async def get_queue():
    """Get current queue info including all queued task IDs."""
    return {
        "current_task_id": _current_task_id,
        "queue": list(_async_queue),
        "queue_size": len(_async_queue),
    }


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskStatus,
    summary="Get task status",
    description="Retrieve the status of a generation task.",
)
async def get_task_status(request: Request, task_id: str) -> TaskStatus:
    """Get the status of a generation task."""
    from shared.api import GenerationResult

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    status = task["status"]
    settings = task.get("settings", {})
    seed = settings.get("seed")

    # Task still pending
    if status == TASK_PENDING:
        return TaskStatus(
            progress=0,
            preview_image=None,
            finished_image=None,
            seed=seed,
        )

    if status == TASK_RUNNING:
        # Check for progress from stored values (updated by worker) or job events
        progress = 5  # Start at 5% when running but no progress yet
        preview_image = None

        # First check stored preview/progress from worker
        latest_preview = task.get("latest_preview")
        latest_progress = task.get("latest_progress")

        if not latest_preview or not latest_progress:
            # Fall back to polling job events
            job = task.get("job")
            if job:
                events = getattr(job, "events", None)
                if events:
                    try:
                        while True:
                            event = events.get(timeout=0.001)
                            if event is None:
                                break
                            if event.kind == "progress":
                                latest_progress = event.data
                            elif event.kind == "preview":
                                latest_preview = event.data
                    except:
                        pass

        if latest_progress and hasattr(latest_progress, "progress"):
            progress = latest_progress.progress

        if latest_preview and hasattr(latest_preview, "image") and latest_preview.image:
            # Convert PIL image to base64 for preview
            import io
            import base64
            from PIL import Image

            img_buffer = io.BytesIO()
            latest_preview.image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            preview_image = (
                f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
            )

        return TaskStatus(
            progress=progress,
            preview_image=preview_image,
            finished_image=None,
            seed=seed,
        )

    # Task completed - check result
    result: GenerationResult = task.get("result")
    settings = task.get("settings", {})
    seed = settings.get("seed")

    if result and result.success and result.generated_files:
        # Use first generated file as finished_image URL
        finished_file_url = build_file_url(request, result.generated_files[0])
        return TaskStatus(
            progress=100,
            preview_image=None,
            finished_image=finished_file_url,
            seed=seed,
        )
    else:
        return TaskStatus(
            progress=100,
            preview_image=None,
            finished_image=None,
            seed=seed,
        )


@app.delete(
    "/api/v1/tasks/{task_id}",
    summary="Cancel task",
    description="Cancel a running or queued generation task.",
)
async def cancel_task(task_id: str):
    """Cancel a running or queued generation task."""
    from shared.api import GenerationResult

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    status = task["status"]

    # Already completed — nothing to cancel
    if status in (TASK_SUCCESS, TASK_FAILED):
        return JSONResponse(
            content={"status": "already_completed", "task_id": task_id},
            status_code=200,
        )

    # Pending task — remove from async queue
    if status == TASK_PENDING:
        with _worker_lock:
            if task_id in _async_queue:
                _async_queue.remove(task_id)
        task["status"] = TASK_FAILED
        task["result"] = GenerationResult(
            success=False,
            generated_files=[],
            errors=[],
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=1,
        )
        return JSONResponse(
            content={"status": "cancelled", "task_id": task_id},
            status_code=200,
        )

    # Running task — cancel via session
    if status == TASK_RUNNING:
        session = get_wgp_session()
        session.cancel()
        return JSONResponse(
            content={"status": "cancelling", "task_id": task_id},
            status_code=200,
        )

    # Fallback
    raise HTTPException(status_code=500, detail="Unexpected task state")


@app.delete(
    "/api/v1/files/{file_name:path}",
    summary="Delete a file",
    description="Delete a file from the outputs directory by filename.",
)
async def delete_file(file_name: str):
    # find the file path given just the file name alone
    for file in glob.iglob(os.path.join(f"{output_dir}", "**", "*"), recursive=True):
        if file_name in file:
            print(f"Deleting file: {file}")

            if os.name == "nt" or sys.platform == "darwin":
                os.remove(file)
            else:
                os.remove(file.replace("/", "\\\\"))

            return Response(status_code=200)

    # file was not found
    return Response(status_code=404)


@app.delete(
    "/api/v1/files",
    summary="Delete files",
    description="Delete one or more files from the outputs directory by filename. "
    "Accepts a JSON array of filenames in the request body. "
    "Supports all file extension types.",
)
async def delete_files(
    file_names: List[str] = Body(
        ...,
        description="List of filenames to delete (e.g., [\"image.png\", \"video.mp4\"])",
        examples=[["image.png", "video.mp4"]],
    ),
):
    """
    Delete one or more files. Accepts a JSON body with a list of filenames.
    Example body: ["image1.png", "video.mp4", "subfolder/image.jpg"]
    Supports all file extension types.
    """
    if not file_names:
        return JSONResponse(
            content={"error": "No filenames provided. Send a JSON array of filenames."},
            status_code=400,
        )

    results: dict[str, list] = {"deleted": [], "not_found": [], "errors": []}

    # Collect all files once to avoid re-walking the directory
    all_files = list(
        glob.iglob(os.path.join(f"{output_dir}", "**", "*"), recursive=True)
    )

    for file_name in file_names:
        matched = False
        for file_path in all_files:
            if file_name in file_path:
                try:
                    os.remove(file_path)
                    results["deleted"].append(file_name)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    results["errors"].append(
                        {"file": file_name, "detail": str(e)}
                    )
                    print(f"Error deleting {file_path}: {e}")
                matched = True
                break  # only delete the first match per filename

        if not matched:
            results["not_found"].append(file_name)

    if not results["deleted"] and not results["errors"]:
        return JSONResponse(content=results, status_code=404)

    return JSONResponse(content=results, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
