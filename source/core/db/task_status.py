"""
Task status updates, failure marking, requeue, and billing timestamp resets.
"""
import os
import json
from dataclasses import dataclass
from pathlib import Path

from postgrest.exceptions import APIError

from source.core.log import headless_logger

__all__ = [
    "requeue_task_for_retry",
    "update_task_status",
    "update_task_status_supabase",
    "TaskStatusUpdateResult",
    "mark_task_failed_supabase",
    "reset_generation_started_at",
]

from . import config as _cfg
from source.core.constants import BYTES_PER_MB
from .config import (
    STATUS_QUEUED,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETE,
    STATUS_FAILED,
    EDGE_FAIL_PREFIX,
    _log_thumbnail)
from .edge_helpers import _call_edge_function_with_retry


@dataclass(frozen=True)
class TaskStatusUpdateResult:
    outcome: str
    output_location: str | None = None
    public_url: str | None = None
    thumbnail_url: str | None = None
    error: str | None = None

def _extract_video_thumbnail(video_path: Path, task_id_str: str) -> Path | None:
    """Extract a thumbnail from a video file.

    Checks for an existing poster (.jpg alongside the video) first.
    Falls back to extracting the first frame via OpenCV.

    Returns:
        Path to the thumbnail file (caller must clean up temp files), or None.
        If the returned path equals ``video_path.with_suffix('.jpg')``, it is
        an existing file and should NOT be deleted by the caller.
    """
    import tempfile
    try:
        from ...utils import save_frame_from_video
        import cv2
    except ImportError:
        _log_thumbnail("Cannot import cv2/save_frame_from_video for thumbnail extraction",
                        level="warning", task_id=task_id_str)
        return None

    # Check for existing poster alongside the video
    existing_poster_path = video_path.with_suffix('.jpg')
    if existing_poster_path.exists():
        _log_thumbnail(f"Found existing poster: {existing_poster_path.name}", level="info", task_id=task_id_str)
        return existing_poster_path

    # Extract first frame from video
    _log_thumbnail(f"Extracting first frame from video {video_path.name}", task_id=task_id_str)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
        temp_frame_path = Path(temp_frame.name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _log_thumbnail(f"Could not open video with cv2: {video_path}", level="warning", task_id=task_id_str)
        temp_frame_path.unlink(missing_ok=True)
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if save_frame_from_video(video_path, 0, temp_frame_path, (width, height)):
        _log_thumbnail("First frame extracted successfully", level="info", task_id=task_id_str)
        return temp_frame_path

    _log_thumbnail(f"save_frame_from_video failed for {video_path.name}", level="warning", task_id=task_id_str)
    temp_frame_path.unlink(missing_ok=True)
    return None

def _mark_task_failed_via_edge_function(task_id_str: str, error_message: str):
    """Mark a task as failed using the update-task-status Edge Function (with retry)."""
    edge_url = (
        os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url:
        headless_logger.error(f"No update-task-status edge function URL available for marking task {task_id_str} as failed", task_id=task_id_str)
        return

    headers = {"Content-Type": "application/json"}
    if _cfg.SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

    payload = {
        "task_id": task_id_str,
        "status": STATUS_FAILED,
        "output_location": error_message
    }

    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload,
        headers=headers,
        function_name="update-task-status",
        context_id=task_id_str,
        timeout=30,
        max_retries=3)

    if resp and resp.status_code == 200:
        headless_logger.debug(f"[DEBUG] Successfully marked task {task_id_str} as Failed via Edge Function")
    elif edge_error:
        headless_logger.error(f"Failed to mark task {task_id_str} as Failed: {edge_error}", task_id=task_id_str)
    elif resp:
        headless_logger.error(f"Failed to mark task {task_id_str} as Failed: {resp.status_code} - {resp.text}", task_id=task_id_str)

def requeue_task_for_retry(task_id_str: str, error_message: str, current_attempts: int, error_category: str = None) -> bool:
    """
    Reset a task to Queued status for retry, incrementing the attempts counter.

    This is used for transient errors (OOM, edge function failures, etc.) that may
    succeed on a subsequent attempt.

    Args:
        task_id_str: Task ID to requeue
        error_message: Error message from the failed attempt
        current_attempts: Current attempt count (will be incremented)
        error_category: Optional category of the retryable error

    Returns:
        True if task was successfully requeued, False otherwise
    """
    new_attempts = current_attempts + 1

    # Build error details message
    error_details = f"Retry {new_attempts}"
    if error_category:
        error_details += f" ({error_category})"
    error_details += f": {error_message[:500]}" if error_message else ""

    headless_logger.essential(f"Requeuing task {task_id_str} for retry (attempt {new_attempts})", task_id=task_id_str)
    headless_logger.debug(f"[RETRY_DEBUG] Error category: {error_category}, Error: {error_message[:200] if error_message else 'N/A'}...")

    # Use edge function to update status back to Queued
    edge_url = (
        os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url:
        headless_logger.error(f"No update-task-status edge function URL available for requeuing task {task_id_str}", task_id=task_id_str)
        # Fallback to direct DB update
        return _requeue_task_direct_db(task_id_str, new_attempts, error_details)

    headers = {"Content-Type": "application/json"}
    if _cfg.SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

    payload = {
        "task_id": task_id_str,
        "status": STATUS_QUEUED,
        "attempts": new_attempts,
        "error_details": error_details,
        "clear_worker": True,  # Signal to clear worker_id assignment
    }

    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload,
        headers=headers,
        function_name="update-task-status",
        context_id=task_id_str,
        timeout=30,
        max_retries=3)

    if resp and resp.status_code == 200:
        headless_logger.essential(f"Task {task_id_str} requeued for retry (attempt {new_attempts})", task_id=task_id_str)
        return True
    elif edge_error:
        headless_logger.error(f"Failed to requeue task {task_id_str}: {edge_error}", task_id=task_id_str)
        # Fallback to direct DB update
        return _requeue_task_direct_db(task_id_str, new_attempts, error_details)
    elif resp:
        headless_logger.error(f"Failed to requeue task {task_id_str}: {resp.status_code} - {resp.text}", task_id=task_id_str)
        # Fallback to direct DB update
        return _requeue_task_direct_db(task_id_str, new_attempts, error_details)

    return False

def _requeue_task_direct_db(task_id_str: str, new_attempts: int, error_details: str) -> bool:
    """
    Fallback: Requeue task directly via Supabase client if edge function fails.
    """
    if not _cfg.SUPABASE_CLIENT:
        headless_logger.error(f"No Supabase client available for direct DB requeue of task {task_id_str}", task_id=task_id_str)
        return False

    try:
        result = _cfg.SUPABASE_CLIENT.table(_cfg.PG_TABLE_NAME).update({
            "status": STATUS_QUEUED,
            "worker_id": None,
            "attempts": new_attempts,
            "error_details": error_details,
            "generation_started_at": None,
        }).eq("id", task_id_str).execute()

        if result.data:
            headless_logger.essential(f"Task {task_id_str} requeued via direct DB (attempt {new_attempts})", task_id=task_id_str)
            return True
        else:
            headless_logger.error(f"Direct DB requeue returned no data for task {task_id_str}", task_id=task_id_str)
            return False
    except (APIError, RuntimeError, ValueError, OSError) as e:
        headless_logger.error(f"Direct DB requeue failed for task {task_id_str}: {e}", task_id=task_id_str)
        return False

def update_task_status(task_id: str, status: str, output_location: str | None = None):
    """Updates a task's status in Supabase."""
    headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG] Called with:")
    headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG]   task_id: '{task_id}'")
    headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG]   status: '{status}'")
    headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG]   output_location: '{output_location}'")

    try:
        headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG] Dispatching to update_task_status_supabase")
        result = update_task_status_supabase(task_id, status, output_location)
        headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG] update_task_status_supabase completed successfully")
        return result
    except (RuntimeError, ValueError, OSError) as e:
        headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG] \u274c Exception in update_task_status: {e}", exc_info=True)
        headless_logger.debug(f"[UPDATE_TASK_STATUS_DEBUG] Exception type: {type(e).__name__}")
        raise

def update_task_status_supabase(task_id_str, status_str, output_location_val=None, thumbnail_url_val=None):
    legacy_result = _update_task_status_supabase_legacy(
        task_id_str,
        status_str,
        output_location_val,
        thumbnail_url_val,
    )
    if legacy_result is False:
        return TaskStatusUpdateResult(
            outcome="failed",
            output_location=output_location_val,
            error=f"status_update_failed:{status_str}",
        )
    if isinstance(legacy_result, dict):
        return TaskStatusUpdateResult(
            outcome="completed",
            output_location=legacy_result.get("public_url") or output_location_val,
            public_url=legacy_result.get("public_url"),
            thumbnail_url=legacy_result.get("thumbnail_url"),
        )
    if isinstance(legacy_result, str):
        return TaskStatusUpdateResult(
            outcome="completed" if status_str == STATUS_COMPLETE else "updated",
            output_location=legacy_result,
        )
    if legacy_result is True:
        return TaskStatusUpdateResult(
            outcome="updated",
            output_location=output_location_val,
            thumbnail_url=thumbnail_url_val,
        )
    return TaskStatusUpdateResult(
        outcome="completed" if status_str == STATUS_COMPLETE else "updated",
        output_location=output_location_val,
        thumbnail_url=thumbnail_url_val,
    )


def _update_task_status_supabase_legacy(task_id_str, status_str, output_location_val=None, thumbnail_url_val=None):
    """Updates a task's status via Supabase Edge Functions.

    Args:
        task_id_str: Task ID
        status_str: Status to set
        output_location_val: Output file location or URL
        thumbnail_url_val: Optional thumbnail URL to pass to edge function
    """
    headless_logger.debug(f"[DEBUG] update_task_status_supabase called: task_id={task_id_str}, status={status_str}, output_location={output_location_val}, thumbnail={thumbnail_url_val}")

    if not _cfg.SUPABASE_CLIENT:
        headless_logger.error("Supabase client not initialized. Cannot update task status.", task_id=task_id_str)
        return

    # --- Use edge functions for ALL status updates ---
    if status_str == STATUS_COMPLETE and output_location_val is not None:
        # Use completion edge function for completion with file
        # NOTE: Canonical deployed edge function is `complete_task` (underscore).
        edge_url = (
            _cfg.SUPABASE_EDGE_COMPLETE_TASK_URL
            or (os.getenv("SUPABASE_EDGE_COMPLETE_TASK_URL") or None)
            or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/complete_task" if _cfg.SUPABASE_URL else None)
        )

        if not edge_url:
            headless_logger.error(f"No complete_task edge function URL available", task_id=task_id_str)
            return

        try:
            # Check if output_location_val is a local file path
            output_path = Path(output_location_val)

            if output_path.exists() and output_path.is_file():
                import base64
                import mimetypes

                # Get file size for logging
                file_size = output_path.stat().st_size
                file_size_mb = file_size / BYTES_PER_MB
                headless_logger.debug(f"[DEBUG] File size: {file_size_mb:.2f} MB")

                headers = {"Content-Type": "application/json"}
                if _cfg.SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

                # Check if this is a video file that needs thumbnail extraction
                from source.core.constants import VIDEO_EXTENSIONS
                is_video = output_path.suffix.lower() in VIDEO_EXTENSIONS
                _log_thumbnail(f"File: {output_path.name}, suffix: {output_path.suffix.lower()}, is_video: {is_video}", task_id=task_id_str)

                # Use base64 encoding for files under 2MB (MODE 1), presigned URLs for larger files (MODE 3)
                FILE_SIZE_THRESHOLD_MB = 2.0
                use_base64 = file_size_mb < FILE_SIZE_THRESHOLD_MB

                if use_base64:
                    headless_logger.debug(f"[DEBUG] Using base64 upload for {output_path.name} ({file_size_mb:.2f} MB < {FILE_SIZE_THRESHOLD_MB} MB)")

                    # MODE 1: Base64 upload (for files under 2MB)
                    # Read and encode file
                    with open(output_path, 'rb') as f:
                        file_bytes = f.read()
                        file_data_base64 = base64.b64encode(file_bytes).decode('utf-8')

                    payload = {
                        "task_id": task_id_str,
                        "file_data": file_data_base64,
                        "filename": output_path.name
                    }

                    # Extract and encode first frame for videos
                    if is_video:
                        try:
                            thumb_path = _extract_video_thumbnail(output_path, task_id_str)
                            if thumb_path:
                                with open(thumb_path, 'rb') as thumb_file:
                                    first_frame_base64 = base64.b64encode(thumb_file.read()).decode('utf-8')
                                payload["first_frame_data"] = first_frame_base64
                                payload["first_frame_filename"] = f"thumb_{task_id_str[:8]}.jpg"
                                # Clean up temp file (but not existing poster)
                                if thumb_path != output_path.with_suffix('.jpg'):
                                    thumb_path.unlink(missing_ok=True)
                        except (OSError, ValueError, RuntimeError) as e:
                            _log_thumbnail(f"Exception during thumbnail extraction: {e}", level="warning", task_id=task_id_str)
                            # Continue without thumbnail

                    headless_logger.debug(f"[DEBUG] Calling complete_task Edge Function with base64 data for task {task_id_str}")
                    resp, edge_error = _call_edge_function_with_retry(
                        edge_url=edge_url,
                        payload=payload,
                        headers=headers,
                        function_name="complete_task",
                        context_id=task_id_str,
                        timeout=60,
                        max_retries=3,
                        fallback_url=None,
                        retry_on_404_patterns=["Task not found", "not found"])

                    if resp is not None and resp.status_code == 200:
                        headless_logger.debug(f"[DEBUG] Edge function SUCCESS for task {task_id_str} \u2192 status COMPLETE with base64 upload")
                        # Parse response to get storage URL and thumbnail URL
                        try:
                            resp_data = resp.json()
                            storage_url = resp_data.get('public_url')
                            thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                            if storage_url:
                                headless_logger.debug(f"[DEBUG] File uploaded to: {storage_url}")
                                if thumbnail_url:
                                    headless_logger.debug(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                                # Return both URLs as a dict
                                return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        except (ValueError, KeyError) as e:
                            headless_logger.debug(f"[DEBUG] Failed to parse complete_task response JSON for task {task_id_str}: {e}")
                        return None
                    else:
                        error_msg = edge_error or f"{EDGE_FAIL_PREFIX}:complete_task:HTTP_{resp.status_code if resp else 'N/A'}] {resp.text[:200] if resp else 'No response'}"
                        headless_logger.error(error_msg, task_id=task_id_str)
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return None

                else:
                    headless_logger.debug(f"[DEBUG] Using presigned upload for {output_path.name} ({file_size_mb:.2f} MB >= {FILE_SIZE_THRESHOLD_MB} MB)")

                    # MODE 3: Presigned URL upload (for files 2MB or larger)
                    # Step 1: Get signed upload URLs (request thumbnail URL for videos) - WITH RETRY
                    generate_url_edge = f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/generate-upload-url"
                    content_type = mimetypes.guess_type(str(output_path))[0] or 'application/octet-stream'

                    upload_url_resp, edge_error = _call_edge_function_with_retry(
                        edge_url=generate_url_edge,
                        payload={
                            "task_id": task_id_str,
                            "filename": output_path.name,
                            "content_type": content_type,
                            "generate_thumbnail_url": is_video  # Request thumbnail URL for videos
                        },
                        headers=headers,
                        function_name="generate-upload-url",
                        context_id=task_id_str,
                        timeout=30,
                        max_retries=3)

                    if edge_error or not upload_url_resp or upload_url_resp.status_code != 200:
                        # Prefer standardized error from helper (avoids double-wrapping)
                        error_msg = edge_error or f"{EDGE_FAIL_PREFIX}:generate-upload-url:HTTP_{upload_url_resp.status_code if upload_url_resp else 'N/A'}] {upload_url_resp.text[:200] if upload_url_resp else 'No response'}"
                        headless_logger.error(error_msg, task_id=task_id_str)
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return

                    upload_data = upload_url_resp.json()

                    # Step 2: Extract and upload thumbnail for videos (if thumbnail URL was generated)
                    thumbnail_storage_path = None
                    if is_video and "thumbnail_upload_url" not in upload_data:
                        _log_thumbnail(f"MODE3: \u26a0\ufe0f Edge function did not return thumbnail_upload_url for video", level="warning", task_id=task_id_str)
                    if is_video and "thumbnail_upload_url" in upload_data:
                        try:
                            thumb_path = _extract_video_thumbnail(output_path, task_id_str)
                            if thumb_path:
                                # Upload thumbnail via signed URL - WITH RETRY
                                thumb_resp, thumb_error = _call_edge_function_with_retry(
                                    edge_url=upload_data["thumbnail_upload_url"],
                                    payload=thumb_path,
                                    headers={"Content-Type": "image/jpeg"},
                                    function_name="storage-upload-thumbnail",
                                    context_id=task_id_str,
                                    timeout=60,
                                    max_retries=3,
                                    method="PUT")
                                if thumb_resp and thumb_resp.status_code in [200, 201]:
                                    thumbnail_storage_path = upload_data["thumbnail_storage_path"]
                                    _log_thumbnail(f"MODE3: Thumbnail uploaded successfully", level="info", task_id=task_id_str)
                                else:
                                    _log_thumbnail(f"MODE3: Thumbnail upload failed: {thumb_error or (thumb_resp.status_code if thumb_resp else 'No response')}", level="warning", task_id=task_id_str)
                                # Clean up temp file (but not existing poster)
                                if thumb_path != output_path.with_suffix('.jpg'):
                                    thumb_path.unlink(missing_ok=True)
                        except (OSError, ValueError, RuntimeError) as e:
                            _log_thumbnail(f"MODE3: Exception during thumbnail handling: {e}", level="warning", task_id=task_id_str)
                            # Continue without thumbnail

                    # Step 3: Upload main file directly to storage using presigned URL - WITH RETRY
                    headless_logger.debug(f"[DEBUG] Uploading main file via signed URL")
                    put_resp, put_error = _call_edge_function_with_retry(
                        edge_url=upload_data["upload_url"],
                        payload=output_path,
                        headers={"Content-Type": content_type},
                        function_name="storage-upload-file",
                        context_id=task_id_str,
                        timeout=600,  # 10 minute base timeout for large files
                        max_retries=3,
                        method="PUT")

                    if put_error or not put_resp or put_resp.status_code not in [200, 201]:
                        error_msg = put_error or f"{EDGE_FAIL_PREFIX}:storage-upload-file:HTTP_{put_resp.status_code if put_resp else 'N/A'}] {put_resp.text[:200] if put_resp else 'No response'}"
                        headless_logger.error(error_msg, task_id=task_id_str)
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return

                    headless_logger.debug(f"[DEBUG] File uploaded successfully via presigned URL")

                    # Step 4: Complete task with storage paths
                    payload = {
                        "task_id": task_id_str,
                        "storage_path": upload_data["storage_path"]
                    }

                    # Add thumbnail storage path if available
                    if thumbnail_storage_path:
                        payload["thumbnail_storage_path"] = thumbnail_storage_path

                    headless_logger.debug(f"[DEBUG] Calling complete_task Edge Function with storage_path for task {task_id_str}")
                    resp, edge_error = _call_edge_function_with_retry(
                        edge_url=edge_url,
                        payload=payload,
                        headers=headers,
                        function_name="complete_task",
                        context_id=task_id_str,
                        timeout=60,
                        max_retries=3,
                        fallback_url=None,
                        retry_on_404_patterns=["Task not found", "not found"])

                    if resp is not None and resp.status_code == 200:
                        headless_logger.debug(f"[DEBUG] Edge function SUCCESS for task {task_id_str} \u2192 status COMPLETE with file upload")
                        # Parse response to get storage URL and thumbnail URL
                        try:
                            resp_data = resp.json()
                            storage_url = resp_data.get('public_url')
                            thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                            if storage_url:
                                headless_logger.debug(f"[DEBUG] File uploaded to: {storage_url}")
                                if thumbnail_url:
                                    headless_logger.debug(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                                # Return both URLs as a dict
                                return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        except (ValueError, KeyError) as e:
                            headless_logger.debug(f"[DEBUG] Failed to parse complete_task response JSON for task {task_id_str}: {e}")
                        return None
                    else:
                        error_msg = edge_error or f"{EDGE_FAIL_PREFIX}:complete_task:HTTP_{resp.status_code if resp else 'N/A'}] {resp.text[:200] if resp else 'No response'}"
                        headless_logger.error(error_msg, task_id=task_id_str)
                        # Use update-task-status edge function to mark as failed
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return None
            else:
                # Not a local file, could be a Supabase storage URL or external reference
                # Check if it's a Supabase storage URL - if so, extract storage_path for MODE 3/4
                storage_path = None

                # SPECIAL CASE: JSON output containing a storage URL (e.g., transition_only mode)
                # JSON outputs contain metadata along with a file URL - we need to:
                # 1. Extract the actual file URL from the JSON
                # 2. Use that URL's storage_path for the edge function to mark complete
                # 3. Update output_location with the full JSON afterward
                if output_location_val.strip().startswith('{') and output_location_val.strip().endswith('}'):
                    try:
                        json_output = json.loads(output_location_val)
                        # Look for common URL fields in the JSON
                        url_in_json = (
                            json_output.get("transition_url") or
                            json_output.get("url") or
                            json_output.get("output_url") or
                            json_output.get("video_url")
                        )
                        if url_in_json and "/storage/v1/object/public/image_uploads/" in url_in_json:
                            # Extract storage_path from the URL inside the JSON
                            path_parts = url_in_json.split("/storage/v1/object/public/image_uploads/", 1)
                            if len(path_parts) == 2:
                                storage_path = path_parts[1]
                                headless_logger.debug(f"[DEBUG] JSON output detected - extracted storage_path: {storage_path}")

                                # Complete task with storage_path and include JSON output_location
                                # in a single call (avoids broken two-step Complete→Complete transition)
                                payload = {
                                    "task_id": task_id_str,
                                    "storage_path": storage_path,
                                    "output_location": output_location_val,  # Full JSON metadata
                                }
                                headless_logger.debug(f"[DEBUG] Completing task {task_id_str} with storage_path + JSON output_location")

                                headers = {"Content-Type": "application/json"}
                                if _cfg.SUPABASE_ACCESS_TOKEN:
                                    headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

                                resp, edge_error = _call_edge_function_with_retry(
                                    edge_url=edge_url,
                                    payload=payload,
                                    headers=headers,
                                    function_name="complete_task",
                                    context_id=task_id_str,
                                    timeout=30,
                                    max_retries=3,
                                    fallback_url=None,
                                    retry_on_404_patterns=["Task not found", "not found"])

                                if resp is not None and resp.status_code == 200:
                                    headless_logger.debug(f"[DEBUG] Task {task_id_str} completed with JSON output_location")
                                    return None
                                else:
                                    error_msg = edge_error or f"{EDGE_FAIL_PREFIX}:complete_task:HTTP_{resp.status_code if resp else 'N/A'}] {resp.text[:200] if resp else 'No response'}"
                                    headless_logger.error(error_msg, task_id=task_id_str)
                                    _mark_task_failed_via_edge_function(task_id_str, f"Completion failed: {error_msg}")
                                    return None
                    except json.JSONDecodeError:
                        headless_logger.debug(f"[DEBUG] Output looks like JSON but failed to parse, treating as regular output")

                if "/storage/v1/object/public/image_uploads/" in output_location_val:
                    # Extract storage path from URL
                    # Format: https://xxx.supabase.co/storage/v1/object/public/image_uploads/{userId}/{filename}
                    # MODE 3: userId/tasks/{task_id}/filename (pre-signed URL uploads)
                    # MODE 4: userId/filename (orchestrator referencing child task upload)
                    try:
                        path_parts = output_location_val.split("/storage/v1/object/public/image_uploads/", 1)
                        if len(path_parts) == 2:
                            storage_path = path_parts[1]  # e.g., "userId/filename.mp4" or "userId/tasks/{task_id}/filename.mp4"
                            headless_logger.debug(f"[DEBUG] Extracted storage_path from URL: {storage_path}")

                            # Determine if this is MODE 3 or MODE 4 based on path structure
                            path_components = storage_path.split('/')
                            if len(path_components) >= 4 and path_components[1] == 'tasks':
                                headless_logger.debug(f"[DEBUG] MODE 3 path detected (pre-signed URL): {storage_path}")
                            else:
                                headless_logger.debug(f"[DEBUG] MODE 4 path detected (orchestrator reference): {storage_path}")
                    except (ValueError, IndexError) as e_extract:
                        headless_logger.debug(f"[DEBUG] Failed to extract storage_path: {e_extract}")

                # Use MODE 3/4 if we have a storage path, otherwise use output_location (legacy)
                if storage_path:
                    payload = {"task_id": task_id_str, "storage_path": storage_path}
                    headless_logger.debug(f"[DEBUG] Using storage_path for task {task_id_str}")

                    # Extract thumbnail storage path if thumbnail URL provided
                    if thumbnail_url_val and "/storage/v1/object/public/image_uploads/" in thumbnail_url_val:
                        try:
                            thumb_parts = thumbnail_url_val.split("/storage/v1/object/public/image_uploads/", 1)
                            if len(thumb_parts) == 2:
                                thumbnail_storage_path = thumb_parts[1]
                                payload["thumbnail_storage_path"] = thumbnail_storage_path
                                headless_logger.debug(f"[DEBUG] Including thumbnail_storage_path: {thumbnail_storage_path}")
                        except (ValueError, IndexError) as e:
                            headless_logger.debug(f"[DEBUG] Failed to extract thumbnail path: {e}")
                else:
                    payload = {"task_id": task_id_str, "output_location": output_location_val}
                    headless_logger.debug(f"[DEBUG] Using output_location (legacy) for task {task_id_str}")

                headers = {"Content-Type": "application/json"}
                if _cfg.SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

                resp, edge_error = _call_edge_function_with_retry(
                    edge_url=edge_url,
                    payload=payload,
                    headers=headers,
                    function_name="complete_task",
                    context_id=task_id_str,
                    timeout=30,
                    max_retries=3,
                    fallback_url=None,
                    retry_on_404_patterns=["Task not found", "not found"])

                if resp is not None and resp.status_code == 200:
                    headless_logger.debug(f"[DEBUG] Edge function SUCCESS for task {task_id_str} \u2192 status COMPLETE")
                    # Parse response to get storage URL and thumbnail URL
                    try:
                        resp_data = resp.json()
                        storage_url = resp_data.get('public_url')
                        thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                        if storage_url:
                            headless_logger.debug(f"[DEBUG] Storage URL: {storage_url}")
                            if thumbnail_url:
                                headless_logger.debug(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                            # Return both URLs as a dict
                            return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        # If no URL in response, return the original output_location_val as string (legacy)
                        return output_location_val
                    except (ValueError, KeyError):
                        return output_location_val
                else:
                    error_msg = edge_error or f"{EDGE_FAIL_PREFIX}:complete_task:HTTP_{resp.status_code if resp else 'N/A'}] {resp.text[:200] if resp else 'No response'}"
                    headless_logger.error(error_msg, task_id=task_id_str)
                    # Use update-task-status edge function to mark as failed
                    _mark_task_failed_via_edge_function(task_id_str, f"Completion failed: {error_msg}")
                    return None
        except (OSError, ValueError, RuntimeError) as e_edge:
            headless_logger.error(f"complete_task edge function exception: {e_edge}", task_id=task_id_str)
            return None
    else:
        # Use update-task-status edge function for all other status updates (with retry)
        edge_url = (
            os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL")
            or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if _cfg.SUPABASE_URL else None)
        )

        if not edge_url:
            headless_logger.error(f"No update-task-status edge function URL available", task_id=task_id_str)
            return

        headers = {"Content-Type": "application/json"}
        if _cfg.SUPABASE_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

        payload = {
            "task_id": task_id_str,
            "status": status_str
        }

        if output_location_val:
            payload["output_location"] = output_location_val

        if thumbnail_url_val:
            payload["thumbnail_url"] = thumbnail_url_val

        headless_logger.debug(f"[DEBUG] Calling update-task-status Edge Function for task {task_id_str} \u2192 {status_str}")
        resp, edge_error = _call_edge_function_with_retry(
            edge_url=edge_url,
            payload=payload,
            headers=headers,
            function_name="update-task-status",
            context_id=task_id_str,
            timeout=30,
            max_retries=3)

        if resp and resp.status_code == 200:
            headless_logger.debug(f"[DEBUG] Edge function SUCCESS for task {task_id_str} \u2192 status {status_str}")
            return
        elif edge_error:
            headless_logger.error(f"update-task-status edge function failed: {edge_error}", task_id=task_id_str)
            return
        elif resp:
            headless_logger.error(f"update-task-status edge function failed: {resp.status_code} - {resp.text}", task_id=task_id_str)
            return

def mark_task_failed_supabase(task_id_str, error_message):
    """Marks a task as Failed with an error message using direct database update."""
    headless_logger.debug(f"Marking task {task_id_str} as Failed with message: {error_message}")
    if not _cfg.SUPABASE_CLIENT:
        headless_logger.error("Supabase client not initialized. Cannot mark task failed.", task_id=task_id_str)
        return

    # Use the standard update function which now uses direct database updates for non-COMPLETE statuses
    update_task_status_supabase(task_id_str, STATUS_FAILED, error_message)

def reset_generation_started_at(task_id_str: str) -> bool:
    """
    Resets the generation_started_at timestamp to NOW for a task.

    This should be called after model loading completes, so users are not
    charged for model loading time. The billing period starts from this
    reset timestamp instead of the original claim time.

    Args:
        task_id_str: Task ID to reset timestamp for

    Returns:
        True if successful, False otherwise
    """
    edge_url = (
        os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL")
        or (f"{_cfg.SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if _cfg.SUPABASE_URL else None)
    )

    if not edge_url:
        headless_logger.error(f"No update-task-status edge function URL available for resetting generation_started_at", task_id=task_id_str)
        return False

    headers = {"Content-Type": "application/json"}
    if _cfg.SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {_cfg.SUPABASE_ACCESS_TOKEN}"

    # Send current status with reset flag - the task is already "In Progress"
    payload = {
        "task_id": task_id_str,
        "status": STATUS_IN_PROGRESS,
        "reset_generation_started_at": True
    }

    headless_logger.essential(f"[BILLING] Resetting generation_started_at for task {task_id_str} (model loading complete)", task_id=task_id_str)

    resp, edge_error = _call_edge_function_with_retry(
        edge_url=edge_url,
        payload=payload,
        headers=headers,
        function_name="update-task-status",
        context_id=task_id_str,
        timeout=30,
        max_retries=3)

    if resp and resp.status_code == 200:
        headless_logger.essential(f"[BILLING] Successfully reset generation_started_at for task {task_id_str}", task_id=task_id_str)
        return True
    elif edge_error:
        headless_logger.error(f"Failed to reset generation_started_at for task {task_id_str}: {edge_error}", task_id=task_id_str)
        return False
    elif resp:
        headless_logger.error(f"Failed to reset generation_started_at for task {task_id_str}: {resp.status_code} - {resp.text}", task_id=task_id_str)
        return False

    return False
