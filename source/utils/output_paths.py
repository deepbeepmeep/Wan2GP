"""Canonical output path and upload helpers."""

import os
import time
from pathlib import Path

from source.core.constants import BYTES_PER_KB as BYTES_PER_KIBIBYTE
from source.core.log import headless_logger

__all__ = [
    "prepare_output_path",
    "sanitize_filename_for_storage",
    "prepare_output_path_with_upload",
    "upload_and_get_final_output_location",
    "upload_intermediate_file_to_storage",
    "wait_for_file_stable",
]


def _get_task_type_directory(task_type: str) -> str:
    """Map a task type to its output subdirectory."""
    return task_type if task_type else "misc"


def prepare_output_path(
    task_id: str,
    filename: str,
    main_output_dir_base: Path,
    task_type: str | None = None,
    *,
    custom_output_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Prepare a local output path plus the DB-facing location string."""
    if custom_output_dir:
        output_dir_for_task = Path(custom_output_dir)
    else:
        if task_type:
            type_subdir = _get_task_type_directory(task_type)
            output_dir_for_task = main_output_dir_base / type_subdir
        else:
            output_dir_for_task = main_output_dir_base

        import re

        uuid_pattern = r"_\d{6}_[a-f0-9]{6}\.(mp4|png|jpg|jpeg)$"
        has_uuid_pattern = re.search(uuid_pattern, filename, re.IGNORECASE)
        if not filename.startswith(task_id) and not has_uuid_pattern:
            filename = f"{task_id}_{filename}"

    output_dir_for_task.mkdir(parents=True, exist_ok=True)
    final_save_path = output_dir_for_task / filename

    if final_save_path.exists():
        stem = final_save_path.stem
        suffix = final_save_path.suffix
        counter = 1
        while final_save_path.exists():
            final_save_path = output_dir_for_task / f"{stem}_{counter}{suffix}"
            counter += 1
        headless_logger.debug(
            f"Task {task_id}: Filename conflict resolved - using {final_save_path.name}",
            task_id=task_id,
        )

    try:
        db_output_location = str(final_save_path.relative_to(Path.cwd()))
    except ValueError:
        db_output_location = str(final_save_path.resolve())

    return final_save_path, db_output_location


def sanitize_filename_for_storage(
    filename: str,
    *,
    whitespace_mode: str = "space",
    empty_fallback: str = "sanitized_file",
) -> str:
    """Sanitize a filename for object storage backends."""
    import re

    if whitespace_mode not in {"space", "underscore"}:
        raise ValueError(f"Unknown whitespace_mode: {whitespace_mode}")

    unsafe_chars = r'[\u00a7\u00ae\u00a9\u2122@\u00b7\u00ba\u00bd\u00be\u00bf\u00a1~\x00-\x1F\x7F-\x9F<>:"/\\|?*,]'
    sanitized = re.sub(unsafe_chars, "", filename)
    sanitized = re.sub(r"\.{2,}", " ", sanitized)
    if whitespace_mode == "underscore":
        sanitized = re.sub(r"\s+", "_", sanitized.strip())
    else:
        sanitized = re.sub(r"\s+", " ", sanitized).strip(" .")
    return sanitized or empty_fallback


def prepare_output_path_with_upload(
    task_id: str,
    filename: str,
    main_output_dir_base: Path,
    task_type: str | None = None,
    *,
    custom_output_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Prepare an output path after sanitizing the storage-facing filename."""
    original_filename = filename
    sanitized_filename = sanitize_filename_for_storage(filename)
    if original_filename != sanitized_filename:
        headless_logger.debug(
            f"Task {task_id}: Sanitized filename '{original_filename}' -> '{sanitized_filename}'",
            task_id=task_id,
        )
    return prepare_output_path(
        task_id,
        sanitized_filename,
        main_output_dir_base,
        task_type=task_type,
        custom_output_dir=custom_output_dir,
    )


def upload_and_get_final_output_location(
    local_file_path: Path,
    initial_db_location: str,
) -> str:
    """Return the resolved local path; edge upload occurs later."""
    del initial_db_location
    headless_logger.debug(f"File ready for edge function upload: {local_file_path}")
    return str(local_file_path.resolve())


def upload_intermediate_file_to_storage(
    local_file_path: Path,
    task_id: str,
    filename: str,
    runtime_config=None,
) -> str | None:
    """Upload an intermediate file for cross-worker access."""
    import httpx
    import mimetypes

    from source.core.db import config as _db_config

    del runtime_config

    retryable_status_codes = {502, 503, 504}
    max_retries = 3
    supabase_url = _db_config.SUPABASE_URL
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or _db_config.SUPABASE_ACCESS_TOKEN
    )

    if not supabase_url or not supabase_key:
        headless_logger.warning("[UPLOAD_INTERMEDIATE] Supabase not configured, cannot upload", task_id=task_id)
        return None
    if not local_file_path.exists():
        headless_logger.warning(f"[UPLOAD_INTERMEDIATE] File not found: {local_file_path}", task_id=task_id)
        return None

    try:
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "apikey": supabase_key,
            "Content-Type": "application/json",
        }
        generate_url_edge = f"{supabase_url.rstrip('/')}/functions/v1/generate-upload-url"
        content_type = mimetypes.guess_type(str(local_file_path))[0] or "application/octet-stream"

        upload_url_resp = None
        for attempt in range(max_retries):
            try:
                upload_url_resp = httpx.post(
                    generate_url_edge,
                    headers=headers,
                    json={
                        "task_id": task_id,
                        "filename": filename,
                        "content_type": content_type,
                    },
                    timeout=30 + (attempt * 15),
                )
                if upload_url_resp.status_code == 200:
                    break
                if upload_url_resp.status_code in retryable_status_codes and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    headless_logger.warning(
                        f"[UPLOAD_INTERMEDIATE] generate-upload-url got {upload_url_resp.status_code}, "
                        f"retrying in {wait_time}s...",
                        task_id=task_id,
                    )
                    time.sleep(wait_time)
                    continue
                break
            except (httpx.HTTPError, OSError, ValueError) as exc:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    headless_logger.warning(
                        f"[UPLOAD_INTERMEDIATE] generate-upload-url error, retrying in {wait_time}s: {exc}",
                        task_id=task_id,
                    )
                    time.sleep(wait_time)
                    continue
                headless_logger.error(
                    f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:generate-upload-url:NETWORK] {exc}",
                    task_id=task_id,
                )
                return None

        if not upload_url_resp or upload_url_resp.status_code != 200:
            error_text = upload_url_resp.text[:200] if upload_url_resp else "No response"
            error_code = upload_url_resp.status_code if upload_url_resp else "N/A"
            headless_logger.error(
                f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:generate-upload-url:HTTP_{error_code}] {error_text}",
                task_id=task_id,
            )
            return None

        upload_data = upload_url_resp.json()
        upload_url = upload_data.get("upload_url")
        storage_path = upload_data.get("storage_path")
        if not upload_url:
            headless_logger.error("[UPLOAD_INTERMEDIATE] No upload_url in response", task_id=task_id)
            return None

        file_size_mb = local_file_path.stat().st_size / BYTES_PER_KIBIBYTE / BYTES_PER_KIBIBYTE
        headless_logger.debug(
            f"[UPLOAD_INTERMEDIATE] Uploading {local_file_path.name} ({file_size_mb:.1f} MB)",
            task_id=task_id,
        )

        put_resp = None
        for attempt in range(max_retries):
            try:
                with open(local_file_path, "rb") as handle:
                    put_resp = httpx.put(
                        upload_url,
                        headers={"Content-Type": content_type},
                        content=handle,
                        timeout=300 + (attempt * 60),
                    )
                if put_resp.status_code in {200, 201}:
                    break
                if put_resp.status_code in retryable_status_codes and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    headless_logger.warning(
                        f"[UPLOAD_INTERMEDIATE] storage-upload got {put_resp.status_code}, retrying in {wait_time}s...",
                        task_id=task_id,
                    )
                    time.sleep(wait_time)
                    continue
                break
            except (httpx.HTTPError, OSError, ValueError) as exc:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    headless_logger.warning(
                        f"[UPLOAD_INTERMEDIATE] storage-upload error, retrying in {wait_time}s: {exc}",
                        task_id=task_id,
                    )
                    time.sleep(wait_time)
                    continue
                headless_logger.error(
                    f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:storage-upload:NETWORK] {exc}",
                    task_id=task_id,
                )
                return None

        if not put_resp or put_resp.status_code not in {200, 201}:
            error_text = put_resp.text[:200] if put_resp else "No response"
            error_code = put_resp.status_code if put_resp else "N/A"
            headless_logger.error(
                f"[UPLOAD_INTERMEDIATE] [EDGE_FAIL:storage-upload:HTTP_{error_code}] {error_text}",
                task_id=task_id,
            )
            return None

        public_url = f"{supabase_url}/storage/v1/object/public/image_uploads/{storage_path}"
        headless_logger.debug_anomaly("UPLOAD_INTERMEDIATE", f"Uploaded to: {public_url}", task_id=task_id)
        return public_url
    except (httpx.HTTPError, OSError, ValueError) as exc:
        headless_logger.error(f"[UPLOAD_INTERMEDIATE] Exception: {exc}", task_id=task_id, exc_info=True)
        return None


def wait_for_file_stable(path: Path | str, checks: int = 3, interval: float = 1.0) -> bool:
    """Return True when a file size remains stable across a few checks."""
    candidate = Path(path)
    if not candidate.exists():
        return False
    last_size = candidate.stat().st_size
    stable_count = 0
    for _ in range(checks):
        time.sleep(interval)
        new_size = candidate.stat().st_size
        if new_size == last_size and new_size > 0:
            stable_count += 1
            if stable_count >= checks - 1:
                return True
        else:
            stable_count = 0
            last_size = new_size
    return False
