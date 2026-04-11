"""URL handling, file downloading, and related utilities."""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from source.core.log import headless_logger

__all__ = [
    "DownloadError",
    "download_file",
    "download_video_if_url",
    "download_image_if_url",
]


class DownloadError(RuntimeError):
    """Raised when a URL download cannot be completed."""


def _stream_download_to_path(url: str, destination_path: Path, timeout: int = 300) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

def _get_unique_target_path(target_dir: Path, base_name: str, extension: str) -> Path:
    """Generates a unique target Path in the given directory by appending a timestamp and random string."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp_short = datetime.now().strftime("%H%M%S")
    # Use a short UUID/random string to significantly reduce collision probability with just timestamp
    unique_suffix = uuid.uuid4().hex[:6]

    # Construct the filename
    # Ensure extension has a leading dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension

    filename = f"{base_name}_{timestamp_short}_{unique_suffix}{extension}"
    return target_dir / filename

def download_file(url, dest_folder, filename):
    from source.utils.lora_validation import validate_lora_file

    dest_path = Path(dest_folder) / filename
    if dest_path.exists():
        # Validate existing file before assuming it's good
        if filename.endswith('.safetensors') or 'lora' in filename.lower():
            is_valid, validation_msg = validate_lora_file(dest_path, filename)
            if is_valid:
                headless_logger.essential(f"File {filename} already exists and is valid in {dest_folder}. {validation_msg}")
                return True
            else:
                headless_logger.warning(f"Existing file {filename} failed validation ({validation_msg}). Re-downloading...")
                dest_path.unlink()
        else:
            headless_logger.essential(f"File {filename} already exists in {dest_folder}.")
            return True

    # Use huggingface_hub for HuggingFace URLs for better reliability
    if "huggingface.co" in url:
        try:
            from huggingface_hub import hf_hub_download

            # Parse HuggingFace URL to extract repo_id and filename
            # Format: https://huggingface.co/USER/REPO/resolve/BRANCH/FILENAME
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')

            if len(path_parts) >= 4 and path_parts[2] == 'resolve':
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                branch = path_parts[3] if len(path_parts) > 4 else "main"
                hf_filename = '/'.join(path_parts[4:]) if len(path_parts) > 4 else filename

                headless_logger.essential(f"Downloading {filename} from HuggingFace repo {repo_id} using hf_hub_download...")

                # Download using huggingface_hub with automatic checksums and resumption
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    revision=branch,
                    cache_dir=str(dest_folder),
                    resume_download=True,
                    local_files_only=False
                )

                # Copy from HF cache to target location if different
                if Path(downloaded_path) != dest_path:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded_path, dest_path)

                # Validate the downloaded file
                if filename.endswith('.safetensors') or 'lora' in filename.lower():
                    is_valid, validation_msg = validate_lora_file(dest_path, filename)
                    if not is_valid:
                        headless_logger.error(f"Downloaded file {filename} failed validation: {validation_msg}")
                        dest_path.unlink(missing_ok=True)
                        return False
                    headless_logger.essential(f"Successfully downloaded and validated {filename}. {validation_msg}")
                else:
                    headless_logger.essential(f"Successfully downloaded {filename} with integrity verification.")
                return True

        except ImportError:
            headless_logger.warning(f"huggingface_hub not available, falling back to requests for {url}")
        except (OSError, ValueError, requests.RequestException) as e:
            headless_logger.warning(f"HuggingFace download failed for {filename}: {e}, falling back to requests")

    # Fallback to requests with basic integrity checks
    try:
        headless_logger.essential(f"Downloading {filename} from {url} to {dest_folder}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        # Get expected content length for verification
        expected_size = int(response.headers.get('content-length', 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)

        # Verify download integrity
        actual_size = dest_path.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            headless_logger.error(f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}")
            dest_path.unlink(missing_ok=True)
            return False

        # Use comprehensive validation for LoRA files
        if filename.endswith('.safetensors') or 'lora' in filename.lower():
            is_valid, validation_msg = validate_lora_file(dest_path, filename)
            if not is_valid:
                headless_logger.error(f"Downloaded file {filename} failed validation: {validation_msg}")
                dest_path.unlink(missing_ok=True)
                return False
            headless_logger.essential(f"Successfully downloaded and validated {filename}. {validation_msg}")
        else:
            # For non-LoRA safetensors files, do basic format check
            if filename.endswith('.safetensors'):
                try:
                    import safetensors.torch as st
                    with st.safe_open(dest_path, framework="pt") as f:
                        pass  # Just verify it can be opened
                    headless_logger.essential(f"Successfully downloaded and verified safetensors file {filename}.")
                except ImportError:
                    headless_logger.debug_anomaly("WARNING", f"safetensors not available for verification of {filename}")
                except (OSError, ValueError, RuntimeError) as e:
                    headless_logger.error(f"Downloaded safetensors file {filename} appears corrupted: {e}")
                    dest_path.unlink(missing_ok=True)
                    return False
            else:
                headless_logger.essential(f"Successfully downloaded {filename}.")

        return True

    except (OSError, requests.RequestException, ValueError) as e:
        headless_logger.error(f"Failed to download {filename}: {e}")
        if dest_path.exists(): # Attempt to clean up partial download
            try: os.remove(dest_path)
            except OSError as e_cleanup:
                headless_logger.warning(f"Failed to clean up partial download {dest_path}: {e_cleanup}")
        return False

def _download_file_if_url(
    file_url_or_path: str,
    download_target_dir: Path | str | None,
    task_id_for_logging: str | None = "generic_task",
    descriptive_name: str | None = None,
    default_extension: str = ".jpg",
    default_stem: str = "file",
    file_type_label: str = "file",
    timeout: int = 300
) -> str:
    """
    Generic function to download a file from URL if needed.

    Args:
        file_url_or_path: URL or local path
        download_target_dir: Directory to save downloaded file
        task_id_for_logging: Task ID for logging
        descriptive_name: Optional descriptive name for the file
        default_extension: Default file extension if none detected
        default_stem: Default stem for filename if none detected
        file_type_label: Label for logging (e.g., "image", "video")
        timeout: Request timeout in seconds

    Returns:
        Local file path string if downloaded, otherwise returns the original string
    """
    if not file_url_or_path:
        return file_url_or_path

    parsed_url = urlparse(file_url_or_path)
    if parsed_url.scheme in ['http', 'https'] and download_target_dir:
        target_dir_path = Path(download_target_dir)
        try:
            target_dir_path.mkdir(parents=True, exist_ok=True)
            headless_logger.debug(f"Task {task_id_for_logging}: Downloading {file_type_label} from URL: {file_url_or_path} to {target_dir_path.resolve()}")

            original_filename = Path(parsed_url.path).name
            original_suffix = Path(original_filename).suffix if Path(original_filename).suffix else default_extension
            if not original_suffix.startswith('.'):
                original_suffix = '.' + original_suffix

            # Use descriptive naming if provided, otherwise fall back to improved default
            if descriptive_name:
                base_name_for_download = descriptive_name[:50]  # Limit length
            else:
                # Improved default naming
                cleaned_stem = Path(original_filename).stem[:30] if Path(original_filename).stem else default_stem
                base_name_for_download = f"input_{cleaned_stem}"

            # _get_unique_target_path expects a Path object for target_dir
            local_file_path = _get_unique_target_path(target_dir_path, base_name_for_download, original_suffix)
            _stream_download_to_path(file_url_or_path, local_file_path, timeout=timeout)

            headless_logger.debug(f"Task {task_id_for_logging}: Successfully downloaded {file_type_label} to {local_file_path}")
            return str(local_file_path)

        except (requests.exceptions.RequestException, Exception) as e:
            headless_logger.debug(f"Task {task_id_for_logging}: ERROR downloading {file_type_label} from {file_url_or_path}: {e}")
            raise DownloadError(str(e)) from e
        except OSError as e_dl:
            headless_logger.debug(f"Task {task_id_for_logging}: ERROR saving {file_type_label} to {target_dir_path}: {e_dl}", exc_info=True)
            raise DownloadError(str(e_dl)) from e_dl
    else:
        # Not a URL, or no download directory provided
        return file_url_or_path

def download_video_if_url(video_url_or_path: str, download_target_dir: Path | str | None, task_id_for_logging: str | None = "generic_task", descriptive_name: str | None = None) -> str:
    """
    Checks if the given string is an HTTP/HTTPS URL. If so, and if download_target_dir is provided,
    downloads the video to a unique path within download_target_dir.
    Returns the local file path string if downloaded, otherwise returns the original string.
    """
    return _download_file_if_url(
        video_url_or_path,
        download_target_dir,
        task_id_for_logging,
        descriptive_name,
        default_extension=".mp4",
        default_stem="structure_video",
        file_type_label="video",
        timeout=600  # 10 min timeout for larger videos
    )

def download_image_if_url(image_url_or_path: str, download_target_dir: Path | str | None, task_id_for_logging: str | None = "generic_task", debug_mode: bool = False, descriptive_name: str | None = None) -> str:
    """
    Checks if the given string is an HTTP/HTTPS URL. If so, and if download_target_dir is provided,
    downloads the image to a unique path within download_target_dir.
    Returns the local file path string if downloaded, otherwise returns the original string.

    Note: debug_mode parameter is kept for backwards compatibility but not currently used.
    """
    # Use the generic download function
    return _download_file_if_url(
        image_url_or_path,
        download_target_dir,
        task_id_for_logging,
        descriptive_name,
        default_extension=".jpg",
        default_stem="image",
        file_type_label="image",
        timeout=300  # 5 min timeout for images
    )
