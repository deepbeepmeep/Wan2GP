"""
LoRA download and cleanup utilities.

This module provides:
- _download_lora_from_url: Download LoRAs from URLs (HuggingFace or direct)
- cleanup_legacy_lora_collisions: Remove collision-prone generic LoRA filenames

Note: LoRA format handling and URL detection are now in source/params/lora.py (LoRAConfig).
"""

import shutil
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import unquote

from source.core.log import model_logger

_LORA_SUFFIXES = {".safetensors", ".pt", ".ckpt"}
_ORPHAN_SUFFIXES = {".tmp", ".part", ".download"}


@dataclass(frozen=True)
class LoRACacheSweepResult:
    scanned_files: int
    removed_files: int
    removed_bytes: int
    removed_paths: tuple[str, ...]
    remaining_files: int
    remaining_bytes: int

    def to_dict(self) -> dict:
        return {
            "scanned_files": self.scanned_files,
            "removed_files": self.removed_files,
            "removed_bytes": self.removed_bytes,
            "removed_paths": list(self.removed_paths),
            "remaining_files": self.remaining_files,
            "remaining_bytes": self.remaining_bytes,
        }


def _download_lora_from_url(url: str, task_id: str, model_type: str = None) -> str:
    """
    Download a LoRA from URL to appropriate local directory.

    Args:
        url: LoRA download URL
        task_id: Task ID for logging
        model_type: Model type to determine correct LoRA directory (e.g., "wan_2_2_vace_lightning_baseline_2_2_2")

    Returns:
        Local filename of downloaded LoRA

    Raises:
        Exception: If download fails
    """
    # Use absolute paths based on this file's location to avoid working directory issues
    repo_root = Path(__file__).parent.parent.parent.parent
    wan_dir = repo_root / "Wan2GP"

    # Extract filename from URL and decode URL-encoded characters
    # e.g., "%E5%BB%B6%E6%97%B6%E6%91%84%E5%BD%B1-high.safetensors" → "延时摄影-high.safetensors"
    url_filename = url.split("/")[-1]
    generic_filename = url_filename  # Save original before modification

    # Handle Wan2.2 Lightning LoRA collisions by prefixing parent folder
    if url_filename in ["high_noise_model.safetensors", "low_noise_model.safetensors"]:
        parts = url.split("/")
        if len(parts) > 2:
            parent = parts[-2].replace("%20", "_")
            url_filename = f"{parent}_{url_filename}"

    local_filename = unquote(url_filename)

    # If we derived a unique filename (collision detected), clean up old generic file
    if local_filename != generic_filename:
        model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Collision-prone LoRA detected: {generic_filename} -> {local_filename}", task_id=task_id)

        # Check ALL standard lora directories (using centralized paths)
        from source.models.lora.lora_paths import get_lora_search_dirs
        lora_search_dirs = get_lora_search_dirs(wan_dir, repo_root)

        for search_dir in lora_search_dirs:
            if search_dir.is_dir():
                old_path = search_dir / generic_filename
                if old_path.is_file():
                    model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Removing legacy LoRA file: {old_path}", task_id=task_id)
                    try:
                        old_path.unlink()
                        model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Successfully deleted legacy file", task_id=task_id)
                    except OSError as e:
                        model_logger.warning(f"[LORA_DOWNLOAD] Task {task_id}: Failed to delete old LoRA {old_path}: {e}", task_id=task_id)

    # Enforce configured cache limits before admitting another downloaded LoRA.
    sweep_lora_cache_from_env(task_id=task_id)

    # Determine LoRA directory based on model type (centralized in lora_paths.py)
    from source.models.lora.lora_paths import get_lora_dir_for_model
    lora_dir = get_lora_dir_for_model(model_type, wan_dir)

    local_path = lora_dir / local_filename

    model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Downloading {local_filename} to {lora_dir} from {url}", task_id=task_id)

    # Normalize HuggingFace URLs: convert /blob/ to /resolve/ for direct downloads
    if "huggingface.co/" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
        model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Normalized HuggingFace URL from /blob/ to /resolve/", task_id=task_id)

    # Check if file already exists
    if not local_path.is_file():
        if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
            # Use HuggingFace hub for HF URLs
            from huggingface_hub import hf_hub_download

            # Parse HuggingFace URL
            url_path = url[len("https://huggingface.co/"):]
            url_parts = url_path.split("/resolve/main/")
            repo_id = url_parts[0]
            rel_path_encoded = url_parts[-1]
            # Decode URL-encoded path components (e.g., Chinese characters)
            rel_path = unquote(rel_path_encoded)
            filename = Path(rel_path).name
            subfolder = Path(rel_path).parent.as_posix() if Path(rel_path).parent != Path(".") else ""

            # Ensure LoRA directory exists
            lora_dir.mkdir(parents=True, exist_ok=True)

            # Download using HuggingFace hub. Some hubs require `subfolder` to locate
            # the file, but we want the final artifact at `loras/<filename>` because
            # WGP expects LoRAs in the root loras directory.
            if len(subfolder) > 0:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(lora_dir), subfolder=subfolder)
                # If the file landed under a nested path, move it up to lora_dir
                nested_path = lora_dir / subfolder / filename
                if nested_path.exists() and not local_path.exists():
                    try:
                        lora_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(nested_path), str(local_path))
                        # Clean up empty subfolder tree if any
                        try:
                            # Remove empty dirs going up from the deepest
                            cur = lora_dir / subfolder
                            while cur.is_relative_to(lora_dir) and cur != lora_dir:
                                if not any(cur.iterdir()):
                                    cur.rmdir()
                                cur = cur.parent
                        except OSError as e_rmdir:
                            model_logger.debug(f"Could not remove empty LoRA subfolder during cleanup: {e_rmdir}")
                    except OSError:
                        # If move fails, leave as-is; higher-level checks may still find it
                        pass
            else:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(lora_dir))
        else:
            # Use urllib for other URLs
            lora_dir.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, str(local_path))
        
        model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: Successfully downloaded {local_filename}", task_id=task_id)
    else:
        model_logger.debug_anomaly("LORA_DOWNLOAD", f"Task {task_id}: {local_filename} already exists", task_id=task_id)

    sweep_lora_cache_from_env(task_id=task_id, protected_paths=[local_path])
    return local_filename


def cleanup_legacy_lora_collisions():
    """
    Remove legacy generic LoRA filenames that collide with new uniquely-named versions.
    
    This runs at worker startup to ensure old collision-prone files like
    'high_noise_model.safetensors' and 'low_noise_model.safetensors' are removed
    before WGP loads models with updated LoRA URLs.
    
    Checks ALL possible LoRA directories to ensure comprehensive cleanup.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    wan_dir = repo_root / "Wan2GP"
    
    # Comprehensive list of all possible LoRA directories
    lora_dirs = [
        # Wan2GP subdirectories (standard)
        wan_dir / "loras",
        wan_dir / "loras" / "wan",
        wan_dir / "loras_i2v",
        wan_dir / "loras_hunyuan",
        wan_dir / "loras_hunyuan" / "1.5",
        wan_dir / "loras_hunyuan_i2v",
        wan_dir / "loras_flux",
        wan_dir / "loras_qwen",
        wan_dir / "loras_ltxv",
        wan_dir / "loras" / "ltx2",
        wan_dir / "loras_kandinsky5",
        # Parent directory (for stray files from previous bugs)
        repo_root / "loras",
        repo_root / "loras" / "wan",
        repo_root / "loras_qwen",
    ]
    
    # Generic filenames that are collision-prone
    collision_prone_files = [
        "high_noise_model.safetensors",
        "low_noise_model.safetensors",
    ]
    
    cleaned_files = []
    for lora_dir in lora_dirs:
        if not lora_dir.exists():
            continue
        
        for filename in collision_prone_files:
            file_path = lora_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
                    model_logger.debug(f"🗑️  Removed legacy LoRA file: {file_path}")
                except OSError as e:
                    model_logger.warning(f"⚠️  Failed to remove legacy LoRA {file_path}: {e}")
    
    if cleaned_files:
        model_logger.debug(f"Cleanup complete: removed {len(cleaned_files)} legacy LoRA file(s)")
    else:
        model_logger.debug("No legacy LoRA files found to clean up")


def sweep_lora_cache_from_env(
    *,
    task_id: str | None = None,
    protected_paths: list[Path] | None = None,
) -> LoRACacheSweepResult:
    """Apply configured LoRA cache size, count, age, and orphan limits."""
    return sweep_lora_cache(
        max_bytes=_optional_int_env("REIGH_LORA_CACHE_MAX_BYTES"),
        max_files=_optional_int_env("REIGH_LORA_CACHE_MAX_FILES"),
        max_age_seconds=_optional_int_env("REIGH_LORA_CACHE_MAX_AGE_SECONDS"),
        orphan_max_age_seconds=_int_env("REIGH_LORA_ORPHAN_MAX_AGE_SECONDS", 3600),
        cache_dirs=_lora_cache_dirs_from_env(),
        protected_paths=protected_paths,
        task_id=task_id,
    )


def sweep_lora_cache(
    *,
    max_bytes: int | None = None,
    max_files: int | None = None,
    max_age_seconds: int | None = None,
    orphan_max_age_seconds: int | None = None,
    cache_dirs: list[Path] | None = None,
    protected_paths: list[Path] | None = None,
    now: float | None = None,
    task_id: str | None = None,
) -> LoRACacheSweepResult:
    """
    Sweep the existing LoRA cache implementation using size/count/age limits.

    Only known LoRA model files and stale partial-download files inside LoRA cache
    directories are eligible. Symlinks and unrelated files are left untouched.
    """
    now = time.time() if now is None else now
    dirs = cache_dirs if cache_dirs is not None else _default_lora_cache_dirs()
    protected = {
        path.expanduser().resolve(strict=False)
        for path in (protected_paths or [])
    }
    entries = _collect_lora_cache_entries(dirs)
    scanned_files = len(entries)
    removed: list[tuple[Path, int]] = []

    def _remove(path: Path, size: int) -> bool:
        if path.expanduser().resolve(strict=False) in protected:
            return False
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            model_logger.warning(f"[LORA_CACHE] Failed to remove {path}: {exc}", task_id=task_id)
            return False
        removed.append((path, size))
        return True

    retained: set[Path] = set()
    if orphan_max_age_seconds is not None and orphan_max_age_seconds > 0:
        for entry in entries:
            path, size, mtime, is_orphan = entry
            if is_orphan and now - mtime >= orphan_max_age_seconds:
                _remove(path, size)
            else:
                retained.add(path)

    entries = [entry for entry in entries if entry[0].exists()]

    if max_age_seconds is not None and max_age_seconds > 0:
        for path, size, mtime, is_orphan in entries:
            if is_orphan or path not in retained:
                continue
            if now - mtime >= max_age_seconds:
                _remove(path, size)

    entries = [entry for entry in entries if entry[0].exists() and not entry[3]]
    remaining_bytes = sum(size for _path, size, _mtime, _orphan in entries)
    remaining_files = len(entries)

    if max_files is not None and max_files > 0 and remaining_files > max_files:
        for path, size, _mtime, _is_orphan in sorted(entries, key=lambda item: (item[2], str(item[0]))):
            if remaining_files <= max_files:
                break
            if _remove(path, size):
                remaining_files -= 1
                remaining_bytes -= size

    entries = [entry for entry in entries if entry[0].exists()]
    remaining_bytes = sum(size for _path, size, _mtime, _orphan in entries)

    if max_bytes is not None and max_bytes > 0 and remaining_bytes > max_bytes:
        for path, size, _mtime, _is_orphan in sorted(entries, key=lambda item: (item[2], str(item[0]))):
            if remaining_bytes <= max_bytes:
                break
            if _remove(path, size):
                remaining_bytes -= size

    final_entries = [entry for entry in _collect_lora_cache_entries(dirs) if not entry[3]]
    result = LoRACacheSweepResult(
        scanned_files=scanned_files,
        removed_files=len(removed),
        removed_bytes=sum(size for _path, size in removed),
        removed_paths=tuple(str(path) for path, _size in removed),
        remaining_files=len(final_entries),
        remaining_bytes=sum(size for _path, size, _mtime, _orphan in final_entries),
    )
    if result.removed_files:
        model_logger.warning(
            "[LORA_CACHE] Swept "
            f"{result.removed_files} file(s), freed {result.removed_bytes} bytes",
            task_id=task_id,
        )
    return result


def _collect_lora_cache_entries(cache_dirs: list[Path]) -> list[tuple[Path, int, float, bool]]:
    entries: list[tuple[Path, int, float, bool]] = []
    seen: set[Path] = set()
    for directory in cache_dirs:
        try:
            root = directory.expanduser().resolve(strict=False)
        except OSError:
            continue
        if root in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(root)
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            is_lora = suffix in _LORA_SUFFIXES
            is_orphan = suffix in _ORPHAN_SUFFIXES
            if not is_lora and not is_orphan:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path, stat.st_size, stat.st_mtime, is_orphan))
    return entries


def _lora_cache_dirs_from_env() -> list[Path] | None:
    configured = os.environ.get("REIGH_LORA_CACHE_DIRS") or os.environ.get("REIGH_LORA_CACHE_DIR")
    if not configured:
        return None
    return [Path(part) for part in configured.split(":") if part]


def _default_lora_cache_dirs() -> list[Path]:
    repo_root = Path(__file__).parent.parent.parent.parent
    wan_dir = repo_root / "Wan2GP"
    from source.models.lora.lora_paths import get_lora_search_dirs

    return get_lora_search_dirs(wan_dir, repo_root)


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Public alias for cross-module use.
download_lora_from_url = _download_lora_from_url
