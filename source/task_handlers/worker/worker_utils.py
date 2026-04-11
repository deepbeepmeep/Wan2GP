import os
import shutil
from pathlib import Path
from source.core.log import headless_logger
from source.core.constants import BYTES_PER_MB, BYTES_PER_GB



# RAM monitoring
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def log_ram_usage(label: str, task_id: str = "unknown") -> dict:
    """
    Log current RAM usage with a descriptive label.
    Returns dict with RAM metrics for programmatic use.
    """
    if not _PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / BYTES_PER_MB
        rss_gb = mem_info.rss / BYTES_PER_GB

        # Get system-wide memory stats
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / BYTES_PER_GB
        sys_available_gb = sys_mem.available / BYTES_PER_GB
        sys_used_percent = sys_mem.percent

        headless_logger.debug(
            f"[RAM] {label}: Process={rss_mb:.0f}MB ({rss_gb:.2f}GB) | "
            f"System={sys_used_percent:.1f}% used, {sys_available_gb:.1f}GB/{sys_total_gb:.1f}GB available",
            task_id=task_id
        )

        return {
            "available": True,
            "process_rss_mb": rss_mb,
            "process_rss_gb": rss_gb,
            "system_total_gb": sys_total_gb,
            "system_available_gb": sys_available_gb,
            "system_used_percent": sys_used_percent
        }

    except (ProcessLookupError, OSError) as e:
        headless_logger.warning(f"[RAM] Failed to get RAM usage: {e}", task_id=task_id)
        return {"available": False, "error": str(e)}

def cleanup_generated_files(output_location: str, task_id: str = "unknown", debug_mode: bool = False) -> None:
    """
    Delete generated files after successful task completion unless in debug mode.
    This includes the main output file/directory and any temporary files that may have been created.

    Args:
        output_location: Path to the generated file or directory to clean up
        task_id: Task ID for logging purposes
        debug_mode: Whether debug mode is enabled (skips cleanup if True)
    """
    if debug_mode:
        headless_logger.debug(f"Debug mode enabled - skipping file cleanup for {output_location}", task_id=task_id)
        return

    if not output_location:
        return

    try:
        file_path = Path(output_location)

        # Clean up main output file/directory
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size
            file_path.unlink()
            headless_logger.debug(f"Cleaned up generated file: {output_location} ({file_size:,} bytes)", task_id=task_id)
        elif file_path.exists() and file_path.is_dir():
            dir_size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
            shutil.rmtree(file_path)
            headless_logger.debug(f"Cleaned up generated directory: {output_location} ({dir_size:,} bytes)", task_id=task_id)
        else:
            headless_logger.debug(f"File cleanup skipped - path not found: {output_location}", task_id=task_id)

    except OSError as e:
        headless_logger.warning(f"Failed to cleanup generated file {output_location}: {e}", task_id=task_id)
