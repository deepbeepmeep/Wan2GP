"""Debug and monitoring utilities for travel between images tasks."""

import os
import threading
from pathlib import Path

# Import structured logging
from ...core.log import travel_logger

# RAM monitoring
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    travel_logger.warning("psutil not available - RAM monitoring disabled")

try:
    import cv2
    _COLOR_MATCH_DEPS_AVAILABLE = True
except ImportError:
    _COLOR_MATCH_DEPS_AVAILABLE = False

from ...utils import get_video_frame_count_and_fps
from source.core.constants import BYTES_PER_MB, MB_PER_GB, BYTES_PER_GB


class RamSnapshotCollector:
    """Thread-safe task-scoped RAM snapshot buffer."""

    def __init__(self) -> None:
        self._snapshots: dict[str, list[tuple[str, float, float, float, float]]] = {}
        self._lock = threading.Lock()

    def push(
        self,
        task_id: str,
        label: str,
        rss_mb: float,
        sys_available_gb: float,
        sys_total_gb: float,
        sys_used_percent: float,
    ) -> None:
        task_key = task_id or "unknown"
        with self._lock:
            self._snapshots.setdefault(task_key, []).append(
                (label, rss_mb, sys_available_gb, sys_total_gb, sys_used_percent)
            )

    def pop(self, task_id: str) -> list[tuple[str, float, float, float, float]]:
        task_key = task_id or "unknown"
        with self._lock:
            return self._snapshots.pop(task_key, [])


_RAM_SNAPSHOT_COLLECTOR = RamSnapshotCollector()


# Add debugging helper function
def debug_video_analysis(video_path: str | Path, label: str, task_id: str = "unknown") -> dict:
    """Analyze a video file and return comprehensive debug info"""
    try:
        path_obj = Path(video_path)
        if not path_obj.exists():
            travel_logger.debug(f"{label}: FILE MISSING - {video_path}", task_id=task_id)
            return {"exists": False, "path": str(video_path)}

        frame_count, fps = get_video_frame_count_and_fps(str(path_obj))
        file_size = path_obj.stat().st_size
        duration = frame_count / fps if fps and fps > 0 else 0

        debug_info = {
            "exists": True,
            "path": str(path_obj.resolve()),
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / BYTES_PER_MB, 2)
        }

        # Silent: debug_video_analysis is called from chain post-processing. The CHAIN
        # card already summarizes segment/run/path info. Color sampling still runs for
        # the return dict; only the log output is suppressed to reduce noise. If a caller
        # needs the color samples, they can read them from the returned debug_info dict.
        try:
            if frame_count and frame_count > 0:
                cap = cv2.VideoCapture(str(path_obj))
                if cap.isOpened():
                    sample_idxs = [0, max(0, int(frame_count // 2)), max(0, int(frame_count - 1))]
                    samples = {}
                    for idx in sample_idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                        ok, fr = cap.read()
                        if not ok or fr is None:
                            samples[str(idx)] = None
                            continue
                        # OpenCV frames are BGR uint8; mean per channel is robust for tint detection.
                        mean_bgr = tuple(float(x) for x in fr.mean(axis=(0, 1)))
                        samples[str(idx)] = {
                            "mean_bgr": mean_bgr,
                            "shape": tuple(int(x) for x in fr.shape),
                        }
                    cap.release()
                    debug_info["frame_color_samples"] = samples
        except (OSError, ValueError, RuntimeError):
            # Never fail the pipeline due to debug sampling.
            pass

        return debug_info

    except (OSError, ValueError, RuntimeError) as e:
        travel_logger.debug(f"{label}: ERROR analyzing video - {e}", task_id=task_id)
        return {"exists": False, "error": str(e), "path": str(video_path)}

# RAM monitoring helper function
def log_ram_usage(label: str, task_id: str = "unknown", logger=None) -> dict:
    """
    Log current RAM usage with a descriptive label.
    Returns dict with RAM metrics for programmatic use.
    """
    if logger is None:
        logger = travel_logger

    if not _PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / BYTES_PER_MB
        rss_gb = rss_mb / MB_PER_GB

        # Get system-wide memory stats
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / BYTES_PER_GB
        sys_available_gb = sys_mem.available / BYTES_PER_GB
        sys_used_percent = sys_mem.percent

        _RAM_SNAPSHOT_COLLECTOR.push(
            task_id,
            label,
            rss_mb,
            sys_available_gb,
            sys_total_gb,
            sys_used_percent,
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
        logger.warning(f"[RAM] Failed to get RAM usage: {e}", task_id=task_id)
        return {"available": False, "error": str(e)}


def flush_ram_snapshots(task_id: str = "unknown", logger=None) -> None:
    """Emit one structured RAM summary for the task and drop buffered snapshots.

    If every snapshot has the same rss_mb value (i.e. the task didn't meaningfully
    move memory) we collapse to a single 'stable' row instead of printing the same
    value four times.
    """
    if logger is None:
        logger = travel_logger

    if not _PSUTIL_AVAILABLE:
        return

    snapshots = _RAM_SNAPSHOT_COLLECTOR.pop(task_id)
    if not snapshots:
        return

    # If the RSS never moved, don't waste rows — emit one summary anomaly line instead of a card.
    unique_rss = {round(s[1]) for s in snapshots}
    if len(unique_rss) == 1:
        first = snapshots[0]
        value = (
            f"{first[1]:.0f}MB (sys {first[4]:.1f}%, "
            f"{first[2]:.1f}/{first[3]:.1f}GB) — stable across {len(snapshots)} checkpoint(s)"
        )
        logger.debug_anomaly("RAM", value, task_id=task_id)
        return

    rows: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    for label, rss_mb, sys_available_gb, sys_total_gb, sys_used_percent in snapshots:
        label_counts[label] = label_counts.get(label, 0) + 1
        row_label = label if label_counts[label] == 1 else f"{label} [{label_counts[label]}]"
        rows[row_label] = (
            f"{rss_mb:.0f}MB (sys {sys_used_percent:.1f}%, "
            f"{sys_available_gb:.1f}/{sys_total_gb:.1f}GB)"
        )

    logger.debug_block("RAM", rows, task_id=task_id)
