"""Video metadata retrieval using ffprobe and OpenCV."""
import json
import subprocess
import time

import cv2

from source.core.log import generation_logger

__all__ = [
    "VideoMetadataError",
    "get_video_frame_count_ffprobe",
    "get_video_fps_ffprobe",
    "get_video_frame_count_and_fps",
]


class VideoMetadataError(RuntimeError):
    """Raised when required video metadata cannot be determined."""


def get_video_frame_count_ffprobe(input_video_path: str) -> int | None:
    """
    Get accurate frame count using ffprobe (more reliable than OpenCV).

    Uses ffprobe to count actual frames in the video stream, which is more
    accurate than OpenCV's CAP_PROP_FRAME_COUNT (especially for VFR videos
    or videos with certain codecs).

    Args:
        input_video_path: Path to video file

    Returns:
        Frame count, or None on error
    """
    try:
        # Method 1: Try to get frame count from stream metadata (fast)
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            str(input_video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and result.stdout.strip():
            frame_count = int(result.stdout.strip())
            if frame_count > 0:
                return frame_count

        # Method 2: Fallback to counting frames (slower but reliable)
        generation_logger.debug(f"[FFPROBE] Metadata count failed, counting frames manually...")
        cmd2 = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'csv=p=0',
            str(input_video_path)
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)

        if result2.returncode == 0 and result2.stdout.strip():
            frame_count = int(result2.stdout.strip())
            if frame_count > 0:
                return frame_count

        return None

    except subprocess.TimeoutExpired:
        generation_logger.warning(f"[FFPROBE] Timeout getting frame count for {input_video_path}")
        return None
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        generation_logger.error(f"[FFPROBE] Error: {e}")
        return None


def _parse_ffprobe_rate(rate_str: str) -> float | None:
    """
    Parse an ffprobe rate string (e.g. '30000/1001', '24/1', '0/0').
    Returns float fps or None.
    """
    try:
        s = (rate_str or "").strip()
        if not s:
            return None
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                return None
            val = num / den
        else:
            val = float(s)
        if val <= 0:
            return None
        return val
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def get_video_fps_ffprobe(input_video_path: str) -> float | None:
    """
    Get FPS using ffprobe (avg_frame_rate preferred; falls back to r_frame_rate).
    This is more reliable than OpenCV for VFR content.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json",
            str(input_video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not result.stdout:
            return None

        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        st = streams[0]

        avg = _parse_ffprobe_rate(st.get("avg_frame_rate"))
        r = _parse_ffprobe_rate(st.get("r_frame_rate"))

        # Prefer avg if present; fall back to r_frame_rate
        fps = avg or r
        if fps and fps > 0:
            return fps
        return None
    except subprocess.TimeoutExpired:
        generation_logger.warning(f"[FFPROBE] Timeout getting FPS for {input_video_path}")
        return None
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        generation_logger.error(f"[FFPROBE] Error getting FPS: {e}")
        return None


def get_video_frame_count_and_fps(input_video_path: str) -> tuple[int, float] | tuple[None, None]:
    """
    Get frame count and FPS from a video file using OpenCV.

    WARNING: OpenCV's CAP_PROP_FRAME_COUNT can be inaccurate for some videos.
    For critical frame-accurate operations, use get_video_frame_count_ffprobe() instead.
    """

    # Try multiple times in case the video metadata is still being written
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            if attempt < max_attempts - 1:
                time.sleep(0.5)  # Wait a bit before retrying
                continue
            return None, None

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # If we got valid values, return them
        if frames > 0 and fps > 0:
            return frames, fps

        # Otherwise, wait and retry
        if attempt < max_attempts - 1:
            time.sleep(0.5)

    # If all attempts failed, return what we got (might be 0 or invalid)
    return frames, fps
