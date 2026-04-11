"""Download and extract motion frames from pre-warped structure videos."""

import sys
import numpy as np
from pathlib import Path
from typing import List
from source.core.constants import BYTES_PER_MB
from source.core.log import generation_logger

__all__ = [
    "download_and_extract_motion_frames",
]


def download_and_extract_motion_frames(
    structure_motion_video_url: str,
    frame_start: int,
    frame_count: int,
    download_dir: Path,
) -> List[np.ndarray]:
    """
    Download pre-warped structure motion video and extract needed frames.

    This is the segment-level function that downloads the orchestrator's
    pre-computed motion video and extracts only the frames this segment needs.

    Args:
        structure_motion_video_url: URL or path to the pre-warped motion video
        frame_start: Starting frame index to extract
        frame_count: Number of frames to extract
        download_dir: Directory to download video to

    Returns:
        List of numpy arrays [H, W, C] uint8 RGB

    Raises:
        ValueError: If video cannot be downloaded or frames extracted
    """
    from urllib.parse import urlparse
    import requests

    try:
        # Determine if this is a URL or local path
        parsed = urlparse(structure_motion_video_url)
        is_url = parsed.scheme in ['http', 'https']

        if is_url:
            # Download the video
            download_dir.mkdir(parents=True, exist_ok=True)
            local_video_path = download_dir / "structure_motion.mp4"

            response = requests.get(structure_motion_video_url, timeout=120)
            response.raise_for_status()

            with open(local_video_path, 'wb') as f:
                f.write(response.content)
        else:
            # Use local path (support plain paths and file:// URLs)
            if parsed.scheme == "file":
                local_video_path = Path(parsed.path)
            else:
                local_video_path = Path(structure_motion_video_url)
            if not local_video_path.exists():
                raise ValueError(f"Structure motion video not found: {local_video_path}")

        # Extract frames (prefer decord, fall back to cv2)

        try:
            # Add Wan2GP to path (decord often installed alongside this stack)
            wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
            if str(wan_dir) not in sys.path:
                sys.path.insert(0, str(wan_dir))

            import decord  # type: ignore
            decord.bridge.set_bridge('torch')

            vr = decord.VideoReader(str(local_video_path))
            total_frames = len(vr)

            if frame_start >= total_frames:
                raise ValueError(f"frame_start {frame_start} >= total frames {total_frames}")

            actual_frame_count = min(frame_count, total_frames - frame_start)
            if actual_frame_count < frame_count:
                generation_logger.warning(f"[STRUCTURE_MOTION] Only {actual_frame_count} frames available (requested {frame_count})")

            frame_indices = list(range(frame_start, frame_start + actual_frame_count))
            frames_tensor = vr.get_batch(frame_indices)  # torch tensor [T, H, W, C]

            frames_list: List[np.ndarray] = []
            for i in range(len(frames_tensor)):
                frame_np = frames_tensor[i].cpu().numpy()
                if frame_np.dtype != np.uint8:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frames_list.append(frame_np)

            generation_logger.debug_block(
                "STRUCTURE_MOTION_EXTRACT",
                {
                    "frames": len(frames_list),
                    "backend": "decord",
                    "source": str(local_video_path),
                },
            )
            return frames_list

        except ModuleNotFoundError:
            import cv2

            cap = cv2.VideoCapture(str(local_video_path))
            if not cap.isOpened():
                raise ValueError(f"cv2 failed to open video: {local_video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total_frames <= 0:
                # Some codecs don't report frame count; we'll read until EOF.
                total_frames = 10**9

            if frame_start >= total_frames:
                cap.release()
                raise ValueError(f"frame_start {frame_start} >= total frames {total_frames}")

            # Seek to frame_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

            frames_list: List[np.ndarray] = []
            for _ in range(frame_count):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
                frames_list.append(frame_rgb)

            cap.release()
            generation_logger.debug_block(
                "STRUCTURE_MOTION_EXTRACT",
                {
                    "frames": len(frames_list),
                    "backend": "cv2",
                    "source": str(local_video_path),
                },
            )
            return frames_list

    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"[ERROR] Failed to extract motion frames: {e}", exc_info=True)
        raise
