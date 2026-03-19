"""Frame extraction utilities for reading frames from video files."""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from source.core.log.video_runtime import generation_logger

__all__ = [
    "extract_frames_from_video",
    "extract_last_frame_as_image",
    "_COLOR_MATCH_DEPS_AVAILABLE",
]

# Flag for dependency availability
_COLOR_MATCH_DEPS_AVAILABLE = True


def extract_frames_from_video(input_video_path: str | Path, start_frame: int = 0, num_frames: int = None) -> list[np.ndarray]:
    """
    Extracts frames from a video file as numpy arrays with retry logic for recently encoded videos.

    Args:
        input_video_path: Path to the video file
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to extract (None = all remaining frames)

    Returns:
        List of frames as BGR numpy arrays
    """
    frames = []
    max_attempts = 3

    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(str(input_video_path))

        if not cap.isOpened():
            if attempt < max_attempts - 1:
                generation_logger.debug(f"Attempt {attempt + 1}: Could not open video {input_video_path}, retrying in 2 seconds...")
                time.sleep(2.0)
                continue
            else:
                generation_logger.error(f"Could not open video {input_video_path} after {max_attempts} attempts")
                return frames

        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check for valid frame count (recently encoded videos might report 0 initially)
        if total_frames_video <= 0:
            cap.release()
            if attempt < max_attempts - 1:
                generation_logger.debug(f"Attempt {attempt + 1}: Video {input_video_path} reports {total_frames_video} frames, retrying in 2 seconds...")
                time.sleep(2.0)
                continue
            else:
                generation_logger.error(f"Video {input_video_path} has invalid frame count {total_frames_video} after {max_attempts} attempts")
                return frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        frames_to_read = num_frames if num_frames is not None else (total_frames_video - start_frame)
        frames_to_read = min(frames_to_read, total_frames_video - start_frame)

        frames_extracted = 0
        for i in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                generation_logger.warning(f"Could not read frame {start_frame + i} from {input_video_path}")
                break
            frames.append(frame)
            frames_extracted += 1

        cap.release()

        # If we successfully extracted frames, return them
        if frames_extracted > 0:
            generation_logger.debug(f"Successfully extracted {frames_extracted} frames from {input_video_path} on attempt {attempt + 1}")
            return frames

        # If no frames extracted and we have more attempts, retry
        if attempt < max_attempts - 1:
            generation_logger.debug(f"Attempt {attempt + 1}: No frames extracted from {input_video_path}, retrying in 2 seconds...")
            frames = []  # Reset frames list for next attempt
            time.sleep(2.0)

    generation_logger.error(f"Failed to extract any frames from {input_video_path} after {max_attempts} attempts")
    return frames


def extract_last_frame_as_image(input_video_path: str | Path, output_dir: Path, task_id_for_log: str) -> str | None:
    """
    Extracts the last frame of a video and saves it as a PNG image.
    """
    if not _COLOR_MATCH_DEPS_AVAILABLE:
        generation_logger.warning(f"Task {task_id_for_log} extract_last_frame_as_image: Skipping due to missing CV2/Numpy dependencies.")
        return None
    try:
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            generation_logger.error(f"Task {task_id_for_log} extract_last_frame_as_image: Could not open video {input_video_path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            generation_logger.warning(f"Task {task_id_for_log} extract_last_frame_as_image: Video has 0 frames {input_video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            output_path = output_dir / f"last_frame_ref_{Path(input_video_path).stem}.png"
            # IMPORTANT: OpenCV frames are BGR. If we save with cv2.imwrite and later
            # read with PIL (RGB), colors will be channel-swapped and can appear as
            # a warm/brown tint. Save via PIL after converting to RGB to preserve
            # correct colors across libraries.
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(str(output_path))
            return str(output_path.resolve())
        generation_logger.warning(f"Task {task_id_for_log} extract_last_frame_as_image: Failed to read last frame from {input_video_path}")
        return None
    except (OSError, ValueError, RuntimeError) as e:
        generation_logger.error(f"Task {task_id_for_log} extract_last_frame_as_image: Exception extracting frame from {input_video_path}: {e}", exc_info=True)
        return None
