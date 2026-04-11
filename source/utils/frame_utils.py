"""Frame manipulation utilities: loading, creating, resizing, video segment operations."""

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from source.core.log import headless_logger
from source.utils.download_utils import download_image_if_url

__all__ = [
    "create_color_frame",
    "get_easing_function",
    "create_video_from_frames_list",
    "extract_video_segment_ffmpeg",
    "stitch_videos_ffmpeg",
    "save_frame_from_video",
    "extract_specific_frame_ffmpeg",
    "concatenate_videos_ffmpeg",
    "get_video_frame_count_and_fps",
    "image_to_frame",
    "apply_strength_to_image",
    "get_image_dimensions_pil",
    "adjust_frame_brightness",
    "get_sequential_target_path",
    "load_pil_images",
]

def _image_to_frame_simple(image_path: str | Path, target_size: tuple[int, int]) -> np.ndarray | None:
    """Loads an image, resizes it, and converts to BGR NumPy array for OpenCV.

    This is the original simple version kept as a private helper.  The public
    ``image_to_frame`` function (below) supports URL downloading and is the
    one exported by the package.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        headless_logger.error(f"Image file not found at {image_path}")
        return None
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.error(f"Error loading or processing image {image_path}: {e}")
        return None

def create_color_frame(size: tuple[int, int], color_bgr: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Creates a single color BGR frame (default black)."""
    height, width = size[1], size[0] # size is (width, height)
    frame = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    return frame

def get_easing_function(name: str):
    """
    Returns an easing function by name.
    """
    if name == 'linear':
        return lambda t: t
    elif name == 'ease_in_quad':
        return lambda t: t * t
    elif name == 'ease_out_quad':
        return lambda t: t * (2 - t)
    elif name == 'ease_in_out_quad' or name == 'ease_in_out': # Added alias
        return lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    elif name == 'ease_in_cubic':
        return lambda t: t * t * t
    elif name == 'ease_out_cubic':
        return lambda t: 1 + ((t - 1) ** 3)
    elif name == 'ease_in_out_cubic':
        return lambda t: 4 * t * t * t if t < 0.5 else 1 + ((2 * t - 2) ** 3) / 2
    elif name == 'ease_in_quart':
        return lambda t: t * t * t * t
    elif name == 'ease_out_quart':
        return lambda t: 1 - ((t - 1) ** 4)
    elif name == 'ease_in_out_quart':
        return lambda t: 8 * t * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 4) / 2
    elif name == 'ease_in_quint':
        return lambda t: t * t * t * t * t
    elif name == 'ease_out_quint':
        return lambda t: 1 + ((t - 1) ** 5)
    elif name == 'ease_in_out_quint':
        return lambda t: 16 * t * t * t * t * t if t < 0.5 else 1 + ((-2 * t + 2) ** 5) / 2
    elif name == 'ease_in_sine':
        return lambda t: 1 - math.cos(t * math.pi / 2)
    elif name == 'ease_out_sine':
        return lambda t: math.sin(t * math.pi / 2)
    elif name == 'ease_in_out_sine':
        return lambda t: -(math.cos(math.pi * t) - 1) / 2
    elif name == 'ease_in_expo':
        return lambda t: 0 if t == 0 else 2 ** (10 * (t - 1))
    elif name == 'ease_out_expo':
        return lambda t: 1 if t == 1 else 1 - 2 ** (-10 * t)
    elif name == 'ease_in_out_expo':
        def _ease_in_out_expo(t):
            if t == 0: return 0
            if t == 1: return 1
            if t < 0.5:
                return (2 ** (20 * t - 10)) / 2
            else:
                return (2 - 2 ** (-20 * t + 10)) / 2
        return _ease_in_out_expo
    else: # Default to ease_in_out
        return lambda t: 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int] # width, height
):
    """Creates an MP4 video from a list of NumPy BGR frames.

    This is a wrapper around source.media.video.create_video_from_frames_list
    with BT.709 colorspace standardization enabled by default.
    """
    # Import here to avoid circular imports
    from source.media.video import create_video_from_frames_list as _create_video
    return _create_video(
        frames_list,
        output_path,
        fps,
        resolution,
        standardize_colorspace=True
    )

def extract_video_segment_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    start_frame_index: int, # 0-indexed
    num_frames_to_keep: int,
    input_fps: float, # FPS of the input video for accurate -ss calculation
    resolution: tuple[int,int]
):
    """Extracts a video segment using FFmpeg with stream copy if possible."""
    headless_logger.debug(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Called with input='{input_video_path}', output='{output_video_path}', start_idx={start_frame_index}, num_frames={num_frames_to_keep}, input_fps={input_fps}")
    if num_frames_to_keep <= 0:
        headless_logger.warning(f"num_frames_to_keep is {num_frames_to_keep} for {output_video_path} (FFmpeg). Nothing to extract.")
        headless_logger.debug("EXTRACT_VIDEO_SEGMENT_FFMPEG: num_frames_to_keep is 0 or less, returning.")
        Path(output_video_path).unlink(missing_ok=True)
        return

    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    start_time_seconds = start_frame_index / input_fps

    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time_seconds),
        '-i', str(input_video_path.resolve()),
        '-vframes', str(num_frames_to_keep),
        '-an',
        str(output_video_path.resolve())
    ]

    headless_logger.debug(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
        headless_logger.debug(f"EXTRACT_VIDEO_SEGMENT_FFMPEG: Successfully extracted segment to {output_video_path}")
        if process.stderr:
            headless_logger.debug(f"FFmpeg stderr (for {output_video_path}):\n{process.stderr}")
        if not output_video_path.exists() or output_video_path.stat().st_size == 0:
            headless_logger.error(f"FFmpeg command for {output_video_path} apparently succeeded but output file is missing or empty.")
            headless_logger.debug(f"FFmpeg command for {output_video_path} produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")

    except subprocess.CalledProcessError as e:
        headless_logger.error(f"Error during FFmpeg segment extraction for {output_video_path}:")
        headless_logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
        if e.stdout: headless_logger.error(f"FFmpeg stdout:\n{e.stdout}")
        if e.stderr: headless_logger.error(f"FFmpeg stderr:\n{e.stderr}")
        headless_logger.debug(f"FFmpeg extraction failed for {output_video_path}. Error: {e}")
        Path(output_video_path).unlink(missing_ok=True)
    except FileNotFoundError:
        headless_logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        headless_logger.debug("FFmpeg command not found during segment extraction.")
        raise

def stitch_videos_ffmpeg(video_paths_list: list[str | Path], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_paths_list:
        headless_logger.warning("No videos to stitch.")
        return

    valid_video_paths = []
    for p in video_paths_list:
        resolved_p = Path(p).resolve()
        if resolved_p.exists() and resolved_p.stat().st_size > 0:
            valid_video_paths.append(resolved_p)
        else:
            headless_logger.warning(f"Video segment {resolved_p} is missing or empty. Skipping from stitch list.")

    if not valid_video_paths:
        headless_logger.warning("No valid video segments found to stitch after checks.")
        return

    with tempfile.TemporaryDirectory(prefix="ffmpeg_concat_") as tmpdir:
        filelist_path = Path(tmpdir) / "ffmpeg_filelist.txt"
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for video_path in valid_video_paths:
                f.write(f"file '{video_path.as_posix()}'\n")

        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(filelist_path),
            '-c', 'copy', str(output_path)
        ]

        headless_logger.debug(f"Running ffmpeg to stitch videos: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
            headless_logger.essential(f"Successfully stitched videos into: {output_path}")
            if process.stderr: headless_logger.debug(f"FFmpeg log (stderr):\n{process.stderr}")
        except subprocess.CalledProcessError as e:
            headless_logger.error(f"Error during ffmpeg stitching for {output_path}:")
            headless_logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
            if e.stdout: headless_logger.error(f"FFmpeg stdout:\n{e.stdout}")
            if e.stderr: headless_logger.error(f"FFmpeg stderr:\n{e.stderr}")
            raise
        except FileNotFoundError:
            headless_logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            raise

def save_frame_from_video(input_video_path: Path, frame_index: int, output_image_path: Path, resolution: tuple[int, int]):
    """Extracts a specific frame from a video, resizes, and saves it as an image."""
    headless_logger.debug(f"SAVE_FRAME_FROM_VIDEO: Input='{input_video_path}', Index={frame_index}, Output='{output_image_path}', Res={resolution}")
    if not input_video_path.exists() or input_video_path.stat().st_size == 0:
        headless_logger.error(f"Video file for frame extraction not found or empty: {input_video_path}")
        return False

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        headless_logger.error(f"Could not open video file: {input_video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Allow Python-style negative indexing (e.g. -1 for last frame)
    if frame_index < 0:
        frame_index = total_frames + frame_index  # Convert to positive index

    if frame_index < 0 or frame_index >= total_frames:
        headless_logger.error(f"Frame index {frame_index} is out of bounds for video {input_video_path} (total frames: {total_frames}).")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        headless_logger.error(f"Could not read frame {frame_index} from {input_video_path}.")
        return False

    try:
        if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
            headless_logger.debug(f"SAVE_FRAME_FROM_VIDEO: Resizing frame from {frame.shape[:2]} to {resolution[:2][::-1]}")
            frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), frame)
        headless_logger.essential(f"Successfully saved frame {frame_index} from {input_video_path} to {output_image_path}")
        return True
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.error(f"Error saving frame to {output_image_path}: {e}", exc_info=True)
        return False

# --- FFMPEG-based specific frame extraction ---
def extract_specific_frame_ffmpeg(
    input_video_path: str | Path,
    frame_number: int, # 0-indexed
    output_image_path: str | Path,
    input_fps: float # Passed by caller, though not strictly needed for ffmpeg frame index selection using 'eq(n,frame_number)'
):
    """Extracts a specific frame from a video using FFmpeg and saves it as an image."""
    headless_logger.debug(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Input='{input_video_path}', Frame={frame_number}, Output='{output_image_path}'")
    input_video_p = Path(input_video_path)
    output_image_p = Path(output_image_path)
    output_image_p.parent.mkdir(parents=True, exist_ok=True)

    if not input_video_p.exists() or input_video_p.stat().st_size == 0:
        headless_logger.error(f"Input video for frame extraction not found or empty: {input_video_p}")
        headless_logger.debug(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Input video {input_video_p} not found or empty. Returning False.")
        return False

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output without asking
        '-i', str(input_video_p.resolve()),
        '-vf', f"select=eq(n\\,{frame_number})", # Escaped comma for ffmpeg filter syntax
        '-vframes', '1',
        str(output_image_p.resolve())
    ]

    headless_logger.debug(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
        headless_logger.debug(f"EXTRACT_SPECIFIC_FRAME_FFMPEG: Successfully extracted frame {frame_number} to {output_image_p}")
        if process.stderr:
            headless_logger.debug(f"FFmpeg stderr (for frame extraction to {output_image_p}):\n{process.stderr}")
        if not output_image_p.exists() or output_image_p.stat().st_size == 0:
            headless_logger.error(f"FFmpeg command for frame extraction to {output_image_p} apparently succeeded but output file is missing or empty.")
            headless_logger.debug(f"FFmpeg command for {output_image_p} (frame extraction) produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        headless_logger.error(f"Error during FFmpeg frame extraction for {output_image_p}:")
        headless_logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
        if e.stdout: headless_logger.error(f"FFmpeg stdout:\n{e.stdout}")
        if e.stderr: headless_logger.error(f"FFmpeg stderr:\n{e.stderr}")
        headless_logger.debug(f"FFmpeg frame extraction failed for {output_image_p}. Error: {e}")
        if output_image_p.exists(): output_image_p.unlink(missing_ok=True)
        return False
    except FileNotFoundError:
        headless_logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        headless_logger.debug("FFmpeg command not found during frame extraction.")
        raise

# --- FFMPEG-based video concatenation (alternative to stitch_videos_ffmpeg if caller manages temp dir) ---
def concatenate_videos_ffmpeg(
    video_paths: list[str | Path],
    output_path: str | Path,
    temp_dir_for_list: str | Path # Directory where the list file will be created
):
    """Concatenates multiple video files into one using FFmpeg, using a provided temp directory for the list file."""
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    temp_dir_p = Path(temp_dir_for_list)
    temp_dir_p.mkdir(parents=True, exist_ok=True)

    if not video_paths:
        headless_logger.warning("No videos to concatenate.")
        headless_logger.debug("CONCATENATE_VIDEOS_FFMPEG: No video paths provided. Returning.")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        return

    valid_video_paths = []
    for p_item in video_paths:
        resolved_p_item = Path(p_item).resolve()
        if resolved_p_item.exists() and resolved_p_item.stat().st_size > 0:
            valid_video_paths.append(resolved_p_item)
        else:
            headless_logger.warning(f"Video segment {resolved_p_item} for concatenation is missing or empty. Skipping.")
            headless_logger.debug(f"CONCATENATE_VIDEOS_FFMPEG: Skipping invalid video segment {resolved_p_item}")

    if not valid_video_paths:
        headless_logger.warning("No valid video segments found to concatenate after checks.")
        headless_logger.debug("CONCATENATE_VIDEOS_FFMPEG: No valid video segments. Returning.")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        return

    filelist_path = temp_dir_p / "ffmpeg_concat_filelist.txt"
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for video_path_item in valid_video_paths:
            f.write(f"file '{video_path_item.as_posix()}'\n") # Use as_posix() for ffmpeg list file

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(filelist_path.resolve()),
        '-c', 'copy',
        str(output_p.resolve())
    ]

    headless_logger.debug(f"CONCATENATE_VIDEOS_FFMPEG: Running command: {' '.join(cmd)} with list file {filelist_path}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
        headless_logger.essential(f"Successfully concatenated videos into: {output_p}")
        headless_logger.debug(f"CONCATENATE_VIDEOS_FFMPEG: Success. Output: {output_p}")
        if process.stderr:
            headless_logger.debug(f"FFmpeg stderr (for concatenation to {output_p}):\n{process.stderr}")
        if not output_p.exists() or output_p.stat().st_size == 0:
             headless_logger.warning(f"FFmpeg concatenation to {output_p} apparently succeeded but output file is missing or empty.")
             headless_logger.debug(f"FFmpeg command for {output_p} (concatenation) produced no output. stdout:\n{process.stdout}\nstderr:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        headless_logger.error(f"Error during FFmpeg concatenation for {output_p}:")
        headless_logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
        if e.stdout: headless_logger.error(f"FFmpeg stdout:\n{e.stdout}")
        if e.stderr: headless_logger.error(f"FFmpeg stderr:\n{e.stderr}")
        headless_logger.debug(f"FFmpeg concatenation failed for {output_p}. Error: {e}")
        if output_p.exists(): output_p.unlink(missing_ok=True)
        raise
    except FileNotFoundError:
        headless_logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        headless_logger.debug("CONCATENATE_VIDEOS_FFMPEG: ffmpeg command not found.")
        raise

# --- Video properties extraction with ffprobe fallback ---
def _get_video_props_ffprobe(video_path_str: str) -> tuple[int | None, float | None]:
    """
    Fallback method using ffprobe to get frame count and FPS.
    More reliable for WebM and other containers where OpenCV struggles.
    """
    import subprocess
    try:
        # Get frame count and fps using ffprobe
        # For containers without frame count in metadata, we use nb_read_frames (requires counting)
        # First try fast method: nb_frames from stream metadata
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,r_frame_rate,duration',
            '-of', 'json',
            video_path_str
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: ffprobe failed: {result.stderr}")
            return None, None

        import json
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if not streams:
            return None, None

        stream = streams[0]

        # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
        fps = None
        r_frame_rate = stream.get('r_frame_rate', '')
        if r_frame_rate and '/' in r_frame_rate:
            num, den = r_frame_rate.split('/')
            if float(den) > 0:
                fps = float(num) / float(den)

        # Try to get frame count from metadata
        frame_count = None
        nb_frames = stream.get('nb_frames')
        if nb_frames and nb_frames != 'N/A':
            try:
                frame_count = int(nb_frames)
            except (ValueError, TypeError):
                pass

        # If no frame count in metadata, estimate from duration * fps
        # This is common for WebM files
        if frame_count is None and fps and fps > 0:
            duration = stream.get('duration')
            if duration and duration != 'N/A':
                try:
                    frame_count = int(float(duration) * fps)
                    headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: Estimated frame count from duration: {frame_count}")
                except (ValueError, TypeError):
                    pass

        # Last resort: actually count frames (slow but accurate)
        if frame_count is None:
            headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: Counting frames for {video_path_str} (may be slow)...")
            count_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'csv=p=0',
                video_path_str
            ]
            count_result = subprocess.run(count_cmd, capture_output=True, text=True, timeout=120)
            if count_result.returncode == 0 and count_result.stdout.strip():
                try:
                    frame_count = int(count_result.stdout.strip())
                except (ValueError, TypeError):
                    pass

        headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: {video_path_str} - Frames: {frame_count}, FPS: {fps}")
        return frame_count, fps

    except subprocess.TimeoutExpired:
        headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: Timeout processing {video_path_str}")
        return None, None
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        headless_logger.debug(f"GET_VIDEO_PROPS_FFPROBE: Exception processing {video_path_str}: {e}")
        return None, None

def get_video_frame_count_and_fps(input_video_path: str | Path) -> tuple[int | None, float | None]:
    """
    Gets frame count and FPS of a video.

    Uses OpenCV first (fast), then falls back to ffprobe for containers
    where OpenCV is unreliable (e.g., WebM with VP8/VP9 codec).

    Returns (None, None) on failure.
    """
    video_path_str = str(Path(input_video_path).resolve())
    cap = None
    try:
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            headless_logger.debug(f"GET_VIDEO_FRAME_COUNT_FPS: Could not open video with OpenCV: {video_path_str}")
            # Try ffprobe as fallback
            return _get_video_props_ffprobe(video_path_str)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Validate frame_count and fps as they can sometimes be 0 or negative for problematic files/streams
        valid_frame_count = frame_count if frame_count > 0 else None
        valid_fps = fps if fps > 0 else None

        # If OpenCV failed to get frame count (common for WebM), try ffprobe
        if valid_frame_count is None:
            headless_logger.debug(f"GET_VIDEO_FRAME_COUNT_FPS: OpenCV reported non-positive frame count ({frame_count}) for {video_path_str}. Trying ffprobe...")
            ffprobe_frame_count, ffprobe_fps = _get_video_props_ffprobe(video_path_str)
            if ffprobe_frame_count is not None:
                valid_frame_count = ffprobe_frame_count
            if valid_fps is None and ffprobe_fps is not None:
                valid_fps = ffprobe_fps

        if valid_fps is None:
            headless_logger.debug(f"GET_VIDEO_FRAME_COUNT_FPS: Video {video_path_str} reported non-positive FPS: {fps}. Treating as unknown.")

        headless_logger.debug(f"GET_VIDEO_FRAME_COUNT_FPS: Video {video_path_str} - Frames: {valid_frame_count}, FPS: {valid_fps}")
        return valid_frame_count, valid_fps
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.debug(f"GET_VIDEO_FRAME_COUNT_FPS: Exception processing {video_path_str}: {e}")
        # Try ffprobe as fallback on any exception
        return _get_video_props_ffprobe(video_path_str)
    finally:
        if cap:
            cap.release()

def image_to_frame(image_path_str: str | Path, target_resolution_wh: tuple[int, int] | None = None, task_id_for_logging: str | None = "generic_task", image_download_dir: Path | str | None = None, debug_mode: bool = False) -> np.ndarray | None:
    """
    Load an image, optionally resize, and convert to BGR NumPy array.
    If image_path_str is a URL and image_download_dir is provided, it attempts to download it first.
    """
    resolved_image_path_str = image_path_str # Default to original path

    if isinstance(image_path_str, str): # Only attempt download if it's a string (potentially a URL)
        resolved_image_path_str = download_image_if_url(image_path_str, image_download_dir, task_id_for_logging, debug_mode)

    image_path = Path(resolved_image_path_str)

    if not image_path.exists():
        headless_logger.debug(f"Task {task_id_for_logging}: Image file not found at {image_path} (original input: {image_path_str}).")
        return None
    try:
        img = Image.open(image_path).convert("RGB") # Ensure RGB for consistent processing
        if target_resolution_wh:
            img = img.resize(target_resolution_wh, Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.debug(f"Error loading image {image_path} (original input {image_path_str}): {e}")
        return None

def apply_strength_to_image(
    image_path_input: Path | str, # Changed name to avoid confusion
    strength: float,
    output_path: Path,
    target_resolution_wh: tuple[int, int] | None,
    task_id_for_logging: str | None = "generic_task",
    image_download_dir: Path | str | None = None
) -> Path | None:
    """
    Applies a brightness adjustment (strength) to an image, optionally resizes, and saves it.
    If image_path_input is a URL string and image_download_dir is provided, it attempts to download it first.
    """
    resolved_image_path_str = str(image_path_input)

    if isinstance(image_path_input, str):
         resolved_image_path_str = download_image_if_url(image_path_input, image_download_dir, task_id_for_logging)
    # Check if image_path_input was a Path object representing a URL (less common for this function)
    elif isinstance(image_path_input, Path) and image_path_input.as_posix().startswith(('http://', 'https://')):
         resolved_image_path_str = download_image_if_url(image_path_input.as_posix(), image_download_dir, task_id_for_logging)

    actual_image_path = Path(resolved_image_path_str)

    if not actual_image_path.exists():
        headless_logger.debug(f"Task {task_id_for_logging}: Source image not found at {actual_image_path} (original input: {image_path_input}) for strength application.")
        return None
    try:
        # Open the potentially downloaded or original local image
        img = Image.open(actual_image_path).convert("RGB") # Ensure RGB

        if target_resolution_wh:
            img = img.resize(target_resolution_wh, Image.Resampling.LANCZOS)

        # Apply the strength factor using PIL.ImageEnhance for brightness
        enhancer = ImageEnhance.Brightness(img)
        processed_img = enhancer.enhance(strength) # 'strength' is the factor for brightness

        # Save the adjusted image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_img.save(output_path) # Save PIL image directly

        headless_logger.debug(f"Task {task_id_for_logging}: Applied strength {strength} to {actual_image_path.name}, saved to {output_path.name} with resolution {target_resolution_wh if target_resolution_wh else 'original'}")
        return output_path
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.debug(f"Task {task_id_for_logging}: Error in _apply_strength_to_image for {actual_image_path}: {e}", exc_info=True)
        return None

def _copy_to_folder_with_unique_name(source_path: Path, target_dir: Path, base_name: str, extension: str) -> Path | None:
    """Copies a file to a target directory with a unique name based on timestamp and random string."""
    import shutil

    if not source_path:
        headless_logger.debug(f"COPY: Source path is None for {base_name}{extension}. Skipping copy.")
        return None

    source_path_obj = Path(source_path)
    if not source_path_obj.exists():
        headless_logger.debug(f"COPY: Source file {source_path_obj} does not exist. Skipping copy.")
        return None

    # Sanitize extension for _get_unique_target_path
    actual_extension = source_path_obj.suffix if source_path_obj.suffix else extension
    if not actual_extension.startswith('.'):
        actual_extension = '.' + actual_extension

    # Determine unique target path using the new helper
    from source.utils.download_utils import _get_unique_target_path
    target_file = _get_unique_target_path(target_dir, base_name, actual_extension)

    try:
        # target_dir.mkdir(parents=True, exist_ok=True) # _get_unique_target_path handles this
        shutil.copy2(str(source_path_obj), str(target_file))
        headless_logger.debug(f"COPY: Copied {source_path_obj.name} to {target_file}")
        return target_file # Return the path of the copied file
    except OSError as e_copy:
        headless_logger.debug(f"COPY: Failed to copy {source_path_obj} to {target_file}: {e_copy}")
        return None

def get_image_dimensions_pil(image_path: str | Path) -> tuple[int, int]:
    """Returns the dimensions of an image file as (width, height)."""
    with Image.open(image_path) as img:
        return img.size

# Added to adjust the brightness of an image/frame.
def adjust_frame_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """Adjusts the brightness of a given frame.
    The 'factor' is interpreted as a delta from the CLI argument:
    - Positive factor (e.g., 0.1) makes it darker (target_alpha = 1.0 - 0.1 = 0.9).
    - Negative factor (e.g., -0.1) makes it brighter (target_alpha = 1.0 - (-0.1) = 1.1).
    - Zero factor means no change (target_alpha = 1.0).
    """
    # Convert the CLI-style factor to an alpha for cv2.convertScaleAbs
    # CLI factor: positive = darker, negative = brighter
    # cv2 alpha: >1 = brighter, <1 = darker
    cv2_alpha = 1.0 - factor
    return cv2.convertScaleAbs(frame, alpha=cv2_alpha, beta=0)

def get_sequential_target_path(target_dir: Path, name_stem: str, suffix: str) -> Path:
    """Generates a unique target Path in the given directory by appending a number if needed."""
    if not suffix.startswith('.'):
        suffix = f".{suffix}"

    final_path = target_dir / f"{name_stem}{suffix}"
    counter = 1
    while final_path.exists():
        final_path = target_dir / f"{name_stem}_{counter}{suffix}"
        counter += 1
    return final_path

def load_pil_images(
    paths_list_or_str: list[str] | str,
    wgp_convert_func: callable,
    image_download_dir: Path | str | None,
    task_id_for_log: str) -> list | None:
    """
    Loads one or more images from paths or URLs, downloads them if necessary,
    and applies a conversion function.
    """
    if not paths_list_or_str:
        return None

    paths_list = paths_list_or_str if isinstance(paths_list_or_str, list) else [paths_list_or_str]
    images = []

    for p_str in paths_list:
        local_p_str = download_image_if_url(p_str, image_download_dir, task_id_for_log)
        if not local_p_str:
            headless_logger.debug_anomaly("Task {task_id_for_log}", f"Skipping image as download_image_if_url returned nothing for: {p_str}")
            continue

        p = Path(local_p_str.strip())
        if not p.is_file():
            headless_logger.debug_anomaly("Task {task_id_for_log}", f"load_pil_images: Image file not found after potential download: {p} (original: {p_str})")
            continue
        try:
            img = Image.open(p)
            images.append(wgp_convert_func(img))
        except (OSError, ValueError, RuntimeError) as e:
            headless_logger.warning(f"Failed to load image {p}: {e}")

    return images if images else None
