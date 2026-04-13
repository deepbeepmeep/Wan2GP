"""FFmpeg-based video operations: creation, FPS conversion, frame extraction to video."""
from dataclasses import dataclass
import os
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from source.core.log import generation_logger
from source.utils.subprocess_utils import run_subprocess

# On Windows, child processes must NOT share our console Ctrl-C group.
# CREATE_NEW_PROCESS_GROUP prevents phantom CTRL_C_EVENT propagation.
_SUBPROCESS_FLAGS = (
    {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    if os.name == "nt" else {}
)
from source.media.video.video_info import (
    get_video_frame_count_and_fps,
    get_video_frame_count_ffprobe,
)

__all__ = [
    "ensure_video_fps",
    "extract_frame_range_to_video",
    "create_video_from_frames_list",
    "apply_saturation_to_video_ffmpeg",
    "video_has_audio",
    "mux_audio_from_segments",
    "mux_audio_from_segments_result",
]


@dataclass(frozen=True)
class AudioMuxResult:
    status: str
    output_path: Path | None = None


def ensure_video_fps(
    input_video_path: str | Path,
    target_fps: float,
    output_dir: str | Path | None = None,
    fps_tolerance: float = 0.5,
) -> Path:
    """
    Ensure a video is at the target FPS, resampling if necessary.

    This is important when frame indices are calculated for a specific FPS
    (e.g., from frontend) but the actual video file may be at a different FPS.

    Args:
        input_video_path: Path to source video
        target_fps: Desired FPS (e.g., 16)
        output_dir: Directory for resampled video (defaults to same dir as source)
        fps_tolerance: Maximum FPS difference before resampling (default 0.5)

    Returns:
        Path to video at target FPS (original if already correct, resampled if not)

    Raises:
        OSError: If the source video does not exist or the resampled output is missing/empty
        RuntimeError: If FFmpeg fails or times out
        ValueError: If the video FPS cannot be determined

    Example:
        # Ensure video is at 16fps before frame-based extraction
        video_16fps = ensure_video_fps(downloaded_video, 16, work_dir)
        extract_frame_range_to_video(video_16fps, output, 0, 252, 16)
    """
    input_video_path = Path(input_video_path)

    # Validate target_fps is not None
    if target_fps is None:
        generation_logger.warning(f"[ENSURE_FPS] target_fps is None, defaulting to 16")
        target_fps = 16

    if not input_video_path.exists():
        raise OSError(f"Video does not exist: {input_video_path}")

    # Get actual fps
    actual_frames, actual_fps = get_video_frame_count_and_fps(str(input_video_path))
    if not actual_fps:
        raise ValueError(f"Could not determine video FPS: {input_video_path}")

    # Check if resampling is needed
    if abs(actual_fps - target_fps) <= fps_tolerance:
        generation_logger.debug_anomaly("ENSURE_FPS", f"Video already at target FPS: {actual_fps} fps (target: {target_fps}, tolerance: \u00b1{fps_tolerance})")
        return input_video_path

    # Need to resample
    generation_logger.debug_anomaly("ENSURE_FPS", f"FPS mismatch: actual={actual_fps}, target={target_fps}")
    generation_logger.debug_anomaly("ENSURE_FPS", f"Resampling {input_video_path.name} to {target_fps} fps...")

    # Determine output path
    if output_dir is None:
        output_dir = input_video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_path = output_dir / f"{input_video_path.stem}_resampled_{int(target_fps)}fps.mp4"

    # Use fps filter (not -r output option) for accurate frame-based resampling
    # The fps filter selects frames based on timestamps, ensuring frame N = time N/fps
    # This is critical for frame-accurate extraction where frame indices must match timestamps
    resample_cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video_path),
        '-vf', f'fps={target_fps}',
        '-an',  # No audio for now
        str(resampled_path)
    ]

    try:
        result = run_subprocess(resample_cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FFmpeg resample timed out after 600s for {input_video_path}") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"FFmpeg resample failed for {input_video_path}: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg resample failed (rc={result.returncode}): {result.stderr[:500]}")

    if not resampled_path.exists() or resampled_path.stat().st_size == 0:
        raise OSError(f"Resampled video missing or empty: {resampled_path}")

    # Verify result
    new_frames, new_fps = get_video_frame_count_and_fps(str(resampled_path))
    generation_logger.debug_anomaly("ENSURE_FPS", f"Resampled: {new_frames} frames @ {new_fps} fps")

    return resampled_path


def extract_frame_range_to_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    start_frame: int,
    end_frame: int | None,
    fps: float,
) -> Path:
    """
    Extract a range of frames from a video to a new video file using FFmpeg.

    Uses FFmpeg's select filter with frame-accurate selection (not time-based).
    This is the canonical function for frame extraction - use this instead of
    inline FFmpeg commands.

    Args:
        input_video_path: Path to source video file
        output_video_path: Path for output video file
        start_frame: First frame to include (0-indexed, inclusive)
        end_frame: Last frame to include (0-indexed, inclusive), or None for all remaining frames
        fps: Output framerate

    Returns:
        Path to extracted video

    Raises:
        OSError: If source video does not exist or output is missing/empty
        ValueError: If frame range is invalid or frame count cannot be determined
        RuntimeError: If FFmpeg fails, times out, or output frame count is wrong

    Examples:
        # Extract frames 0-252 (253 frames)
        extract_frame_range_to_video(src, out, 0, 252, 16)

        # Extract frames from 13 onwards (skip first 13)
        extract_frame_range_to_video(src, out, 13, None, 16)
    """
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify source exists
    if not input_video_path.exists():
        raise OSError(f"Source video does not exist: {input_video_path}")

    # Get source properties - use ffprobe for accurate frame count
    source_frames = get_video_frame_count_ffprobe(str(input_video_path))
    _, source_fps = get_video_frame_count_and_fps(str(input_video_path))

    if not source_frames:
        # Fallback to OpenCV if ffprobe fails
        source_frames, source_fps = get_video_frame_count_and_fps(str(input_video_path))
        generation_logger.warning(f"[EXTRACT_RANGE] Using OpenCV frame count (ffprobe failed): {source_frames}")

    if not source_frames:
        raise ValueError(f"Could not determine source video frame count: {input_video_path}")

    # Validate range
    if start_frame < 0:
        raise ValueError(f"start_frame cannot be negative: {start_frame}")

    if end_frame is not None and end_frame >= source_frames:
        raise ValueError(f"end_frame {end_frame} >= source frames {source_frames}")

    # Build filter based on whether we have an end_frame
    if end_frame is not None:
        # Extract specific range: frames start_frame to end_frame (inclusive)
        filter_str = f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/FR/TB"
        expected_frames = end_frame - start_frame + 1
        range_desc = f"frames {start_frame}-{end_frame} ({expected_frames} frames)"
    else:
        # Extract from start_frame onwards (skip first start_frame frames)
        filter_str = f"select=gte(n\\,{start_frame}),setpts=N/FR/TB"
        expected_frames = source_frames - start_frame
        range_desc = f"frames {start_frame}-{source_frames-1} ({expected_frames} frames)"

    generation_logger.debug_anomaly("EXTRACT_RANGE", f"Source: {input_video_path.name} ({source_frames} frames @ {source_fps} fps)")
    generation_logger.debug_anomaly("EXTRACT_RANGE", f"Extracting: {range_desc}")

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video_path),
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',  # Better quality at same bitrate
        '-crf', '10',  # Near-lossless quality for intermediate files
        '-r', str(fps),
        '-an',  # No audio
        str(output_video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FFmpeg frame extraction timed out after 300s for {input_video_path}") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"FFmpeg frame extraction failed for {input_video_path}: {e}") from e

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed (rc={result.returncode}): {result.stderr[:500]}")

    if not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise OSError(f"Output video missing or empty: {output_video_path}")

    # Verify frame count (allow small tolerance for OpenCV/FFmpeg discrepancies)
    actual_frames, _ = get_video_frame_count_and_fps(str(output_video_path))
    if actual_frames is None:
        raise RuntimeError(f"Could not verify output frame count for {output_video_path}")

    # Allow up to 3 frames difference - OpenCV and FFmpeg often disagree on frame counts
    # especially for end-of-video segments or variable framerate videos
    frame_diff = abs(actual_frames - expected_frames)
    if frame_diff > 3:
        raise RuntimeError(f"Frame count mismatch too large: expected {expected_frames}, got {actual_frames} (diff: {frame_diff})")
    elif frame_diff > 0:
        generation_logger.warning(f"[EXTRACT_RANGE] Minor frame count difference: expected {expected_frames}, got {actual_frames} (diff: {frame_diff}, within tolerance)")

    generation_logger.debug_anomaly("EXTRACT_RANGE", f"Extracted {actual_frames} frames to {output_video_path.name}")
    return output_video_path


def create_video_from_frames_list(
    frames_list: list[np.ndarray],
    output_path: str | Path,
    fps: int,
    resolution: tuple[int, int],
    *,
    standardize_colorspace: bool = False
) -> Path:
    """Creates a video from a list of NumPy BGR frames using FFmpeg subprocess.

    Uses streaming to pipe frames to FFmpeg incrementally, avoiding loading
    all frame data into memory at once. This is critical for large videos
    (e.g., 2000+ frames at 1080p would otherwise require 30+ GB RAM).

    Args:
        frames_list: List of BGR numpy arrays
        output_path: Output video path
        fps: Frames per second
        resolution: (width, height) tuple
        standardize_colorspace: If True, adds BT.709 colorspace standardization

    Returns:
        Path object of the successfully written file

    Raises:
        ValueError: If no valid frames are provided
        RuntimeError: If FFmpeg fails, times out, or output is missing/empty
        OSError: If FFmpeg is not installed or file system errors occur
    """

    output_path_obj = Path(output_path)
    output_path_mp4 = output_path_obj.with_suffix('.mp4')
    output_path_mp4.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{resolution[0]}x{resolution[1]}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",  # Better quality at same bitrate
        "-crf", "10",  # Near-lossless quality for intermediate files
    ]

    # Add colorspace standardization if requested
    if standardize_colorspace:
        ffmpeg_cmd.extend([
            "-vf", "format=yuv420p,colorspace=bt709:iall=bt709:fast=1",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
        ])

    ffmpeg_cmd.append(str(output_path_mp4.resolve()))

    # Count valid frames first (without storing them)
    valid_count = 0
    skipped_none = 0
    skipped_invalid = 0
    for frame_np in frames_list:
        if frame_np is None or not isinstance(frame_np, np.ndarray):
            skipped_none += 1
        elif len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
            skipped_invalid += 1
        else:
            valid_count += 1

    if skipped_none > 0:
        generation_logger.warning(f"[CREATE_VIDEO] Will skip {skipped_none} None/invalid frames")
    if skipped_invalid > 0:
        generation_logger.warning(f"[CREATE_VIDEO] Will skip {skipped_invalid} non-RGB frames")

    if valid_count == 0:
        raise ValueError(f"No valid frames to process! Input had {len(frames_list)} frames, all invalid")

    try:
        # Use Popen to stream frames incrementally instead of loading all into memory
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_SUBPROCESS_FLAGS,
        )
    except FileNotFoundError as e:
        raise OSError("FFmpeg not found - is it installed?") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"Failed to start FFmpeg process: {e}") from e

    frames_written = 0
    resize_warnings = 0

    for frame_idx, frame_np in enumerate(frames_list):
        # Skip invalid frames
        if frame_np is None or not isinstance(frame_np, np.ndarray):
            continue
        if len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
            continue

        # Ensure uint8
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)

        # Resize if needed
        if frame_np.shape[0] != resolution[1] or frame_np.shape[1] != resolution[0]:
            try:
                frame_np = cv2.resize(frame_np, resolution, interpolation=cv2.INTER_AREA)
            except (OSError, ValueError, RuntimeError) as e:
                if resize_warnings < 5:
                    generation_logger.warning(f"[CREATE_VIDEO] Failed to resize frame {frame_idx}: {e}")
                resize_warnings += 1
                continue

        # Write frame bytes directly to FFmpeg stdin
        try:
            proc.stdin.write(frame_np.tobytes())
            frames_written += 1
        except BrokenPipeError:
            generation_logger.error(f"[CREATE_VIDEO] FFmpeg pipe broken after {frames_written} frames")
            break

    if resize_warnings > 5:
        generation_logger.warning(f"[CREATE_VIDEO] {resize_warnings} total resize failures (only first 5 logged)")

    # Close stdin to signal end of input
    proc.stdin.close()

    # Read stderr in background thread to avoid blocking
    stderr_output = []
    def read_stderr():
        try:
            stderr_output.append(proc.stderr.read())
        except OSError as e:
            generation_logger.debug_anomaly("CREATE_VIDEO", f"Failed to read FFmpeg stderr: {e}")

    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()

    try:
        proc.wait(timeout=300)  # 5 minute timeout for encoding
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"FFmpeg timed out after 300 seconds creating video {output_path_mp4}")
    finally:
        stderr_thread.join(timeout=5)

    stderr = stderr_output[0] if stderr_output else b""

    if proc.returncode == 0:
        if output_path_mp4.exists() and output_path_mp4.stat().st_size > 0:
            return output_path_mp4
        raise RuntimeError(f"FFmpeg succeeded but output file missing or empty: {output_path_mp4}")
    else:
        stderr_str = stderr.decode('utf-8', errors='replace') if stderr else "no stderr"
        if output_path_mp4.exists():
            try:
                output_path_mp4.unlink()
            except OSError as e:
                generation_logger.debug_anomaly("CREATE_VIDEO", f"Failed to clean up partial output file {output_path_mp4}: {e}")
        raise RuntimeError(f"FFmpeg failed (rc={proc.returncode}): {stderr_str[:500]}")


def apply_saturation_to_video_ffmpeg(
    input_video_path: str | Path,
    output_video_path: str | Path,
    saturation_level: float,
    preset: str = "medium"
) -> bool:
    """Applies a saturation adjustment to the full video using FFmpeg's eq filter.
    Returns: True if FFmpeg succeeds and the output file exists & is non-empty, else False.
    """
    inp = Path(input_video_path)
    outp = Path(output_video_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp.resolve()),
        "-vf", f"eq=saturation={saturation_level}",
        "-c:v", "libx264",
        "-crf", "10",  # Near-lossless quality for intermediate files
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-an",
        str(outp.resolve())
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", timeout=600)
        if outp.exists() and outp.stat().st_size > 0:
            return True
        return False
    except subprocess.CalledProcessError:
        return False


def video_has_audio(video_path: str | Path) -> bool:
    """Check if a video file contains an audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout
    except (subprocess.SubprocessError, OSError):
        return False


def mux_audio_from_segments(
    silent_video: str | Path,
    segment_videos: list[str | Path],
    output_path: str | Path,
    overlap_frames: list[int] | None = None,
    fps: float = 24.0,
) -> Path | None:
    """Concatenate audio from segment videos and mux onto a silent stitched video.

    Extracts audio from each segment, trims overlap regions to match the video
    stitch, concatenates, and muxes the result onto *silent_video*.

    Args:
        silent_video: Path to the stitched video (no audio).
        segment_videos: Ordered list of segment video paths (with audio).
        output_path: Destination path for the video+audio result.
        overlap_frames: Frame overlaps between adjacent segments (len = n-1).
        fps: Frames per second (used to convert overlap frames to seconds).

    Returns:
        Path to the output video with audio, or None on failure.
    """
    silent_video = Path(silent_video)
    output_path = Path(output_path)

    # Quick check: do any segments have audio?
    segments_with_audio = [p for p in segment_videos if video_has_audio(p)]
    if not segments_with_audio:
        generation_logger.debug_anomaly("AUDIO_MUX", "No segment videos contain audio — skipping audio mux")
        return None

    generation_logger.debug(
        f"[AUDIO_MUX] {len(segments_with_audio)}/{len(segment_videos)} segments have audio, "
        f"muxing onto stitched video"
    )

    try:
        import tempfile

        work_dir = Path(tempfile.mkdtemp(prefix="audio_mux_"))
        overlaps = overlap_frames or [0] * (len(segment_videos) - 1)

        # --- Step 1: Extract audio from each segment, trimming overlaps ---
        audio_parts: list[Path] = []
        for i, seg_path in enumerate(segment_videos):
            seg_path = Path(seg_path)
            if not video_has_audio(seg_path):
                # Generate silence matching this segment's duration
                try:
                    dur_result = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                         "-of", "csv=p=0", str(seg_path)],
                        capture_output=True, text=True, timeout=10,
                    )
                    duration = float(dur_result.stdout.strip())
                except (subprocess.SubprocessError, OSError, ValueError):
                    duration = 1.0

                # Trim overlap from start (except first segment)
                if i > 0 and i - 1 < len(overlaps):
                    trim_start = overlaps[i - 1] / fps
                    duration = max(0.01, duration - trim_start)

                silence_path = work_dir / f"silence_{i}.aac"
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     f"anullsrc=channel_layout=stereo:sample_rate=44100",
                     "-t", f"{duration:.4f}", "-c:a", "aac", str(silence_path)],
                    capture_output=True, timeout=30,
                )
                if silence_path.exists():
                    audio_parts.append(silence_path)
                continue

            part_path = work_dir / f"audio_{i}.aac"

            # Build ffmpeg command with optional trim
            cmd = ["ffmpeg", "-y", "-i", str(seg_path), "-vn", "-c:a", "aac"]

            if i > 0 and i - 1 < len(overlaps) and overlaps[i - 1] > 0:
                # Trim overlap seconds from the beginning of this segment's audio
                trim_seconds = overlaps[i - 1] / fps
                cmd.extend(["-ss", f"{trim_seconds:.4f}"])

            cmd.append(str(part_path))
            subprocess.run(cmd, capture_output=True, timeout=120)

            if part_path.exists() and part_path.stat().st_size > 0:
                audio_parts.append(part_path)

        if not audio_parts:
            generation_logger.debug_anomaly("AUDIO_MUX", "No audio parts extracted — skipping")
            return None

        # --- Step 2: Concatenate audio parts ---
        concat_list = work_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in audio_parts),
            encoding="utf-8",
        )

        merged_audio = work_dir / "merged_audio.aac"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c:a", "aac", str(merged_audio)],
            capture_output=True, timeout=120,
        )

        if not merged_audio.exists() or merged_audio.stat().st_size == 0:
            generation_logger.warning("[AUDIO_MUX] Audio concatenation failed")
            return None

        # --- Step 3: Mux merged audio onto the silent video ---
        # Use -shortest so the output matches the video duration
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(silent_video),
             "-i", str(merged_audio),
             "-c:v", "copy", "-c:a", "aac", "-shortest",
             str(output_path)],
            capture_output=True, timeout=120,
        )

        if output_path.exists() and output_path.stat().st_size > 0:
            generation_logger.debug_anomaly("AUDIO_MUX", f"Successfully muxed audio → {output_path}")
            # Clean up temp dir
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
            return output_path

        generation_logger.warning("[AUDIO_MUX] Final mux produced empty output")
        return None

    except (subprocess.SubprocessError, OSError, ValueError) as e:
        generation_logger.warning(f"[AUDIO_MUX] Audio mux failed: {e}")
        return None


def mux_audio_from_segments_result(
    *,
    silent_video: Path,
    segment_videos: list[Path],
    output_path: Path,
) -> AudioMuxResult:
    if not any(video_has_audio(path) for path in segment_videos):
        return AudioMuxResult(status="no_audio", output_path=None)

    muxed = mux_audio_from_segments(
        silent_video=silent_video,
        segment_videos=segment_videos,
        output_path=output_path,
    )
    if muxed is None:
        return AudioMuxResult(status="failed", output_path=None)
    return AudioMuxResult(status="ok", output_path=Path(muxed))
