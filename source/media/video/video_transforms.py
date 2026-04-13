"""Video transforms: brightness, reverse, aspect ratio, audio, overlay."""
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from source.core.log import headless_logger, generation_logger
from source.media.video.frame_extraction import extract_frames_from_video
from source.media.video.video_info import get_video_frame_count_and_fps
from source.media.video.ffmpeg_ops import create_video_from_frames_list
from source.utils.subprocess_utils import run_subprocess

try:
    import moviepy.editor as mpe
    _MOVIEPY_AVAILABLE = True
except ImportError:
    _MOVIEPY_AVAILABLE = False

__all__ = [
    "adjust_frame_brightness",
    "apply_brightness_to_video_frames",
    "reverse_video",
    "standardize_video_aspect_ratio",
    "add_audio_to_video",
    "overlay_start_end_images_above_video",
]

def adjust_frame_brightness(frame: np.ndarray, brightness_adjust: float) -> np.ndarray:
    if brightness_adjust == 0:
        return frame
    factor = 1 + brightness_adjust
    adjusted = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted

def apply_brightness_to_video_frames(input_video_path: str, output_video_path: Path, brightness_adjust: float, task_id_for_logging: str) -> Path | None:
    """
    Applies brightness adjustment to a video by processing its frames.
    A brightness_adjust of 0 means no change. Negative values darken, positive values brighten.
    """
    try:
        headless_logger.essential(f"Task {task_id_for_logging}: Applying brightness adjustment {brightness_adjust} to {input_video_path}", task_id=task_id_for_logging)

        total_frames, fps = get_video_frame_count_and_fps(input_video_path)
        if total_frames is None or fps is None or total_frames == 0:
            headless_logger.error(f"Task {task_id_for_logging}: Could not get frame count or fps for {input_video_path}, or video has 0 frames.", task_id=task_id_for_logging)
            return None

        frames = extract_frames_from_video(input_video_path)
        if frames is None:
            headless_logger.error(f"Task {task_id_for_logging}: Could not extract frames from {input_video_path}", task_id=task_id_for_logging)
            return None

        adjusted_frames = []
        first_frame = None
        for frame in frames:
            if first_frame is None:
                first_frame = frame
            adjusted_frame = adjust_frame_brightness(frame, brightness_adjust)
            adjusted_frames.append(adjusted_frame)

        if not adjusted_frames or first_frame is None:
            headless_logger.error(f"Task {task_id_for_logging}: No frames to write for brightness-adjusted video.", task_id=task_id_for_logging)
            return None

        h, w, _ = first_frame.shape
        resolution = (w, h)

        created_video_path = create_video_from_frames_list(adjusted_frames, output_video_path, fps, resolution)
        headless_logger.essential(f"Task {task_id_for_logging}: Successfully created brightness-adjusted video at {created_video_path}", task_id=task_id_for_logging)
        return created_video_path
    except (OSError, ValueError, RuntimeError) as e:
        headless_logger.error(f"Task {task_id_for_logging}: Exception in apply_brightness_to_video_frames: {e}", task_id=task_id_for_logging, exc_info=True)
        return None

def reverse_video(
    input_video_path: str | Path,
    output_video_path: str | Path) -> Path | None:
    """
    Reverse a video using FFmpeg (play it backwards) with visually lossless quality.

    Uses CRF 17 encoding which is visually indistinguishable from the original.

    Args:
        input_video_path: Path to input video
        output_video_path: Path for reversed output video

    Returns:
        Path to reversed video if successful, None otherwise
    """
    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    if not input_path.exists():
        generation_logger.debug_anomaly("REVERSE_VIDEO", f"Input video not found: {input_path}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get input video FPS to preserve it exactly
    _, input_fps = get_video_frame_count_and_fps(str(input_path))
    if not input_fps:
        input_fps = 16  # Default fallback

    # Use ffmpeg to reverse the video with near-lossless quality
    # -crf 10: Near-lossless for intermediate files (minimizes generation loss)
    # -preset slow: Better compression efficiency, maintains quality
    # -r: Preserve exact framerate from input
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', 'reverse',
        '-an',  # No audio (reversed audio sounds bad anyway)
        '-c:v', 'libx264',
        '-crf', '10',  # Near-lossless quality for intermediate files
        '-preset', 'slow',  # Better quality at same bitrate
        '-pix_fmt', 'yuv420p',
        '-r', str(input_fps),  # Preserve exact framerate
        str(output_path)
    ]

    try:
        generation_logger.debug_anomaly("REVERSE_VIDEO", f"Reversing video (visually lossless): {input_path.name} -> {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # Longer timeout for slow preset

        if result.returncode != 0:
            generation_logger.debug_anomaly("REVERSE_VIDEO", f"FFmpeg failed: {result.stderr[:500]}")
            return None

        if not output_path.exists() or output_path.stat().st_size == 0:
            generation_logger.debug_anomaly("REVERSE_VIDEO", f"Output missing or empty: {output_path}")
            return None

        # Verify the reversed video
        reversed_frames, reversed_fps = get_video_frame_count_and_fps(str(output_path))
        original_frames, _ = get_video_frame_count_and_fps(str(input_path))

        generation_logger.debug_anomaly("REVERSE_VIDEO", f"\u2705 Successfully reversed video: {original_frames} -> {reversed_frames} frames @ {reversed_fps} fps (CRF 17)")
        return output_path

    except subprocess.TimeoutExpired:
        generation_logger.debug_anomaly("REVERSE_VIDEO", f"FFmpeg timeout")
        return None
    except (subprocess.SubprocessError, OSError) as e:
        generation_logger.error(f"REVERSE_VIDEO exception: {e}", exc_info=True)
        return None

def standardize_video_aspect_ratio(
    input_video_path: str | Path,
    output_video_path: str | Path,
    target_aspect_ratio: str,
    task_id_for_logging: str = "") -> Path | None:
    """
    Standardize video to target aspect ratio by center-cropping.

    Args:
        input_video_path: Path to input video
        output_video_path: Path where standardized video will be saved
        target_aspect_ratio: Target aspect ratio as string (e.g., "16:9", "9:16", "1:1")
        task_id_for_logging: Task ID for logging

    Returns:
        Path to standardized video if successful, None otherwise
    """
    import subprocess
    from pathlib import Path

    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    if not input_path.exists():
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Input video not found: {input_path}")
        return None

    # Parse aspect ratio
    try:
        ar_parts = target_aspect_ratio.split(":")
        if len(ar_parts) != 2:
            raise ValueError(f"Invalid aspect ratio format: {target_aspect_ratio}")
        target_w, target_h = float(ar_parts[0]), float(ar_parts[1])
        target_aspect = target_w / target_h
    except (ValueError, TypeError) as e:
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Failed to parse aspect ratio '{target_aspect_ratio}': {e}")
        return None

    # Get input video dimensions using ffprobe
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(input_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: ffprobe failed: {result.stderr}")
            return None

        width_str, height_str = result.stdout.strip().split(',')
        src_w, src_h = int(width_str), int(height_str)
        src_aspect = src_w / src_h

        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Input video: {src_w}x{src_h} (aspect: {src_aspect:.3f})")
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Target aspect: {target_aspect:.3f}")

    except (subprocess.SubprocessError, OSError, ValueError) as e:
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Failed to get video dimensions: {e}")
        return None

    # Check if aspect ratio is already correct (within 1% tolerance)
    if abs(src_aspect - target_aspect) < 0.01:
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Video already has target aspect ratio, copying...")
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
        except OSError as e:
            generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Failed to copy video: {e}")
            return None

    # Calculate crop dimensions
    if src_aspect > target_aspect:
        # Source is wider - crop width
        new_w = int(src_h * target_aspect)
        new_h = src_h
        crop_x = (src_w - new_w) // 2
        crop_y = 0
    else:
        # Source is taller - crop height
        new_w = src_w
        new_h = int(src_w / target_aspect)
        crop_x = 0
        crop_y = (src_h - new_h) // 2

    generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Center-cropping to {new_w}x{new_h}")

    # Apply crop using ffmpeg
    try:
        crop_cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-vf', f'crop={new_w}:{new_h}:{crop_x}:{crop_y}',
            '-c:a', 'copy',  # Copy audio stream if present
            str(output_path)
        ]

        result = subprocess.run(crop_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: ffmpeg crop failed: {result.stderr}")
            return None

        if not output_path.exists() or output_path.stat().st_size == 0:
            generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Output video not created or empty")
            return None

        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Successfully standardized video to {target_aspect_ratio}")
        return output_path

    except (subprocess.SubprocessError, OSError) as e:
        generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: Failed to crop video: {e}")
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as e_cleanup:
                generation_logger.debug_anomaly("STANDARDIZE_ASPECT", f"Task {task_id_for_logging}: DEBUG: Failed to clean up output {output_path}: {e_cleanup}")
        return None

def add_audio_to_video(
    input_video_path: str | Path,
    audio_url: str,
    output_video_path: str | Path,
    temp_dir: str | Path) -> Path | None:
    """
    Add audio to a video file, trimming audio to match video duration.

    If the audio is longer than the video, it will be trimmed to match.
    If the audio is shorter than the video, the video will have audio only
    for the duration of the audio (remainder will be silent).

    Args:
        input_video_path: Path to input video (no audio or audio will be replaced)
        audio_url: URL or local path to audio file (mp3, wav, etc.)
        output_video_path: Path for output video with audio
        temp_dir: Directory for temporary files (audio download)

    Returns:
        Path to output video with audio, or None if failed
    """
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    temp_dir = Path(temp_dir)

    if not input_video_path.exists():
        generation_logger.debug_anomaly("ADD_AUDIO", f"Input video does not exist: {input_video_path}")
        return None

    if not audio_url:
        generation_logger.debug_anomaly("ADD_AUDIO", f"No audio URL provided")
        return None

    generation_logger.debug_anomaly("ADD_AUDIO", f"Adding audio to video")
    generation_logger.debug_anomaly("ADD_AUDIO", f"  Video: {input_video_path}")
    generation_logger.debug_anomaly("ADD_AUDIO", f"  Audio: {audio_url[:80]}...")

    try:
        # Download audio if it's a URL
        if audio_url.startswith(('http://', 'https://')):
            import requests
            import uuid

            # Determine file extension from URL or default to mp3
            audio_ext = '.mp3'
            url_lower = audio_url.lower()
            for ext in ['.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac']:
                if ext in url_lower:
                    audio_ext = ext
                    break

            audio_filename = f"temp_audio_{uuid.uuid4().hex[:8]}{audio_ext}"
            local_audio_path = temp_dir / audio_filename

            generation_logger.debug_anomaly("ADD_AUDIO", f"Downloading audio to {local_audio_path}...")

            response = requests.get(audio_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(local_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            generation_logger.debug_anomaly("ADD_AUDIO", f"Audio downloaded: {local_audio_path.stat().st_size} bytes")
        else:
            local_audio_path = Path(audio_url)
            if not local_audio_path.exists():
                generation_logger.debug_anomaly("ADD_AUDIO", f"Local audio file does not exist: {audio_url}")
                return None

        # Get video duration for logging
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(input_video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            video_duration = float(result.stdout.strip()) if result.returncode == 0 else None
            if video_duration:
                generation_logger.debug_anomaly("ADD_AUDIO", f"Video duration: {video_duration:.2f}s")
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            generation_logger.debug_anomaly("ADD_AUDIO", f"DEBUG: Failed to probe video duration: {e}")
            video_duration = None

        # Mux video with audio using FFmpeg
        # -shortest: Stop when shortest stream ends (trims audio to video length)
        # -c:v copy: Don't re-encode video (fast)
        # -c:a aac: Encode audio to AAC for compatibility
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', str(input_video_path),
            '-i', str(local_audio_path),
            '-c:v', 'copy',           # Copy video stream (no re-encoding)
            '-c:a', 'aac',            # Encode audio to AAC
            '-b:a', '192k',           # Audio bitrate
            '-shortest',              # Trim to shortest stream (video)
            '-map', '0:v:0',          # Take video from first input
            '-map', '1:a:0',          # Take audio from second input
            str(output_video_path)
        ]

        generation_logger.debug_anomaly("ADD_AUDIO", f"Running FFmpeg to mux audio...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            generation_logger.debug_anomaly("ADD_AUDIO", f"FFmpeg failed: {result.stderr[:500] if result.stderr else 'No error message'}")
            return None

        if not output_video_path.exists() or output_video_path.stat().st_size == 0:
            generation_logger.debug_anomaly("ADD_AUDIO", f"Output file not created or empty")
            return None

        # Clean up downloaded audio if we downloaded it
        if audio_url.startswith(('http://', 'https://')):
            try:
                local_audio_path.unlink()
            except OSError as e:
                generation_logger.debug_anomaly("ADD_AUDIO", f"DEBUG: Failed to clean up downloaded audio file {local_audio_path}: {e}")

        generation_logger.debug_anomaly("ADD_AUDIO", f"Successfully added audio to video: {output_video_path}")
        generation_logger.debug_anomaly("ADD_AUDIO", f"  Output size: {output_video_path.stat().st_size} bytes")

        return output_video_path

    except (subprocess.SubprocessError, OSError) as e:
        generation_logger.error(f"ADD_AUDIO error adding audio: {e}", exc_info=True)
        if output_video_path.exists():
            try:
                output_video_path.unlink()
            except OSError as e_cleanup:
                generation_logger.debug_anomaly("ADD_AUDIO", f"DEBUG: Failed to clean up output file {output_video_path}: {e_cleanup}")
        return None

def overlay_start_end_images_above_video(
    start_image_path: str | Path,
    end_image_path: str | Path,
    input_video_path: str | Path,
    output_video_path: str | Path) -> bool:
    """Creates a composite video that shows *start_image* (left) and *end_image* (right)
    on a row above the *input_video*.

    Layout:
        START | END  (static images, full video duration)
        ----------------------
               VIDEO           (original video frames)

    The resulting video keeps the original width of *input_video*.  Each image is
    scaled to exactly half that width and the same height as the video to ensure
    perfect alignment.  The final output therefore has a height of
    ``video_height * 2`` and a width equal to ``video_width``.

    Args:
        start_image_path: Path to the starting image.
        end_image_path:   Path to the ending image.
        input_video_path: Source video that was generated.
        output_video_path: Desired path for the composite video.

    Returns:
        True if the composite video was created successfully, else False.
    """
    try:
        start_image_path = Path(start_image_path)
        end_image_path = Path(end_image_path)
        input_video_path = Path(input_video_path)
        output_video_path = Path(output_video_path)

        if not (start_image_path.exists() and end_image_path.exists() and input_video_path.exists()):
            generation_logger.debug(
                f"overlay_start_end_images_above_video: One or more input paths are missing.\n"
                f"  start_image_path = {start_image_path}\n"
                f"  end_image_path   = {end_image_path}\n"
                f"  input_video_path = {input_video_path}"
            )
            return False

        # ---------------------------------------------------------
        #   Preferred implementation: MoviePy (simpler, robust)
        # ---------------------------------------------------------
        if _MOVIEPY_AVAILABLE:
            try:
                video_clip = mpe.VideoFileClip(str(input_video_path))

                half_width_px = int(video_clip.w / 2)

                img1_clip = mpe.ImageClip(str(start_image_path)).resize(width=half_width_px).set_duration(video_clip.duration)
                img2_clip = mpe.ImageClip(str(end_image_path)).resize(width=half_width_px).set_duration(video_clip.duration)

                # top row (images side-by-side)
                top_row = mpe.clips_array([[img1_clip, img2_clip]])

                # Build composite video
                final_h = top_row.h + video_clip.h
                composite = mpe.CompositeVideoClip([
                    top_row.set_position((0, 0)),
                    video_clip.set_position((0, top_row.h))
                ], size=(video_clip.w, final_h))

                # Write video
                composite.write_videofile(
                    str(output_video_path.with_suffix('.mp4')),
                    codec="libx264",
                    audio=False,
                    fps=video_clip.fps or fps,
                    preset="veryfast")

                video_clip.close(); img1_clip.close(); img2_clip.close(); composite.close()

                if output_video_path.exists() and output_video_path.stat().st_size > 0:
                    return True
                else:
                    generation_logger.debug("overlay_start_end_images_above_video: MoviePy output missing after write.")
            except (OSError, ValueError, RuntimeError) as e_mov:
                generation_logger.debug(f"overlay_start_end_images_above_video: MoviePy path failed \u2013 {e_mov}. Falling back to ffmpeg.")

        # ---------------------------------------------------------
        #   Fallback: FFmpeg filter_complex (no MoviePy or MoviePy failed)
        # ---------------------------------------------------------
        if not _MOVIEPY_AVAILABLE:
            try:
                # ---------------------------------------------------------
                #   Determine the resolution of the **input video**
                # ---------------------------------------------------------
                cap = cv2.VideoCapture(str(input_video_path))
                if not cap.isOpened():
                    generation_logger.debug(f"overlay_start_end_images_above_video: Could not open video {input_video_path}")
                    return False
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if video_width == 0 or video_height == 0:
                    generation_logger.debug(
                        f"overlay_start_end_images_above_video: Failed to read resolution from {input_video_path}"
                    )
                    return False

                half_width = video_width // 2  # Integer division for even-pixel alignment

                # Ensure output directory exists
                output_video_path = output_video_path.with_suffix('.mp4')
                output_video_path.parent.mkdir(parents=True, exist_ok=True)

                # ---------------------------------------------------------
                #   Build & run ffmpeg command
                # ---------------------------------------------------------
                # 1) Scale both images to half width / full height
                # 2) hstack -> side-by-side static banner (labelled [top])
                # 3) vstack banner with original video (labelled [vid])
                # NOTE: We explicitly scale the *video* as well into label [vid]
                #       to guarantee width/height match.  This is mostly defensive --
                #       if the video is already the desired size the scale is a NOP.

                filter_complex = (
                    f"[1:v]scale={half_width}:{video_height}[left];"  # scale start img
                    f"[2:v]scale={half_width}:{video_height}[right];"  # scale end img
                    f"[left][right]hstack=inputs=2[top];"              # combine images
                    f"[0:v]scale={video_width}:{video_height}[vid];"   # ensure video size
                    f"[top][vid]vstack=inputs=2[output]"              # stack banner + video
                )

                # Determine FPS from the input video for consistent output
                fps = 0.0
                try:
                    cap2 = cv2.VideoCapture(str(input_video_path))
                    if cap2.isOpened():
                        fps = cap2.get(cv2.CAP_PROP_FPS)
                    cap2.release()
                except (OSError, ValueError, RuntimeError) as e:
                    generation_logger.debug(f"overlay_start_end_images_above_video: DEBUG: Failed to read FPS from {input_video_path}: {e}")
                    fps = 0.0
                if fps is None or fps <= 0.1:
                    fps = 16  # sensible default

                ffmpeg_cmd = [
                    "ffmpeg", "-y",  # overwrite output
                    "-loglevel", "error",
                    "-i", str(input_video_path),
                    "-loop", "1", "-i", str(start_image_path),
                    "-loop", "1", "-i", str(end_image_path),
                    "-filter_complex", filter_complex,
                    "-map", "[output]",
                    "-r", str(int(round(fps))),  # set output fps
                    "-shortest",  # stop when primary video stream ends
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_video_path.resolve()),
                ]

                generation_logger.debug(f"overlay_start_end_images_above_video: Running ffmpeg to create composite video.\nCommand: {' '.join(ffmpeg_cmd)}")

                proc = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=600)
                if proc.returncode != 0:
                    generation_logger.debug(
                        f"overlay_start_end_images_above_video: ffmpeg failed (returncode={proc.returncode}).\n"
                        f"stderr: {proc.stderr.decode(errors='ignore')[:500]}"
                    )
                    # Clean up partially written file if any
                    if output_video_path.exists():
                        try:
                            output_video_path.unlink()
                        except OSError as e_cleanup:
                            generation_logger.debug(f"overlay_start_end_images_above_video: DEBUG: Failed to clean up partial output {output_video_path}: {e_cleanup}")
                    return False

                if not output_video_path.exists() or output_video_path.stat().st_size == 0:
                    generation_logger.debug(
                        f"overlay_start_end_images_above_video: Output video not created or empty at {output_video_path}"
                    )
                    return False

                return True

            except (subprocess.SubprocessError, OSError) as e_ffmpeg:
                generation_logger.debug(f"overlay_start_end_images_above_video: ffmpeg failed \u2013 {e_ffmpeg}. Falling back to MoviePy.")
                return False

    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e_ov:
        generation_logger.debug(f"overlay_start_end_images_above_video: Exception \u2013 {e_ov}", exc_info=True)
        try:
            if output_video_path and output_video_path.exists():
                output_video_path.unlink()
        except OSError as e_cleanup:
            generation_logger.debug(f"overlay_start_end_images_above_video: DEBUG: Failed to clean up output {output_video_path}: {e_cleanup}")
        return False
