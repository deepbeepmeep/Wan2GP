"""Transparent video companions from a sequence of uint8 BGRA frames."""
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import zipfile

import cv2

from .video_codecs import get_video_encode_args
from .video_decode import resolve_media_binary


def write_zip_file(zip_path, frames, interrupt_check=None):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, frame in enumerate(frames):
            if interrupt_check is not None and interrupt_check():
                return False
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                raise RuntimeError(f"Failed to encode RGBA frame {index}")
            archive.writestr(f"img_{index:03d}.png", encoded.tobytes())
    return True


def rgba_video_side_files(frames, fps, output_format="png_zip", interrupt_check=None):
    """Return companion bytes, or None on cancellation; final names belong to WanGP."""
    if interrupt_check is not None and interrupt_check():
        return None
    if output_format == "png_zip":
        with BytesIO() as stream:
            if not write_zip_file(stream, frames, interrupt_check):
                return None
            return {".zip": stream.getvalue()}
    if output_format != "prores_4444":
        raise ValueError(f"Unsupported RGBA video output: {output_format}")

    ffmpeg = resolve_media_binary("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for ProRes 4444 output")
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(prefix="wangp_rgba_") as directory, tempfile.TemporaryFile() as errors:
        output = Path(directory) / "rgba.mov"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-video_size", f"{width}x{height}", "-framerate", f"{float(fps):.12g}", "-i", "pipe:0", "-an", *get_video_encode_args("prores_4444", "mov"), str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            try:
                for frame in frames:
                    if interrupt_check is not None and interrupt_check():
                        return None
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
            except BrokenPipeError:
                pass  # Report FFmpeg's encoder error below.
            while True:
                if interrupt_check is not None and interrupt_check():
                    return None
                try:
                    returncode = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if returncode != 0:
                errors.seek(0)
                raise RuntimeError(f"ProRes 4444 encoding failed: {errors.read().decode('utf-8', errors='replace').strip()}")
            return {"_rgba.mov": output.read_bytes()}
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
