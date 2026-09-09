from __future__ import annotations

import io
import time
from typing import Iterable

from PIL import Image

from .types import PreviewContext, PreviewMedia, PreviewOptions

_NVENC_AVAILABLE: bool | None = None


def _resize_frames(frames: list[Image.Image], max_edge: int) -> list[Image.Image]:
    output = []
    for frame in frames:
        frame = frame.convert("RGB")
        if max(frame.size) > max_edge:
            scale = max_edge / max(frame.size)
            frame = frame.resize((max(2, round(frame.width * scale) // 2 * 2), max(2, round(frame.height * scale) // 2 * 2)), Image.Resampling.LANCZOS)
        elif frame.width % 2 or frame.height % 2:
            frame = frame.resize((max(2, frame.width + frame.width % 2), max(2, frame.height + frame.height % 2)), Image.Resampling.LANCZOS)
        output.append(frame)
    return output


def _encode_webp(frames: list[Image.Image], preview_fps: int, quality: int = 72) -> bytes:
    buffer = io.BytesIO()
    base, remainder = divmod(round(len(frames) * 1000 / preview_fps), len(frames))
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=[base + (index < remainder) for index in range(len(frames))], loop=0, quality=quality, method=4)
    return buffer.getvalue()


def _encode_mp4(frames: list[Image.Image], preview_fps: int) -> bytes | None:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is False:
        return None
    try:
        import av
        import numpy as np

        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}) as container:
            stream = container.add_stream("h264_nvenc", rate=preview_fps)
            stream.pix_fmt = "yuv420p"
            stream.width, stream.height = frames[0].size
            for index, image in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        _NVENC_AVAILABLE = True
        return buffer.getvalue()
    except Exception:
        _NVENC_AVAILABLE = False
        return None


def _encoded_frame_count(payload: bytes) -> int:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return max(1, int(getattr(image, "n_frames", 1)))
    except Exception:
        return 1


def encode_preview(frames: Iterable[Image.Image], context: PreviewContext, options: PreviewOptions, *, decode_ms: float, dropped_count: int = 0) -> PreviewMedia:
    source_frames = [frame.convert("RGB") for frame in frames]
    if not source_frames:
        raise ValueError("preview encoder received no frames")
    started = time.perf_counter()
    selected = _resize_frames(source_frames, options.max_edge)
    first = selected[0]
    warning = None
    payload = _encode_mp4(selected, options.preview_fps) if len(selected) > 1 else None
    if payload is not None:
        media_kind, mime_type, frame_count = "video", "video/mp4", len(selected)
    else:
        try:
            payload = _encode_webp(selected, options.preview_fps, options.webp_quality)
            frame_count = _encoded_frame_count(payload)
        except Exception:
            selected = [first]
            payload = _encode_webp(selected, options.preview_fps, options.webp_quality)
            frame_count = 1
            warning = "Animated preview unavailable; showing first frame"
        media_kind, mime_type = ("animated_image", "image/webp") if frame_count > 1 else ("image", "image/webp")
        if len(selected) > 1 and frame_count == 1:
            warning = warning or "Animated preview contained no temporal change; showing first frame"
    return PreviewMedia(
        generation_id=context.generation_id, context_id=context.context_id, sequence=context.sequence,
        media_kind=media_kind, mime_type=mime_type, data=payload, width=first.width, height=first.height,
        frame_count=frame_count, fps=options.preview_fps, duration_ms=round(frame_count * 1000 / options.preview_fps),
        step=context.step, total_steps=context.total_steps, decoder_id=context.decoder_id or "",
        decode_ms=round(float(decode_ms), 3), encode_ms=round((time.perf_counter() - started) * 1000, 3),
        dropped_count=dropped_count, warning=warning, pass_no=context.pass_no, window_no=context.window_no, first_frame=first,
    )
