import json
import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np
import torch

from .virtual_media import clamp_virtual_frame_range, parse_virtual_media_path, strip_virtual_media_suffix

_ZSCALE_TRANSFER_MAP = {"smpte2084": "smpte2084", "arib-std-b67": "arib-std-b67", "bt709": "bt709", "bt2020-10": "2020_10", "bt2020-12": "2020_12"}
_ZSCALE_PRIMARIES_MAP = {"bt2020": "2020", "bt709": "709", "smpte170m": "170m", "bt470bg": "470bg"}
_ZSCALE_MATRIX_MAP = {"bt2020nc": "2020_ncl", "bt2020c": "2020_cl", "bt709": "709", "smpte170m": "170m", "bt470bg": "470bg"}
_ZSCALE_RANGE_MAP = {"tv": "limited", "limited": "limited", "pc": "full", "full": "full"}
_HDR_REFERENCE_WHITE_NITS = 203


def _parse_media_ratio(value, default=None):
    if value in [None, "", "N/A", "0:1", "0/0"]:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        num, den = text.split(":", 1)
    elif "/" in text:
        num, den = text.split("/", 1)
    else:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return default
    return default if den == 0 else num / den


def _resample_frame_indices(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame):
    import math

    video_frame_duration = 1 / video_fps
    target_frame_duration = 1 / target_fps
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)
    cur_time = frame_no * video_frame_duration
    frame_ids = []
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count:
            break
        diff = round((target_time - cur_time) / video_frame_duration, 5)
        add_frames_count = math.ceil(diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    return frame_ids[:max_target_frames_count]


def _resolve_media_binary(binary_name: str):
    env_map = {"ffmpeg": "FFMPEG_BINARY", "ffprobe": "FFPROBE_BINARY", "ffplay": "FFPLAY_BINARY"}
    binary_path = os.environ.get(env_map.get(binary_name, ""), "")
    if len(binary_path) > 0 and os.path.isfile(binary_path):
        return binary_path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(repo_root, "ffmpeg_bins", binary_name + (".exe" if os.name == "nt" else ""))
    if os.path.isfile(candidate):
        return candidate
    return shutil.which(binary_name + (".exe" if os.name == "nt" else "")) or shutil.which(binary_name)


def resolve_media_binary(binary_name: str):
    return _resolve_media_binary(binary_name)


def _augment_virtual_metadata(video_path, metadata):
    spec = parse_virtual_media_path(video_path)
    if spec is None or metadata is None:
        return metadata
    total_frames = int(metadata.get("frame_count") or 0)
    start_frame, end_frame = clamp_virtual_frame_range(spec, total_frames)
    virtual_metadata = dict(metadata)
    virtual_metadata["source_path"] = spec.source_path
    virtual_metadata["virtual_start_frame"] = start_frame
    virtual_metadata["virtual_end_frame"] = end_frame
    if end_frame is None:
        return virtual_metadata
    virtual_frame_count = max(0, end_frame - start_frame + 1)
    virtual_metadata["frame_count"] = virtual_frame_count
    fps_float = float(virtual_metadata.get("fps_float") or 0.0)
    fps = int(virtual_metadata.get("fps") or 0)
    effective_fps = fps_float if fps_float > 0 else float(fps or 0)
    if effective_fps > 0:
        virtual_metadata["duration"] = virtual_frame_count / effective_fps
    return virtual_metadata


@lru_cache(maxsize=128)
def probe_video_stream_metadata(video_path):
    video_path = os.fspath(video_path)
    source_path = os.fspath(strip_virtual_media_suffix(video_path))
    ffprobe_path = _resolve_media_binary("ffprobe")
    if ffprobe_path is None:
        return None
    probe_cmd = [ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", source_path]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
    if probe.returncode != 0:
        return None
    try:
        probe_data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return None
    streams = probe_data.get("streams") or []
    if len(streams) == 0:
        return None
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    sar = _parse_media_ratio(stream.get("sample_aspect_ratio"), 1.0) or 1.0
    dar = _parse_media_ratio(stream.get("display_aspect_ratio"))
    display_width = width
    if abs(sar - 1.0) > 1e-6:
        display_width = max(2, (int(width * sar) // 2) * 2)
    elif dar is not None and dar > 0:
        display_width = max(2, (int(height * dar) // 2) * 2)
    fps_float = _parse_media_ratio(stream.get("avg_frame_rate"), 0.0) or _parse_media_ratio(stream.get("r_frame_rate"), 0.0) or 0.0
    duration = stream.get("duration") or (probe_data.get("format") or {}).get("duration") or 0.0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = int(round(duration * fps_float)) if duration > 0 and fps_float > 0 else 0
    side_data = stream.get("side_data_list") or []
    color_transfer = str(stream.get("color_transfer") or "").lower()
    color_primaries = str(stream.get("color_primaries") or "").lower()
    color_space = str(stream.get("color_space") or "").lower()
    color_range = str(stream.get("color_range") or "").lower()
    sample_aspect_ratio = str(stream.get("sample_aspect_ratio") or "1:1")
    display_aspect_ratio = str(stream.get("display_aspect_ratio") or "")
    is_hdr = color_transfer in {"smpte2084", "arib-std-b67"} or color_primaries == "bt2020" or any(
        str(item.get("side_data_type") or "").lower() in {"mastering display metadata", "content light level metadata"} for item in side_data
    )
    return _augment_virtual_metadata(video_path, {
        "width": width,
        "height": height,
        "display_width": display_width,
        "display_height": height,
        "fps_float": fps_float,
        "fps": int(round(fps_float)) if fps_float > 0 else 0,
        "frame_count": frame_count,
        "duration": duration,
        "sample_aspect_ratio": sample_aspect_ratio,
        "display_aspect_ratio": display_aspect_ratio,
        "color_transfer": color_transfer,
        "color_primaries": color_primaries,
        "color_space": color_space,
        "color_range": color_range,
        "needs_sar_fix": display_width != width,
        "needs_tonemap": is_hdr,
    })


def video_needs_corrected_decode(video_path):
    metadata = probe_video_stream_metadata(video_path)
    return metadata is not None and (metadata["needs_sar_fix"] or metadata["needs_tonemap"])


def _build_hdr_tonemap_filter(metadata):
    zscale_parts = ["t=linear", f"npl={_HDR_REFERENCE_WHITE_NITS}"]
    if transfer := _ZSCALE_TRANSFER_MAP.get(metadata["color_transfer"]):
        zscale_parts.append(f"tin={transfer}")
    if primaries := _ZSCALE_PRIMARIES_MAP.get(metadata["color_primaries"]):
        zscale_parts.append(f"pin={primaries}")
    if matrix := _ZSCALE_MATRIX_MAP.get(metadata["color_space"]):
        zscale_parts.append(f"min={matrix}")
    if color_range := _ZSCALE_RANGE_MAP.get(metadata.get("color_range")):
        zscale_parts.append(f"rin={color_range}")
    return ["zscale=" + ":".join(zscale_parts), "format=gbrpf32le", "tonemap=reinhard", "zscale=t=bt709:p=bt709:m=bt709:r=limited"]


def _build_corrected_video_filter(metadata, target_fps=None, start_frame=0, end_frame=None):
    filters = []
    if target_fps is not None and float(target_fps) > 0:
        filters.append(f"fps={float(target_fps):.12g}")
    if start_frame > 0 or end_frame is not None:
        trim_parts = [f"start_frame={int(start_frame)}"]
        if end_frame is not None:
            trim_parts.append(f"end_frame={int(end_frame)}")
        filters.append("trim=" + ":".join(trim_parts))
        filters.append("setpts=PTS-STARTPTS")
    if metadata["needs_sar_fix"]:
        filters += [f"scale={int(metadata['display_width'])}:{int(metadata['display_height'])}:flags=lanczos", "setsar=1"]
    if metadata["needs_tonemap"]:
        filters += _build_hdr_tonemap_filter(metadata)
    return ",".join(filters)


def _read_exact(stream, size):
    buf = bytearray(size)
    view = memoryview(buf)
    read_pos = 0
    while read_pos < size:
        chunk = stream.read(size - read_pos)
        if not chunk:
            return None if read_pos == 0 else bytes(view[:read_pos])
        view[read_pos:read_pos + len(chunk)] = chunk
        read_pos += len(chunk)
    return buf


def _decode_contiguous_video_frames_ffmpeg(video_path, start_frame, max_frames, bridge="torch"):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    decode_path = os.fspath(metadata.get("source_path") or strip_virtual_media_suffix(video_path))
    ffmpeg_path = _resolve_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg binary not found")
    start_frame = int(start_frame)
    max_frames = int(max_frames)
    if metadata.get("virtual_end_frame") is not None:
        available_frames = max(0, int(metadata["frame_count"]) - max(0, start_frame))
        max_frames = min(max_frames, available_frames)
    if max_frames <= 0:
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=np.uint8)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    actual_start = start_frame + int(metadata.get("virtual_start_frame") or 0)
    fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
    video_filter = _build_corrected_video_filter(metadata)
    cmd = [ffmpeg_path, "-v", "error", "-nostdin", "-threads", "0"]
    if fps_float > 0 and actual_start > 0:
        cmd += ["-ss", f"{actual_start / fps_float:.12g}"]
    cmd += ["-i", decode_path, "-an", "-sn"]
    if len(video_filter) > 0:
        cmd += ["-vf", video_filter]
    cmd += ["-fps_mode", "passthrough", "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    frame_bytes = metadata["display_width"] * metadata["display_height"] * 3
    frames = np.empty((max_frames, metadata["display_height"], metadata["display_width"], 3), dtype=np.uint8)
    frame_count = 0
    try:
        while frame_count < max_frames:
            raw_frame = _read_exact(process.stdout, frame_bytes)
            if raw_frame is None or len(raw_frame) < frame_bytes:
                break
            frames[frame_count] = np.frombuffer(raw_frame, dtype=np.uint8).reshape(metadata["display_height"], metadata["display_width"], 3)
            frame_count += 1
        stderr = process.stderr.read().decode("utf-8", errors="ignore").strip()
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if return_code != 0 and frame_count == 0:
        raise RuntimeError(f"ffmpeg decode failed for {video_path}: {stderr}")
    frames = frames[:frame_count]
    return torch.from_numpy(frames) if bridge == "torch" else frames


def decode_video_frames_ffmpeg(video_path, start_frame, max_frames, target_fps=None, bridge="torch"):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    start_frame = int(start_frame)
    if metadata.get("virtual_end_frame") is not None and start_frame >= int(metadata["frame_count"]):
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=np.uint8)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    if target_fps is None or float(target_fps) <= 0:
        return _decode_contiguous_video_frames_ffmpeg(video_path, start_frame, max_frames, bridge)
    source_fps = metadata["fps"] if metadata["fps"] > 0 else max(1, int(round(metadata["fps_float"] or 0)))
    frame_nos = _resample_frame_indices(source_fps, metadata["frame_count"], int(max_frames), float(target_fps), int(start_frame))
    if len(frame_nos) == 0:
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=np.uint8)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    decode_start = frame_nos[0]
    decoded = _decode_contiguous_video_frames_ffmpeg(video_path, decode_start, frame_nos[-1] - decode_start + 1, bridge)
    index_list = [frame_no - decode_start for frame_no in frame_nos if frame_no - decode_start < decoded.shape[0]]
    if bridge == "torch":
        return decoded[index_list]
    return decoded[index_list]


def get_video_summary_extras(video_path):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        return [], []
    values, labels = [], []
    if metadata["needs_sar_fix"]:
        values += [f"{metadata['width']}x{metadata['height']}", metadata["sample_aspect_ratio"]]
        labels += ["Stored Raster", "Pixel Aspect Ratio"]
        if len(metadata["display_aspect_ratio"]) > 0:
            values += [f"{metadata['display_aspect_ratio']} (square-pixel {metadata['display_width']}x{metadata['display_height']})"]
            labels += ["Display Aspect Ratio"]
    if metadata["needs_tonemap"]:
        hdr_parts = []
        if metadata["color_transfer"] == "smpte2084":
            hdr_parts += ["HDR PQ"]
        elif metadata["color_transfer"] == "arib-std-b67":
            hdr_parts += ["HDR HLG"]
        elif len(metadata["color_transfer"]) > 0:
            hdr_parts += [metadata["color_transfer"].upper()]
        if len(metadata["color_primaries"]) > 0:
            hdr_parts += [metadata["color_primaries"].upper()]
        if len(metadata["color_space"]) > 0 and metadata["color_space"] != metadata["color_primaries"]:
            hdr_parts += [metadata["color_space"].upper()]
        values += [" / ".join(hdr_parts) if len(hdr_parts) > 0 else "HDR source"]
        labels += ["Color"]
    return values, labels
