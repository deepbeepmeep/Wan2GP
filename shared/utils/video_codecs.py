SDR_VIDEO_CODEC_CHOICES = [
    ("x265 CRF 28 (Balanced)", "libx265_28"),
    ("x264 Level 8 (Balanced)", "libx264_8"),
    ("x265 CRF 8 (High Quality)", "libx265_8"),
    ("x264 Level 10 (High Quality)", "libx264_10"),
    ("x264 Lossless", "libx264_lossless"),
    ("ProRes 422 (editing)", "prores_422"),
    ("DNxHR HQ (editing)", "dnxhr_hq"),
]

VIDEO_CONTAINER_CHOICES = [
    ("MP4", "mp4"),
    ("MOV / QuickTime", "mov"),
    ("MKV / Matroska", "mkv"),
]

SUPPORTED_VIDEO_CONTAINERS = {"mkv", "mov", "mp4"}
CONFIG_VIDEO_CONTAINERS = {value for _, value in VIDEO_CONTAINER_CHOICES}
PROFESSIONAL_VIDEO_CODECS = {"prores_422", "dnxhr_hq"}
QUICKTIME_AUDIO_CODEC_KEYS = {"aac_128", "aac_192", "aac_256", "aac_320", "alac"}


def normalize_video_container(container: str | None) -> str:
    return str(container or "mp4").strip().lower() or "mp4"


def normalize_video_codec(codec_key: str | None) -> str:
    return str(codec_key or "libx264_8").strip().lower() or "libx264_8"


def normalize_video_audio_codec(codec_key: str | None) -> str:
    return str(codec_key or "aac_128").strip().lower() or "aac_128"


def get_video_container_extension(container: str | None) -> str:
    container = normalize_video_container(container)
    return f".{container}" if container in SUPPORTED_VIDEO_CONTAINERS else ".mp4"


def get_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    codec_key = normalize_video_codec(codec_key)
    container = normalize_video_container(container)
    if codec_key == "libx264_8":
        return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]
    if codec_key == "libx264_10":
        return ["-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p"]
    if codec_key == "libx265_28":
        return ["-c:v", "libx265", "-crf", "28", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx265_8":
        return ["-c:v", "libx265", "-crf", "8", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none"]
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return ["-c:v", "ffv1", "-pix_fmt", "rgb24"]
        return ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"]
    if codec_key == "prores_422":
        return ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"]
    if codec_key == "dnxhr_hq":
        return ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"]
    return ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"]


def get_imageio_codec_params(codec_key: str | None, container: str | None) -> dict:
    codec_key = normalize_video_codec(codec_key)
    container = normalize_video_container(container)
    if codec_key == "libx264_8":
        return {"codec": "libx264", "quality": 8, "pixelformat": "yuv420p"}
    if codec_key == "libx264_10":
        return {"codec": "libx264", "quality": 10, "pixelformat": "yuv420p"}
    if codec_key == "libx265_28":
        return {"codec": "libx265", "pixelformat": "yuv420p", "output_params": ["-crf", "28", "-x265-params", "log-level=none", "-hide_banner", "-nostats"]}
    if codec_key == "libx265_8":
        return {"codec": "libx265", "pixelformat": "yuv420p", "output_params": ["-crf", "8", "-x265-params", "log-level=none", "-hide_banner", "-nostats"]}
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return {"codec": "ffv1", "pixelformat": "rgb24"}
        return {"codec": "libx264", "output_params": ["-crf", "0"], "pixelformat": "yuv444p"}
    if codec_key == "prores_422":
        return {"codec": "prores_ks", "pixelformat": "yuv422p10le", "output_params": ["-profile:v", "2", "-hide_banner", "-nostats"]}
    if codec_key == "dnxhr_hq":
        return {"codec": "dnxhd", "pixelformat": "yuv422p", "output_params": ["-profile:v", "dnxhr_hq", "-hide_banner", "-nostats"]}
    return {"codec": "libx264", "pixelformat": "yuv420p"}


def validate_video_output_settings(video_codec: str | None, video_container: str | None, audio_codec: str | None = None, width: int | None = None, height: int | None = None, *, allowed_containers: set[str] | None = None) -> str | None:
    video_codec = normalize_video_codec(video_codec)
    video_container = normalize_video_container(video_container)
    audio_codec = normalize_video_audio_codec(audio_codec)
    allowed = CONFIG_VIDEO_CONTAINERS if allowed_containers is None else allowed_containers
    if video_container not in allowed:
        return f"Unsupported video container: {video_container}."
    if video_codec in PROFESSIONAL_VIDEO_CODECS and video_container not in {"mkv", "mov"}:
        return "ProRes 422 and DNxHR HQ require the MOV / QuickTime or MKV container."
    if video_container in {"mp4", "mov"} and audio_codec not in QUICKTIME_AUDIO_CODEC_KEYS:
        return f"{video_container.upper()} output does not support audio codec setting '{audio_codec}'."
    if video_codec == "dnxhr_hq" and width is not None and height is not None and (int(width) < 256 or int(height) < 120):
        return "DNxHR HQ output requires a resolution of at least 256x120."
    return None
