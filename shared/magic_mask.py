import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageOps

from shared.utils.audio_video import _get_codec_params
from shared.utils.utils import get_resampled_video_transparent, get_video_info_details, has_image_file_extension, rgb_bw_to_rgba_mask, sanitize_file_name
from shared.utils.virtual_media import get_virtual_image, strip_virtual_media_suffix


PROCESS_ID = "magic_mask"
PROCESS_NAME = "Magic Mask"
DOWNLOAD_REPO_ID = "DeepBeepMeep/Wan2.1"
DOWNLOAD_FOLDER = "sam3"
DOWNLOAD_FILES = ["sam3.1_multiplex_bf16.safetensors", "bpe_simple_vocab_16e6.txt.gz"]
DEFAULT_FILL_HOLE_AREA = 2
DEFAULT_POSTPROCESS_BATCH_SIZE = 1
OUTPUT_DIR = "mask_outputs"


def parse_keywords(keyword_text: str | Iterable[str]) -> list[str]:
    if isinstance(keyword_text, str):
        candidates = re.split(r"[\n,;]+", keyword_text)
    else:
        candidates = keyword_text
    return [str(keyword).strip() for keyword in candidates if str(keyword).strip()]


def query_download_def():
    return {"repoId": DOWNLOAD_REPO_ID, "sourceFolderList": [DOWNLOAD_FOLDER], "fileList": [list(DOWNLOAD_FILES)]}


def _fill_hole_area(no_hole):
    return DEFAULT_FILL_HOLE_AREA if bool(no_hole) else 0


def _open_image(image):
    if isinstance(image, dict):
        image = image.get("path") or image.get("name") or image.get("orig_name")
    virtual_image = get_virtual_image(image) if isinstance(image, str) else None
    if virtual_image is not None:
        image = virtual_image
    elif isinstance(image, str):
        image = Image.open(strip_virtual_media_suffix(image))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise ValueError("Magic Mask needs a control image.")
    return ImageOps.exif_transpose(image).convert("RGB")


def _media_path(path):
    if isinstance(path, dict):
        path = path.get("path") or path.get("name") or path.get("orig_name")
    return path


def _video_to_numpy(video_path):
    video_path = _media_path(video_path)
    if not video_path:
        raise ValueError("Magic Mask needs a control video.")
    if isinstance(video_path, str) and has_image_file_extension(video_path):
        image = _open_image(video_path)
        width, height = image.size
        return np.asarray(image, dtype=np.uint8)[None], 1, width, height
    details = get_video_info_details(video_path)
    fps = details.get("fps_float") or details.get("fps") or 1
    width = int(details.get("display_width") or details.get("width") or 0)
    height = int(details.get("display_height") or details.get("height") or 0)
    frame_count = int(details.get("frame_count") or 1)
    frames = get_resampled_video_transparent(video_path, 0, frame_count, fps, bridge="torch")
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    elif hasattr(frames, "asnumpy"):
        frames = frames.asnumpy()
    else:
        frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("Magic Mask could not read any control video frames.")
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if width > 0 and height > 0 and frames.shape[1:3] != (height, width):
        frames = np.stack([np.asarray(Image.fromarray(frame).resize((width, height), resample=Image.Resampling.LANCZOS)) for frame in frames], axis=0)
    return frames.astype(np.uint8, copy=False), fps, width, height


def _run_sam3(video: np.ndarray, keywords: list[str], batch_size, no_hole, progress_callback=None) -> np.ndarray:
    from preprocessing.sam3.preprocessor import run_sam3_video

    with torch.inference_mode():
        return run_sam3_video(
            video,
            keywords,
            batched_grounding_batch_size=batch_size,
            postprocess_batch_size=DEFAULT_POSTPROCESS_BATCH_SIZE,
            use_batched_grounding=True,
            fill_hole_area=_fill_hole_area(no_hole),
            progress_callback=progress_callback,
        )


def prepare_image_mask_input(image) -> tuple[Image.Image, np.ndarray]:
    image = _open_image(image)
    return image, np.asarray(image, dtype=np.uint8)[None]


def prepare_video_mask_input(video_path) -> tuple[str, np.ndarray, int]:
    video_path = _media_path(video_path)
    if not video_path:
        raise ValueError("Magic Mask needs a control video.")
    video, fps, _, _ = _video_to_numpy(video_path)
    return video_path, video, fps


def generate_keyword_masks(video: np.ndarray, keyword_text: str | Iterable[str], *, batch_size=None, no_hole=True, progress_callback=None) -> np.ndarray:
    keywords = parse_keywords(keyword_text)
    if len(keywords) == 0:
        return np.zeros(video.shape[:3], dtype=np.bool_)
    return _run_sam3(video, keywords, batch_size, no_hole, progress_callback=progress_callback)


def merge_keyword_masks(current_mask: np.ndarray | None, keyword_mask: np.ndarray) -> np.ndarray:
    keyword_mask = keyword_mask.astype(bool, copy=False)
    return keyword_mask.copy() if current_mask is None else (current_mask | keyword_mask)


def finalize_masks(mask: np.ndarray, *, negative_mask=False) -> np.ndarray:
    if negative_mask:
        mask = ~mask
    return mask


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def _magic_mask_video_codec_params():
    params = dict(_get_codec_params("libx264_10", "mp4"))
    params["macro_block_size"] = 1
    if params.get("pixelformat") == "yuv420p":
        params["pixelformat"] = "yuv444p"
    return params


def save_mask_video(video_path: str, masks: np.ndarray, fps: float, keywords: list[str], *, codec_type=None, output_dir=OUTPUT_DIR, abort_callback=None) -> str:
    # codec_type is kept for compatibility; Magic Mask outputs are always MP4 libx264_10.
    masks = masks.astype(np.uint8) * 255
    mask_frames = np.repeat(masks[..., None], 3, axis=-1)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(strip_virtual_media_suffix(video_path)).stem
    keywords_suffix = truncate_keywords_for_path(keywords)
    output_path = Path(output_dir) / f"{sanitize_file_name(stem)}_magic_mask_{keywords_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.fspath(output_path)
    writer = imageio.get_writer(output_path, fps=fps, ffmpeg_log_level="error", **_magic_mask_video_codec_params())
    try:
        for frame in mask_frames:
            if abort_callback is not None:
                abort_callback()
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


def generate_image_mask(image, keyword_text, *, batch_size=None, no_hole=True, negative_mask=False) -> tuple[Image.Image, Image.Image, list[str]]:
    keywords = parse_keywords(keyword_text)
    if len(keywords) == 0:
        raise ValueError("Enter at least one keyword.")
    image, video = prepare_image_mask_input(image)
    mask = finalize_masks(_run_sam3(video, keywords, batch_size, no_hole)[0], negative_mask=negative_mask)
    mask_image = mask_to_image(mask)
    return image, mask_image, keywords


def generate_video_mask(video_path, keyword_text, *, batch_size=None, no_hole=True, negative_mask=False, codec_type=None, output_dir=OUTPUT_DIR) -> tuple[str, list[str]]:
    keywords = parse_keywords(keyword_text)
    if len(keywords) == 0:
        raise ValueError("Enter at least one keyword.")
    video_path, video, fps = prepare_video_mask_input(video_path)
    masks = finalize_masks(_run_sam3(video, keywords, batch_size, no_hole), negative_mask=negative_mask)
    return save_mask_video(video_path, masks, fps, keywords, output_dir=output_dir), keywords


def truncate_keywords_for_path(keywords: list[str]) -> str:
    suffix = sanitize_file_name("_".join(keywords), "_").strip("_")
    return suffix[:40] or "mask"


def build_image_editor_value(background: Image.Image, mask_image: Image.Image):
    return {"background": background, "composite": None, "layers": [rgb_bw_to_rgba_mask(mask_image)]}
