import importlib
import importlib.util
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from shared.utils import files_locator as fl
from .logger import get_logger
from .model.device_utils import accelerator_autocast, empty_accelerator_cache, get_accelerator_device, is_accelerator_device


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SAM3_FOLDER = "sam3"
_SAM3_CHECKPOINT_NAME = "sam3.1_multiplex_bf16.safetensors"
_SAM3_BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"
KEEP_VIDEO_FRAMES_ON_CUDA = True
_TEXT_ENCODER_CACHE = None
_TEXT_ENCODER_CACHE_KEY = None
DEFAULT_INSTANCE_PALETTE_RGB = np.array([
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
], dtype=np.uint8)
logger = get_logger(__name__)


def _cleanup():
    import gc

    gc.collect()
    empty_accelerator_cache()


def _load_model_builder():
    try:
        return importlib.import_module(".model_builder", package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name != importlib.util.resolve_name(".model_builder", __package__):
            raise
    raise FileNotFoundError("SAM3.1 code was not found under preprocessing/sam3.")


def _checkpoint_path():
    for candidate in [
        os.path.join(_SAM3_FOLDER, _SAM3_CHECKPOINT_NAME),
        os.path.join("sam3.1", _SAM3_CHECKPOINT_NAME),
        _SAM3_CHECKPOINT_NAME,
    ]:
        checkpoint = fl.locate_file(candidate, error_if_none=False)
        if checkpoint is not None:
            return checkpoint, "sam3.1"
    checkpoint = _PACKAGE_ROOT / _SAM3_CHECKPOINT_NAME
    if checkpoint.is_file():
        return os.fspath(checkpoint), "sam3.1"
    raise FileNotFoundError("SAM3.1 bf16 safetensors checkpoint was not found by files_locator as sam3/sam3.1_multiplex_bf16.safetensors, sam3.1/sam3.1_multiplex_bf16.safetensors, or sam3.1_multiplex_bf16.safetensors, nor under preprocessing/sam3.")


def _bpe_path():
    for candidate in [
        os.path.join(_SAM3_FOLDER, _SAM3_BPE_NAME),
        os.path.join("sam3.1", _SAM3_BPE_NAME),
        _SAM3_BPE_NAME,
    ]:
        bpe_path = fl.locate_file(candidate, error_if_none=False)
        if bpe_path is not None:
            return bpe_path
    bpe_path = _PACKAGE_ROOT / "assets" / _SAM3_BPE_NAME
    if bpe_path.is_file():
        return os.fspath(bpe_path)
    raise FileNotFoundError("SAM3 BPE vocabulary was not found by files_locator as sam3/bpe_simple_vocab_16e6.txt.gz, sam3.1/bpe_simple_vocab_16e6.txt.gz, or bpe_simple_vocab_16e6.txt.gz, nor under preprocessing/sam3/assets.")


def _autocast_context():
    return accelerator_autocast()


def _bf16_prompt_payload(value):
    if torch.is_tensor(value):
        return value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _bf16_prompt_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bf16_prompt_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_bf16_prompt_payload(item) for item in value)
    return value


def _format_keywords_for_log(keywords: list[str]):
    return ", ".join(f"'{keyword}'" for keyword in keywords)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sam3_outputs_to_binary_mask(outputs, height: int, width: int, color_palette=None, object_color_map=None, max_colored_objects=None, fill_hole_area: int = 0):
    if outputs is None or "out_binary_masks" not in outputs:
        return np.zeros((height, width, 3), dtype=np.uint8) if color_palette is not None else np.zeros((height, width), dtype=np.bool_)
    masks = _to_numpy(outputs["out_binary_masks"])
    if masks.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8) if color_palette is not None else np.zeros((height, width), dtype=np.bool_)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim > 3:
        masks = masks.reshape((-1, *masks.shape[-2:]))
    if masks.shape[-2:] != (height, width):
        masks = np.stack([np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((width, height), resample=Image.Resampling.NEAREST)) for mask in masks], axis=0)
    masks = masks.astype(bool, copy=False)
    if color_palette is None and max_colored_objects is None:
        return masks.any(axis=0)
    obj_ids = _to_numpy(outputs.get("out_obj_ids", np.arange(masks.shape[0]))).reshape(-1)
    if obj_ids.shape[0] != masks.shape[0]:
        obj_ids = np.arange(masks.shape[0], dtype=np.int64)
    object_color_map = object_color_map if object_color_map is not None else {}
    obj_ids = obj_ids.astype(np.int64, copy=False)
    max_colored_objects = len(color_palette) if color_palette is not None and max_colored_objects is None else max_colored_objects
    if max_colored_objects <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8) if color_palette is not None else np.zeros((height, width), dtype=np.bool_)
    new_indices = [i for i, obj_id in enumerate(obj_ids) if int(obj_id) not in object_color_map]
    slots = max_colored_objects - len(object_color_map)
    if slots > 0 and new_indices:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        for idx in sorted(new_indices, key=lambda i: areas[i], reverse=True)[:slots]:
            object_color_map[int(obj_ids[idx])] = color_palette[len(object_color_map)] if color_palette is not None else True
    if color_palette is None:
        selected_masks = [mask for obj_id, mask in zip(obj_ids, masks) if int(obj_id) in object_color_map]
        return np.any(selected_masks, axis=0) if selected_masks else np.zeros((height, width), dtype=np.bool_)
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for obj_id, mask in zip(obj_ids, masks):
        obj_id = int(obj_id)
        if obj_id not in object_color_map:
            continue
        if fill_hole_area > 0:
            mask = fill_sam3_binary_mask_holes(mask, fill_hole_area)
        colored_mask[mask] = object_color_map[obj_id]
    return colored_mask


def resolve_sam3_grounding_batch_size(batch_size=None) -> int:
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size > 0:
            return batch_size
    if not torch.cuda.is_available():
        return 2
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 4 if total_vram_gb >= 8 else 2


def _encode_text_outputs(text_encoder, captions: list[str], device: torch.device):
    masks, memories, embeds = [], [], []
    if is_accelerator_device(device):
        text_encoder.to(device=device, dtype=torch.bfloat16)
    for caption in captions:
        with torch.inference_mode(), _autocast_context():
            text_attention_mask, text_memory, text_embeds = text_encoder([caption], device=device)
        masks.append(text_attention_mask.detach().cpu())
        memories.append(text_memory.detach().cpu())
        embeds.append(text_embeds.detach().cpu())
        del text_attention_mask, text_memory, text_embeds
        _cleanup()
    return {
        "language_features": torch.cat(memories, dim=1),
        "language_mask": torch.cat(masks, dim=0),
        "language_embeds": torch.cat(embeds, dim=1),
    }


def _encode_keyword_prompts(model_builder, checkpoint_path: str, bpe_path: str, keywords: list[str], keep_text_encoder_loaded: bool = False):
    global _TEXT_ENCODER_CACHE, _TEXT_ENCODER_CACHE_KEY
    text_encoder = None
    device = get_accelerator_device()
    cache_key = (checkpoint_path, bpe_path)
    preencoded = {}
    try:
        if keep_text_encoder_loaded and _TEXT_ENCODER_CACHE is not None and _TEXT_ENCODER_CACHE_KEY == cache_key:
            text_encoder = _TEXT_ENCODER_CACHE
        else:
            text_encoder = model_builder.build_sam3_text_encoder(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
            if keep_text_encoder_loaded:
                _TEXT_ENCODER_CACHE = text_encoder
                _TEXT_ENCODER_CACHE_KEY = cache_key
        for keyword in keywords:
            preencoded[keyword] = _encode_text_outputs(text_encoder, [keyword, "visual", "geometric"], device)
    finally:
        if keep_text_encoder_loaded and text_encoder is not None:
            text_encoder.to("cpu")
        elif text_encoder is not None:
            del text_encoder
        _cleanup()
    return preencoded


def encode_sam3_keyword_prompts(keywords: Iterable[str], keep_text_encoder_loaded: bool = False):
    keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if len(keywords) == 0:
        return {}
    model_builder = _load_model_builder()
    checkpoint_path, _ = _checkpoint_path()
    bpe_path = _bpe_path()
    return _encode_keyword_prompts(model_builder, checkpoint_path, bpe_path, keywords, keep_text_encoder_loaded=keep_text_encoder_loaded)


def clear_sam3_text_encoder_cache():
    global _TEXT_ENCODER_CACHE, _TEXT_ENCODER_CACHE_KEY
    if _TEXT_ENCODER_CACHE is not None:
        del _TEXT_ENCODER_CACHE
    _TEXT_ENCODER_CACHE = None
    _TEXT_ENCODER_CACHE_KEY = None
    _cleanup()


def fill_sam3_binary_mask_holes(mask: np.ndarray, fill_hole_area: int):
    fill_hole_area = max(0, int(fill_hole_area))
    if fill_hole_area == 0 or not np.any(mask):
        return mask.astype(np.bool_, copy=False)
    from .model.sam3_tracker_utils import fill_holes_in_mask_scores

    scores = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    scores = scores * 2 - 1
    filled = fill_holes_in_mask_scores(scores, max_area=fill_hole_area, fill_holes=True, remove_sprinkles=False)
    return filled[0, 0].numpy() > 0


def _load_predictor(
    model_builder=None,
    checkpoint_path=None,
    bpe_path=None,
    version=None,
    include_text_encoder=True,
    batched_grounding_batch_size=None,
    postprocess_batch_size=1,
    use_batched_grounding=True,
    trim_past_non_cond_mem_for_eval=True,
    fill_hole_area: int = 0,
    manual_model_loading: bool = False,
):
    model_builder = model_builder or _load_model_builder()
    checkpoint_path, version = (checkpoint_path, version) if checkpoint_path is not None and version is not None else _checkpoint_path()
    bpe_path = bpe_path or _bpe_path()
    grounding_batch_size = resolve_sam3_grounding_batch_size(batched_grounding_batch_size)
    return model_builder.build_sam3_predictor(checkpoint_path=checkpoint_path, bpe_path=bpe_path, version=version, use_fa3=False, use_rope_real=True, compile=False, warm_up=False, include_text_encoder=include_text_encoder, postprocess_batch_size=postprocess_batch_size, use_batched_grounding=use_batched_grounding, batched_grounding_batch_size=grounding_batch_size, trim_past_non_cond_mem_for_eval=trim_past_non_cond_mem_for_eval, fill_hole_area=fill_hole_area, manual_model_loading=manual_model_loading)


def load_sam3_mask_predictor(
    *,
    include_text_encoder: bool = True,
    postprocess_batch_size: int = 1,
    use_batched_grounding: bool = True,
    batched_grounding_batch_size=None,
    trim_past_non_cond_mem_for_eval: bool = True,
    fill_hole_area: int = 0,
    manual_model_loading: bool = False,
):
    model_builder = _load_model_builder()
    checkpoint_path, version = _checkpoint_path()
    bpe_path = _bpe_path()
    return _load_predictor(
        model_builder,
        checkpoint_path,
        bpe_path,
        version,
        include_text_encoder=include_text_encoder,
        batched_grounding_batch_size=batched_grounding_batch_size,
        postprocess_batch_size=postprocess_batch_size,
        use_batched_grounding=use_batched_grounding,
        trim_past_non_cond_mem_for_eval=trim_past_non_cond_mem_for_eval,
        fill_hole_area=fill_hole_area,
        manual_model_loading=manual_model_loading,
    )


def run_sam3_video(
    video: np.ndarray,
    keywords: Iterable[str],
    *,
    include_text_encoder: bool = False,
    preencode_text: bool = True,
    batched_grounding_batch_size=None,
    postprocess_batch_size: int = 1,
    use_batched_grounding: bool = True,
    trim_past_non_cond_mem_for_eval: bool = True,
    keep_video_frames_on_cuda: bool = KEEP_VIDEO_FRAMES_ON_CUDA,
    cache_frame_outputs: bool = False,
    fill_hole_area: int = 0,
    colorize_objects: bool = False,
    color_palette=None,
    max_colored_objects=None,
    progress_callback=None,
):
    keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if len(keywords) == 0:
        return np.zeros((*video.shape[:3], 3), dtype=np.uint8) if colorize_objects else np.zeros(video.shape[:3], dtype=np.bool_)

    model_builder = _load_model_builder()
    checkpoint_path, version = _checkpoint_path()
    bpe_path = _bpe_path()
    _cleanup()
    if version == "sam3.1" and preencode_text:
        logger.info("SAM3 encoding keywords before propagation: %s", _format_keywords_for_log(keywords))
        preencoded_prompts = _encode_keyword_prompts(model_builder, checkpoint_path, bpe_path, keywords)
    else:
        preencoded_prompts = None
    video_predictor = None
    video_pil = None
    video_predictor = _load_predictor(
        model_builder,
        checkpoint_path,
        bpe_path,
        version,
        include_text_encoder=include_text_encoder or preencoded_prompts is None,
        batched_grounding_batch_size=batched_grounding_batch_size,
        postprocess_batch_size=postprocess_batch_size,
        use_batched_grounding=use_batched_grounding,
        trim_past_non_cond_mem_for_eval=trim_past_non_cond_mem_for_eval,
        fill_hole_area=0,
    )
    num_frames, height, width, _ = video.shape
    video_pil = [Image.fromarray(video[i]) for i in range(num_frames)]
    session_id = None
    response = video_predictor.handle_request({"type": "start_session", "resource_path": video_pil, "offload_video_to_cpu": not keep_video_frames_on_cuda, "cache_frame_outputs": cache_frame_outputs})
    session_id = response["session_id"]
    dynamic_mask = np.zeros((num_frames, height, width, 3), dtype=np.uint8) if colorize_objects else np.zeros((num_frames, height, width), dtype=np.bool_)
    color_palette = np.asarray(DEFAULT_INSTANCE_PALETTE_RGB if color_palette is None else color_palette, dtype=np.uint8).reshape(-1, 3) if colorize_objects else None
    if colorize_objects:
        max_colored_objects = len(color_palette) if max_colored_objects is None else min(max(0, int(max_colored_objects)), len(color_palette))
    elif max_colored_objects is not None:
        max_colored_objects = max(0, int(max_colored_objects))
    object_color_map = {}

    def merge_outputs(frame_index, outputs):
        frame_mask = _sam3_outputs_to_binary_mask(outputs, height, width, color_palette=color_palette, object_color_map=object_color_map, max_colored_objects=max_colored_objects, fill_hole_area=fill_hole_area)
        if colorize_objects:
            selector = frame_mask.any(axis=-1)
            dynamic_mask[frame_index][selector] = frame_mask[selector]
        else:
            dynamic_mask[frame_index] |= frame_mask

    try:
        total_progress_steps = len(keywords) * num_frames
        for keyword_index, keyword in enumerate(keywords):
            progress_base = keyword_index * num_frames
            logger.info("SAM3 keyword currently being processed: '%s'", keyword)
            request = {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": keyword}
            if preencoded_prompts is not None:
                request["preencoded_text_outputs"] = _bf16_prompt_payload(preencoded_prompts[keyword])
            with _autocast_context():
                result = video_predictor.handle_request(request)
                merge_outputs(0, result.get("outputs") if isinstance(result, dict) else None)
                if progress_callback is not None:
                    progress_callback(progress_base, total_progress_steps)
                internal_progress_seen = False

                def model_progress_callback(done, total):
                    nonlocal internal_progress_seen
                    internal_progress_seen = True
                    progress_callback(min(progress_base + int(done), total_progress_steps), total_progress_steps)

                stream_request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": num_frames,
                }
                if progress_callback is not None:
                    stream_request["progress_callback"] = model_progress_callback
                propagated_frames = 0
                for result in video_predictor.handle_stream_request(stream_request):
                    propagated_frames += 1
                    if progress_callback is not None and not internal_progress_seen:
                        progress_callback(min(progress_base + propagated_frames, total_progress_steps), total_progress_steps)
                    outputs = result["outputs"]
                    merge_outputs(result["frame_index"], outputs)
    finally:
        try:
            if session_id is not None and video_predictor is not None:
                try:
                    video_predictor.handle_request({"type": "close_session", "session_id": session_id})
                except Exception as exc:
                    logger.warning("SAM3 close_session failed during cleanup: %s", exc)
        finally:
            if video_predictor is not None:
                try:
                    video_predictor.shutdown()
                except Exception as exc:
                    logger.warning("SAM3 predictor shutdown failed during cleanup: %s", exc)
        video_predictor = None
        preencoded_prompts = None
        video_pil = None
        _cleanup()
    if fill_hole_area > 0 and not colorize_objects:
        dynamic_mask = np.stack([fill_sam3_binary_mask_holes(mask, fill_hole_area) for mask in dynamic_mask], axis=0)
    return dynamic_mask
