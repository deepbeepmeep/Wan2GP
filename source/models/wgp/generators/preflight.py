"""Pre-generation setup: SVI bridging, model-specific parameter configuration,
and image input preparation.

These functions run before the WGP generate_video() call to ensure all
parameters are in the format that WGP expects.
"""

from typing import Any, Callable, Dict, Optional

from source.core.log import generation_logger, is_debug_enabled
from source.runtime.wgp_bridge import get_model_min_frames_and_step


def prepare_svi_image_refs(kwargs: dict) -> None:
    """Convert ``image_refs_paths`` (list[str]) to PIL ``image_refs`` (in-place).

    Our task pipeline passes paths for JSON-serializability, but Wan2GP/WGP
    expects ``image_refs`` as a list of PIL.Image objects.
    """
    strict_alignment = "image_refs_strengths" in kwargs
    try:
        if (
            "image_refs_paths" in kwargs
            and kwargs.get("image_refs") in (None, "", [])
            and isinstance(kwargs["image_refs_paths"], (list, tuple))
            and len(kwargs["image_refs_paths"]) > 0
        ):
            from PIL import Image
            from PIL import ImageOps

            refs: list = []
            for p in kwargs["image_refs_paths"]:
                if not p:
                    if strict_alignment:
                        raise ValueError(
                            "Anchor image path is empty "
                            "(alignment invariant violated)"
                        )
                    continue
                try:
                    img = Image.open(str(p)).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    refs.append(img)
                except (OSError, ValueError, RuntimeError) as e_img:
                    if strict_alignment:
                        raise ValueError(
                            "Anchor image failed to load "
                            f"(alignment invariant violated): {p}: {e_img}"
                        ) from e_img
                    if is_debug_enabled():
                        generation_logger.warning(
                            f"[SVI_GROUND_TRUTH] Failed to load image_ref path '{p}': {e_img}"
                        )
            if refs:
                kwargs["image_refs"] = refs
                if is_debug_enabled():
                    generation_logger.debug(
                        f"[SVI_GROUND_TRUTH] Converted image_refs_paths -> image_refs (count={len(refs)})"
                    )
            else:
                if is_debug_enabled():
                    generation_logger.warning(
                        "[SVI_GROUND_TRUTH] image_refs_paths provided but no images could be loaded"
                    )
    except ValueError:
        if strict_alignment:
            raise
        if is_debug_enabled():
            generation_logger.warning(
                "[SVI_GROUND_TRUTH] Exception while converting image_refs_paths -> image_refs"
            )
    except (OSError, RuntimeError, TypeError) as e_refs:
        if is_debug_enabled():
            generation_logger.warning(
                f"[SVI_GROUND_TRUTH] Exception while converting image_refs_paths -> image_refs: {e_refs}"
            )


def configure_model_specific_params(
    *,
    is_flux: bool,
    is_qwen: bool,
    is_z_image: bool = False,
    is_vace: bool,
    resolved_params: dict,
    final_video_length: int,
    final_batch_size: int,
    final_guidance_scale: float,
    final_embedded_guidance: float,
    video_guide: Optional[str],
    video_mask: Optional[str],
    video_prompt_type: Optional[str],
    control_net_weight: Optional[float],
    control_net_weight2: Optional[float],
    model_type: str = "",
) -> Dict[str, Any]:
    """Compute model-specific generation parameters.

    Returns a dict with keys:
        image_mode, actual_video_length, actual_batch_size, actual_guidance,
        video_guide, video_mask, video_prompt_type, control_net_weight, control_net_weight2
    """
    if is_flux:
        image_mode = 1
        actual_video_length = 1
        actual_batch_size = final_video_length
        actual_guidance = final_embedded_guidance
    elif is_qwen:
        image_mode = 1
        actual_video_length = 1
        actual_batch_size = resolved_params.get("batch_size", 1)
        actual_guidance = final_guidance_scale
    elif is_z_image:
        # Z Image models are image-only (image_outputs: True in handler).
        # They produce single images, not videos — set image_mode=1 and
        # do NOT boost video_length to the video minimum frame count.
        image_mode = 1
        actual_video_length = 1
        actual_batch_size = resolved_params.get("batch_size", 1)
        actual_guidance = final_guidance_scale
    else:
        # T2V or VACE
        image_mode = 0
        actual_video_length = final_video_length

        # Safety check: models crash if video_length < frames_minimum
        # (e.g. Wan needs >= 5, LTX-2 needs >= 17)
        min_frames = 5  # Safe default
        try:
            _model_key = model_type or resolved_params.get("model") or resolved_params.get("model_name", "")
            _min, _step, _latent = get_model_min_frames_and_step(_model_key)
            min_frames = max(_min, 5)
        except (ImportError, TypeError, ValueError, KeyError, AttributeError):
            pass
        if actual_video_length < min_frames:
            generation_logger.warning(
                f"[SAFETY] Boosting video_length from {actual_video_length} to {min_frames} "
                "to prevent quantization crash"
            )
            actual_video_length = min_frames

        actual_batch_size = final_batch_size
        actual_guidance = final_guidance_scale

    # Disable VACE controls when not applicable
    if not is_vace and not video_guide and not video_mask:
        video_guide = None
        video_mask = None
        if video_prompt_type is None or video_prompt_type == "":
            video_prompt_type = "disabled"
        control_net_weight = 0.0
        control_net_weight2 = 0.0

    return {
        "image_mode": image_mode,
        "actual_video_length": actual_video_length,
        "actual_batch_size": actual_batch_size,
        "actual_guidance": actual_guidance,
        "video_guide": video_guide,
        "video_mask": video_mask,
        "video_prompt_type": video_prompt_type,
        "control_net_weight": control_net_weight,
        "control_net_weight2": control_net_weight2,
    }


def prepare_image_inputs(
    wgp_params: Dict[str, Any],
    *,
    is_qwen: bool,
    image_mode: int,
    load_image_fn: Callable,
) -> None:
    """Load PIL images for image_start/image_end/image_guide/image_mask (in-place).

    WGP expects PIL Image objects, not file paths.  This function converts
    string paths to loaded images and handles resolution matching.
    """
    # Extract target resolution for resizing
    target_width, target_height = None, None
    if wgp_params.get('resolution'):
        try:
            w_str, h_str = wgp_params['resolution'].split('x')
            target_width, target_height = int(w_str), int(h_str)
        except (ValueError, TypeError, AttributeError) as e_res:
            generation_logger.warning(
                f"[PREFLIGHT] Could not parse resolution '{wgp_params.get('resolution')}': {e_res}"
            )

    # Load image_start / image_end
    for img_param in ('image_start', 'image_end'):
        val = wgp_params.get(img_param)
        if not val:
            continue

        if isinstance(val, str):
            img = load_image_fn(val, mask=False)
            if img and target_width and target_height:
                from PIL import Image
                if img.size != (target_width, target_height):
                    generation_logger.debug_anomaly(
                        "PREFLIGHT",
                        f"resized {img_param} from {img.size[0]}x{img.size[1]} to {target_width}x{target_height}"
                    )
                    img = img.resize((target_width, target_height), Image.LANCZOS)
            wgp_params[img_param] = img

        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
            loaded_imgs = []
            for p in val:
                img = load_image_fn(p, mask=False)
                if img and target_width and target_height:
                    from PIL import Image
                    if img.size != (target_width, target_height):
                        generation_logger.debug_anomaly(
                            "PREFLIGHT",
                            f"resized {img_param} image from {img.size[0]}x{img.size[1]} to {target_width}x{target_height}"
                        )
                        img = img.resize((target_width, target_height), Image.LANCZOS)
                loaded_imgs.append(img)
            wgp_params[img_param] = loaded_imgs

    # For image-based models, load guide and mask images
    if is_qwen or image_mode == 1:
        if wgp_params.get('image_guide') and isinstance(wgp_params['image_guide'], str):
            wgp_params['image_guide'] = load_image_fn(wgp_params['image_guide'], mask=False)
        if wgp_params.get('image_mask') and isinstance(wgp_params['image_mask'], str):
            wgp_params['image_mask'] = load_image_fn(wgp_params['image_mask'], mask=True)

        # Ensure proper parameter coordination for Qwen models
        if is_qwen:
            if not wgp_params.get('image_mask'):
                wgp_params['image_mask'] = None
                generation_logger.debug_anomaly("PREFLIGHT", "set image_mask=None for Qwen regular generation")
            else:
                wgp_params['model_mode'] = 1
                generation_logger.debug_anomaly("PREFLIGHT", "set model_mode=1 for Qwen inpainting")

    # Sanitize image_refs: WGP expects None when there are no refs
    try:
        if isinstance(wgp_params.get('image_refs'), list) and len(wgp_params['image_refs']) == 0:
            wgp_params['image_refs'] = None
    except (TypeError, KeyError):
        pass
