import math

import torch
import torch.nn.functional as F

from .ltx2_runtime import LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE, LTX2_MASKED_CONTROL_VIDEO_PAD_RGB


def _normalize_outpainting_dims(outpainting_dims) -> list[float] | None:
    if outpainting_dims is None:
        return None
    if isinstance(outpainting_dims, str):
        outpainting_dims = outpainting_dims.strip()
        if not outpainting_dims or outpainting_dims.startswith("#"):
            return None
        outpainting_dims = outpainting_dims.split()
    if not isinstance(outpainting_dims, (list, tuple)) or len(outpainting_dims) != 4:
        return None
    dims = [max(0.0, float(v)) for v in outpainting_dims]
    return dims if any(dims) else None


def _get_outpainting_inner_rect(height: int, width: int, outpainting_dims) -> tuple[int, int, int, int] | None:
    dims = _normalize_outpainting_dims(outpainting_dims)
    if dims is None or height <= 0 or width <= 0:
        return None
    from shared.utils.utils import get_outpainting_frame_location

    inner_height, inner_width, margin_top, margin_left = get_outpainting_frame_location(int(height), int(width), dims, 1)
    top = max(0, min(int(margin_top), int(height)))
    left = max(0, min(int(margin_left), int(width)))
    bottom = max(top, min(top + int(inner_height), int(height)))
    right = max(left, min(left + int(inner_width), int(width)))
    return (top, bottom, left, right) if bottom > top and right > left else None


def _ltx2_inpainting_enabled(video_prompt_type: str) -> bool:
    return "M" in (video_prompt_type or "") and "A" in (video_prompt_type or "")


def _build_outpainting_mask_cthw(video_tensor: torch.Tensor | None, outpainting_dims) -> torch.Tensor | None:
    if video_tensor is None or not torch.is_tensor(video_tensor) or video_tensor.dim() != 4:
        return None
    rect = _get_outpainting_inner_rect(video_tensor.shape[-2], video_tensor.shape[-1], outpainting_dims)
    if rect is None:
        return None
    mask = torch.ones((1, video_tensor.shape[1], video_tensor.shape[-2], video_tensor.shape[-1]), dtype=torch.float32, device=video_tensor.device)
    top, bottom, left, right = rect
    mask[:, :, top:bottom, left:right] = 0.0
    return mask


def _merge_ltx2_masks(mask: torch.Tensor | None, extra_mask: torch.Tensor | None) -> torch.Tensor | None:
    if extra_mask is None:
        return mask
    if mask is None:
        return extra_mask
    extra_mask = extra_mask.to(device=mask.device, dtype=torch.float32)
    return torch.maximum(mask.float(), extra_mask).to(dtype=mask.dtype)


def _apply_ltx2_inpaint_preprocess_dilation(video: torch.Tensor | None, mask: torch.Tensor | None, spatial_radius: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mask is None or spatial_radius <= 0:
        return video, mask
    dilated_mask = F.max_pool2d(mask.float().clamp(0.0, 1.0).permute(1, 0, 2, 3), kernel_size=spatial_radius * 2 + 1, stride=1, padding=spatial_radius).permute(1, 0, 2, 3).clamp(0.0, 1.0).to(device=mask.device, dtype=mask.dtype)
    if video is None:
        return video, dilated_mask
    color = torch.tensor(LTX2_MASKED_CONTROL_VIDEO_PAD_RGB, device=video.device, dtype=torch.float32)
    color = color.to(dtype=torch.uint8) if video.dtype == torch.uint8 else color.div(127.5).sub(1.0).to(dtype=video.dtype)
    video = torch.where(dilated_mask.to(device=video.device) > 0, color.view(3, 1, 1, 1), video)
    return video, dilated_mask


def _pad_ltx2_masked_control_video_tail(video: torch.Tensor | None, mask: torch.Tensor | None, target_frames: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if video is None:
        return video, mask
    target_frames = int(target_frames)
    video_frames = int(video.shape[1])
    if video_frames < target_frames:
        pad_frames = target_frames - video_frames
        color = torch.tensor(LTX2_MASKED_CONTROL_VIDEO_PAD_RGB, device=video.device, dtype=torch.float32)
        if video.dtype == torch.uint8:
            color = color.to(dtype=torch.uint8)
        else:
            color = color.div_(127.5).sub_(1.0).to(dtype=video.dtype)
        pad = color.view(3, 1, 1, 1).expand(3, pad_frames, video.shape[-2], video.shape[-1])
        video = torch.cat([video, pad], dim=1)
    if mask is not None and int(mask.shape[1]) < int(video.shape[1]):
        pad = torch.ones((mask.shape[0], int(video.shape[1]) - int(mask.shape[1]), mask.shape[-2], mask.shape[-1]), device=mask.device, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=1)
    return video, mask


def _to_unit_video(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0).clamp_(0.0, 1.0)
    return tensor.float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _resize_preserving_aspect_ratio(tensor: torch.Tensor, long_side: int, mode: str) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    current_long_side = max(height, width)
    if current_long_side == long_side:
        return tensor
    scale = long_side / current_long_side
    size = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


def _apply_low_res_mask_dilation(mask: torch.Tensor, spatial_radius: int, long_side: int = LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE) -> torch.Tensor:
    if spatial_radius <= 0:
        return mask
    original_size = mask.shape[-2:]
    mask_low_res = _resize_preserving_aspect_ratio(mask.float(), long_side, "bilinear")
    mask_low_res = F.max_pool2d(mask_low_res, kernel_size=spatial_radius * 2 + 1, stride=1, padding=spatial_radius)
    return F.interpolate(mask_low_res, size=original_size, mode="bilinear", align_corners=False)


def _laplacian_pyramid_blend(generated: torch.Tensor, source: torch.Tensor, mask: torch.Tensor, levels: int = 7, mask_low_res_dilation: int = 0) -> torch.Tensor:
    generated, source, mask = generated.cpu(), source.cpu(), mask.cpu()
    generated = _to_unit_video(generated).permute(1, 0, 2, 3)
    source = _to_unit_video(source).permute(1, 0, 2, 3)
    mask = mask.float().clamp_(0.0, 1.0).permute(1, 0, 2, 3)
    mask = _apply_low_res_mask_dilation(mask, mask_low_res_dilation).clamp_(0.0, 1.0)
    levels = max(1, min(int(levels), int(math.log2(max(2, min(generated.shape[-2:])))) - 2))

    def gaussian_pyramid(x):
        pyramid = [x]
        for _ in range(1, levels):
            if min(pyramid[-1].shape[-2:]) <= 8:
                break
            pyramid.append(F.interpolate(pyramid[-1], scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False))
        return pyramid

    def laplacian_pyramid(x):
        gaussian = gaussian_pyramid(x)
        laps = [cur - F.interpolate(nxt, size=cur.shape[-2:], mode="bilinear", align_corners=False) for cur, nxt in zip(gaussian[:-1], gaussian[1:])]
        laps.append(gaussian[-1])
        return laps

    generated_pyr = laplacian_pyramid(generated)
    source_pyr = laplacian_pyramid(source)
    mask_pyr = gaussian_pyramid(mask)
    blended = [gen * m + src * (1.0 - m) for gen, src, m in zip(generated_pyr, source_pyr, mask_pyr)]
    result = blended[-1]
    for level in reversed(blended[:-1]):
        result = F.interpolate(result, size=level.shape[-2:], mode="bilinear", align_corners=False) + level
    return result.clamp_(0.0, 1.0).permute(1, 0, 2, 3)


def _apply_ltx2_mask_blend(video_tensor: torch.Tensor, source: torch.Tensor | None, mask: torch.Tensor | None, output_frame_num: int, height: int, width: int, mask_low_res_dilation: int = 0, sanitize_masked_source: bool = False) -> torch.Tensor:
    if source is None or mask is None:
        return video_tensor
    frames = min(int(output_frame_num), int(video_tensor.shape[1]), int(source.shape[1]), int(mask.shape[1]))
    if frames <= 0:
        return video_tensor
    source = source.detach().cpu()[:, :frames, :height, :width].contiguous()
    mask = mask.detach().cpu()[:1, :frames, :height, :width].contiguous()
    generated = video_tensor[:, :frames, :height, :width]
    if sanitize_masked_source:
        generated_source = generated.detach().cpu()
        if source.dtype == torch.uint8 and generated_source.dtype != torch.uint8:
            generated_source = _to_unit_video(generated_source.clone()).mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8)
        elif source.dtype != torch.uint8 and generated_source.dtype == torch.uint8:
            generated_source = generated_source.float().div_(127.5).sub_(1.0).to(dtype=source.dtype)
        else:
            generated_source = generated_source.to(dtype=source.dtype)
        source = torch.where(mask > 0, generated_source, source)
    blended = _laplacian_pyramid_blend(generated, source, mask, mask_low_res_dilation=mask_low_res_dilation)
    if video_tensor.dtype == torch.uint8:
        blended = blended.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8)
    else:
        blended = blended.mul_(2.0).sub_(1.0).to(dtype=video_tensor.dtype)
    if frames == video_tensor.shape[1]:
        return blended
    video_tensor = video_tensor.clone()
    video_tensor[:, :frames, :height, :width] = blended
    return video_tensor
