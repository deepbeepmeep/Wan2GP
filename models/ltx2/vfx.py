"""LTX-2.5 VFX recipes, using the shared LTX pipeline and RGBA exporter.

Recipes: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen
and https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render.
"""

import torch


ALPHA_INFOS = """### Alpha Gen
Generate a soft grayscale matte from a source video: white is opaque, black is transparent.
Upload the RGB clip as **Source Video**. Its original dimensions are preserved; dimensions are
padded internally to multiples of 32, then cropped back. Keep clips at **145 frames or fewer**
(121 is a good starting point), up to 1920×1088, with a frame count of 8n+1. The selected clip is
trimmed to a supported frame count. Resize larger footage before loading it.

**Whole Frame** needs no mask. **Masked Area** optionally restricts the result to your uploaded
or Magic Mask selection; **Non Masked Area** uses the inverse. The selection multiplies the final
matte, preserving soft edges. It does not change which foreground the model detects.

The gallery video is the grayscale matte. The original RGB clip with that alpha is saved beside
it as RGBA PNG frames in a ZIP or ProRes 4444 MOV, chosen in **Configuration → Outputs → RGBA
Video Output**. Alpha Gen uses one phase, eight steps and no text conditioning.
"""
ALPHA_PROMPT_INFOS = """Alpha Gen does not use a text prompt. Keep the default **Generate Alpha Matte**
operation label; it is ignored during inference. Use the optional selection mask to restrict the
finished matte. Text cannot select a subject or create a foreground missed by the model."""
LAYOUT_INFOS = """### Layout to Render
Turn a clay viewport animation or a simple blockout/playblast into a finished shot.
Upload the animation as **Layout Video** and a finished version of its first frame as the
**Appearance Reference**. The clip supplies camera motion, composition and object placement;
the appearance image supplies lighting, palette, materials and detail. Do not use the clay
frame itself as the appearance image. An optional End Image can help maintain the look.

Start with two phases, eight first-pass steps, LoRA strength 1 in both phases and 24 fps.
1920×1088 output uses a 960×544 first pass and three full-resolution refinement steps.
Dimensions must be multiples of 64; clip lengths use 8n+1 frames.
**Two Phases with Tiling** evaluates overlapping spatial crops during each second-pass step
and blends their predictions before advancing the whole video. It reduces the spatial extent
of each transformer call and can change fine details. VAE tiling remains a separate setting.
"""
LAYOUT_PROMPT_INFOS = """Describe the finished shot in one or two sentences, matching the appearance
reference: materials, lighting, atmosphere and subject details. Avoid describing it as clay,
3D or Unreal footage. Example: “A weathered stone well surrounded by summer grass and small
white flowers. Warm midday sunlight, realistic stone and wood, matching the reference image.”
The layout clip already supplies camera movement and object placement."""


def vfx_model_def(model_def):
    alpha = model_def.get("ltx2_alpha_gen", False)
    layout = model_def.get("ltx2_layout_to_render", False)
    if not (alpha or layout):
        return {}
    result = {
        "accelerated": "native",
        "sliding_window": False, "video_guide_outpainting": [], "custom_settings": [],
        "v2i_switch_supported": False, "multiple_images_as_text_prompts": False,
        "image_prompt_types_allowed": "T" if alpha else "TE", "custom_frames_injection": False,
        "control_video_trim": True, "extra_control_frames": 0,
        "guide_custom_choices": {"choices": [("Source Video" if alpha else "Layout Video", "VG")], "letters_filter": "OPDEMVG&KF", "default": "VG", "label": "Video Input"},
        "guide_custom_choices_image": None,
        "mask_preprocessing": {"selection": ["", "A", "NA"] if alpha else [""], "visible": alpha, "label": "Alpha Selection"},
        "self_refiner": False, "NAG": False,
        "capability_overrides": {"text_to_video": False},
    }
    if alpha:
        result.update({
            "infos": ALPHA_INFOS, "prompt_infos": ALPHA_PROMPT_INFOS,
            "deepy_infos": "Alpha Gen: source RGB video → grayscale matte plus RGBA ZIP/ProRes companion. Native source dimensions, ≤1920×1088, ≤145 frames, 8n+1. One phase, 8 steps. Optional mask restricts the final alpha; source RGB stays unchanged.",
            "deepy_prompt_infos": ALPHA_PROMPT_INFOS,
            "specialities": [{"name": "alpha output", "aliases": ["RGBA", "alpha matte", "video matting", "transparent video"], "description": "Extract a grayscale matte from source video and export original RGB with alpha as PNG ZIP or ProRes 4444. An optional selection mask restricts the final matte."}],
            "returns_audio": False, "multimedia_generation": False, "auto_null_audio": False,
            "any_audio_prompt": False, "audio_prompt_choices": None,
            "audio_prompt_type_sources": None, "output_audio_is_input_audio": False,
            "end_frames_always_enabled": False,
            "capability_overrides": {"text_to_video": False, "inpainting": False},
            "prompt_enhancer_def": {"selection": [], "default": ""},
            "guidance_max_phases": 1, "lock_guidance_phases": True, "phase_2_spatial_tiling": False,
            "vae_block_size": 32, "frames_minimum": 9, "frames_selection_maximum": 145,
            "custom_preprocessor": "Preparing Alpha Source and Selection", "custom_preprocessor_raw_inputs": True,
        })
    else:
        result.update({
            "infos": LAYOUT_INFOS, "prompt_infos": LAYOUT_PROMPT_INFOS,
            "deepy_infos": "Layout to Render: provide a layout/playblast video plus one finished appearance image based on its first frame. Optional End Image. Two phases (8+3 steps), LoRA 1; optional phase-2 spatial tiling. 24 fps, 8n+1 frames, dimensions divisible by 64.",
            "deepy_prompt_infos": LAYOUT_PROMPT_INFOS,
            "image_ref_choices": {"choices": [("Appearance Reference", "I")], "letters_filter": "I", "default": "I", "label": "Appearance Reference", "show_label": True},
            "one_image_ref_needed": True, "one_image_ref_only": True, "no_background_removal": True,
            "phase_2_spatial_tiling": True,
        })
    return result


def preprocess_alpha_source(video, mask, video_prompt_type):
    """Raw THWC bytes from the shared reader; do not erase RGB outside the selection."""
    source = video.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    selection = None
    if mask is not None:
        if mask.shape[1:3] != video.shape[1:3]:
            raise ValueError("Alpha selection mask must have the same dimensions as the source video.")
        if mask.shape[0] not in (1, video.shape[0]):
            raise ValueError("Alpha selection mask must be one frame or match the source clip length.")
        selection = mask[..., :3].float().mean(dim=-1).unsqueeze(0).div_(255.0)
        if selection.shape[1] == 1:
            selection = selection.expand(-1, source.shape[1], -1, -1)
        if "N" in video_prompt_type:
            selection = 1.0 - selection
    return source, None, selection, None


def alpha_result(matte, source, selection, fps, output_format, interrupt_check):
    from shared.utils.rgba_video import rgba_video_side_files

    alpha = matte.float().mean(dim=0, keepdim=True)
    alpha = alpha.div_(255.0) if matte.dtype == torch.uint8 else alpha.add_(1.0).mul_(0.5)
    alpha.clamp_(0.0, 1.0)
    if selection is not None:
        alpha.mul_(selection[:, :alpha.shape[1]].to(alpha))
    alpha = alpha.mul_(255.0).round_().to(device="cpu", dtype=torch.uint8)
    rgb = source[:, :alpha.shape[1]].to(device="cpu", dtype=torch.float32).add(1).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    bgra = torch.cat((rgb[[2, 1, 0]], alpha), dim=0).permute(1, 2, 3, 0).contiguous().numpy()
    side_files = rgba_video_side_files(bgra, fps, output_format, interrupt_check=interrupt_check)
    if side_files is None:
        return None
    return {"x": alpha.expand(3, -1, -1, -1).contiguous(), "side_files": side_files}
