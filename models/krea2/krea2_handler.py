import os

import gradio as gr
import torch

from shared.utils.hf import build_hf_url


_PROJECT_REPO = "DeepBeepMeep/krea-2"
_QWEN_IMAGE_REPO = "DeepBeepMeep/Qwen_image"
_WAN_REPO = "DeepBeepMeep/Wan2.1"
_VAE_UPSAMPLER_FILENAME = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"
_TEXT_ENCODER_FOLDER = "Qwen3-VL-4B-Instruct"
_TEXT_ENCODER_BF16_FILENAME = "Qwen3-VL-4B-Instruct_text_bf16.safetensors"
_TEXT_ENCODER_INT8_FILENAME = "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors"
_VISION_ENCODER_FILENAME = "Qwen3-VL-4B-Instruct_vision_bf16.safetensors"
_RAW_MODEL_TYPE = "krea2_raw"
_TURBO_MODEL_TYPE = "krea2_turbo"
_RAW_EDIT_MODEL_TYPE = "krea2_raw_edit"
_TURBO_EDIT_MODEL_TYPE = "krea2_turbo_edit"
_PROFILE_DIR = "krea2"
_PRESET_PROFILE_DIR = "krea2_presets"

class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        edit = base_model_type in (_RAW_EDIT_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE)
        lanpaint_choices = [
            ("LanPaint (2 steps): ~2x slower, easy task", 2),
            ("LanPaint (5 steps): ~5x slower, medium task", 3),
            ("LanPaint (10 steps): ~10x slower, hard task", 4),
            ("LanPaint (15 steps): ~15x slower, very hard task", 5),
        ]
        result = {
            **({"accelerated": "native"} if base_model_type in (_TURBO_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE) else {}),
            "image_outputs": True,
            "guidance_max_phases": 1 if base_model_type in (_RAW_MODEL_TYPE, _RAW_EDIT_MODEL_TYPE) else 0,
            "NAG": True,
            "NAG_scale": {"min": 1.0, "max": 1.5, "step": 0.01},
            "NAG_tau": {"min": 1.0, "max": 5.0, "step": 0.05},
            "NAG_alpha": {"min": 0.0, "max": 1.0, "step": 0.01},
            "inference_steps": True,
            "inpaint_support": True,
            "inpaint_video_prompt_type": "VA", # "VAG",
            "inpaint_color": "FFFFFF",
            "guide_custom_choices_image": {
                "choices": [("No Control Image", ""), ("Control Image", "V"), ("Control Image with Masked Denoising", "VG")],
                "letters_filter": "V", # "VG",
                "default": "",
                "label": "Control Image",
                "visible": False,
            },
            "model_modes": {
                # "choices": [("Masked Denoising : Inpainted area may reuse some content that has been masked", 0)] + lanpaint_choices,
                "choices": lanpaint_choices,
                "default": 2, #0,
                "label": "Inpainting Method",
                "image_modes": [2],
            },
            "fit_into_canvas_image_refs": 0,
            "preset_profiles_dir": [_PRESET_PROFILE_DIR],
            "profiles_dir": [_PROFILE_DIR],
            "text_encoder_folder": _TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(_PROJECT_REPO, _TEXT_ENCODER_FOLDER, _TEXT_ENCODER_BF16_FILENAME),
                build_hf_url(_PROJECT_REPO, _TEXT_ENCODER_FOLDER, _TEXT_ENCODER_INT8_FILENAME),
            ],
            "no_negative_prompt": False,
            "no_background_removal": True,
            "resolutions_categories": ["<=2k"],
            "vae_block_size": 16,
            "vae_upsampler": [1],
            "vae_upsamplers": {"qwen_vae_pid(1.5)": [1]},
            "excluded_spatial_upsamplers": ["qwen_pid(1.5)"],
        }
        if not edit:
            result.update({
                "deepy_infos": "Text-to-image from `prompt`. Inpainting combines a Control Image, mask and prompt through the selected LanPaint method; the retained image supplies context.",
                "deepy_prompt_infos": "Describe the finished image in natural language: subject/action, framing, setting, lighting and style. Add detail for precise composition and quote exact lettering. For inpainting, describe the desired content in the masked area within the finished scene.",
                "infos": "Generate an image from the Text Prompt (`prompt`). In Inpainting mode, a Control Image and mask provide the surrounding context while the prompt describes the replacement content; WanGP uses the selected LanPaint method. The text, retained context and masked region jointly determine the result.",
                "prompt_infos": 'Describe the image you want in natural language, including subject, pose or action, framing, environment, lighting and style. Start with a short concrete idea; add detail when precise composition matters. Put literal lettering in quotes. For inpainting, describe the finished scene and the desired content inside the mask, for example: "A man in a green jacket sits at the cafe table, warm evening light."',
            })
        if edit:
            result.update({
                "specialities": [{"name": "identity-preserving edits", "aliases": ["identity preservation"]}, {"name": "subject placement"}],
                "infos": "Edit images with the built-in Identity Edit LoRA by combining the Text Prompt (`prompt`) with Reference Images (`image_refs`). One reference supplies the image to edit; with two, use the scene first and the person or object second, and identify their roles in the prompt. Reference mode `KI` keeps the first image as the main scene; `I` treats references as people or objects. Text-only generation is also available.\n\nInpainting combines a Control Image and mask with the prompt; choose Masked Denoising or a LanPaint method. A Control Image is placed before other references, so it becomes image 1 in the prompt. Outpainting extends the source canvas using the selected margins. Enable background removal only for references whose surrounding scene should be discarded.",
                "prompt_infos": 'Use a short, direct editing instruction and name what must remain unchanged. For example: "Change the car to matte black; keep its shape, camera angle, lighting and surroundings." For two-image composition: "Place the person from image 2 at the table in image 1; preserve their face and clothing, and match the scene lighting." Image numbers follow input order, including a Control Image placed first.\n\nState identity, pose, clothing or background constraints explicitly when they matter. For inpainting, identify the replacement inside the mask; for outpainting, describe the scene continuing into the new area. Quote exact visible text. For text-only generation, describe the finished image instead of referring to a source.',
                "deepy_infos": "Identity Edit combines `prompt` and ordered `image_refs`: one source, or scene first + person/object second. `KI` keeps the first image as the main scene; `I` uses people/objects. A Control Image is prepended as image 1. Inpainting uses source + mask with Masked Denoising or LanPaint; outpainting extends the canvas. Text-only generation is supported.",
                "deepy_prompt_infos": "Give a direct edit instruction and preservation constraints: 'Place the person from image 2 at the table in image 1; keep their face/clothing and match scene lighting.' Number inputs in order, including any Control Image first. Describe masked replacements or continuation into outpainted margins; quote exact lettering. Text-only: describe the finished image.",
                "inpaint_support": True,
                "inpaint_video_prompt_type": "VAG",
                "image_ref_choices": {
                    "choices": [
                        ("None",""),
                        ("Conditional Image is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                        ("Conditional Images are People / Objects", "I"),
                    ],
                    "letters_filter": "KI",
                    "default": "KI",
                },
                "at_least_one_image_ref_needed": False,
                "no_background_removal": False,
                "background_removal_label": "Remove Backgrounds only behind People / Objects except main Subject / Landscape",
                "video_guide_outpainting": [1, 2],
                "outpainting_quantize_margins": 16,
                "vision_encoder_filename": os.path.join(_TEXT_ENCODER_FOLDER, _VISION_ENCODER_FILENAME),
                "preload_URLs": [
                    build_hf_url(_PROJECT_REPO, _TEXT_ENCODER_FOLDER, _VISION_ENCODER_FILENAME)
                    + f"|{_TEXT_ENCODER_FOLDER}"
                ],
                "model_modes": {
                    "choices": [("Masked Denoising: inpainted area may reuse masked content", 0)] + lanpaint_choices,
                    "default": 0,
                    "label": "Inpainting Method",
                    "image_modes": [2],
                },
            })
        return result

    @staticmethod
    def query_supported_types():
        return [_RAW_MODEL_TYPE, _TURBO_MODEL_TYPE, _RAW_EDIT_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        compatible = [_RAW_MODEL_TYPE, _TURBO_MODEL_TYPE, _RAW_EDIT_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE]
        return {}, {model_type: compatible for model_type in compatible}

    @staticmethod
    def query_model_family():
        return "krea2"

    @staticmethod
    def query_family_infos():
        return {"krea2": (1150, "Krea 2")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument("--lora-dir-krea2", type=str, default=None, help=f"Path to a directory that contains Krea 2 LoRAs (default: {os.path.join(lora_root, 'krea2')})")

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_krea2", None) or os.path.join(lora_root, "krea2")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": _PROJECT_REPO,
                "sourceFolderList": [_TEXT_ENCODER_FOLDER],
                "fileList": [
                    ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json"],
                ],
            },
            {
                "repoId": _QWEN_IMAGE_REPO,
                "sourceFolderList": [""],
                "fileList": [["qwen_vae.safetensors", "qwen_vae_config.json"]],
            },
            {
                "repoId": _WAN_REPO,
                "sourceFolderList": [""],
                "fileList": [[_VAE_UPSAMPLER_FILENAME]],
            },
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        VAE_upsampling=None,
        **kwargs,
    ):
        from .krea2_main import model_factory

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            VAE_upsampling=VAE_upsampling,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder.language_model,
            "vae": pipe_processor.vae,
        }
        if hasattr(pipe_processor.text_encoder, "visual"):
            pipe["vision_encoder"] = pipe_processor.text_encoder.visual
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        edit = base_model_type in (_RAW_EDIT_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE)
        ui_defaults.update({"image_mode": 1, "batch_size": 1, "model_mode": 0 if edit else 2, "denoising_strength": 1.0, "masking_strength": 1.0})
        if base_model_type in (_TURBO_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE):
            ui_defaults.update({"num_inference_steps": 8, "guidance_scale": 0, "resolution": "1024x1024"})
        else:
            ui_defaults.update({"num_inference_steps": 20 if base_model_type == _RAW_EDIT_MODEL_TYPE else 52, "guidance_scale": 2 if base_model_type == _RAW_EDIT_MODEL_TYPE else 3.5, "resolution": "1024x1024"})
        if edit:
            ui_defaults.update({"video_prompt_type": "KI", "remove_background_images_ref": 0})

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("image_mode", 1)
        if settings_version < 2.66:
            ui_defaults["denoising_strength"] = 1.0
            ui_defaults["masking_strength"] = 1.0
        if settings_version < 2.66 and ui_defaults["image_mode"] == 2:
            ui_defaults["video_prompt_type"] = model_def["inpaint_video_prompt_type"]

    @staticmethod
    def normalize_lanpaint_strengths(inputs):
        model_mode = inputs.get("model_mode")
        model_mode_int = None
        if model_mode is not None:
            try:
                model_mode_int = int(model_mode)
            except (TypeError, ValueError):
                model_mode_int = None
        if model_mode_int in (2, 3, 4, 5):
            inputs["denoising_strength"] = 1.0
            inputs["masking_strength"] = 1.0
        return model_mode_int

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, prompt):
        family_handler.normalize_lanpaint_strengths(inputs)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type in (_RAW_EDIT_MODEL_TYPE, _TURBO_EDIT_MODEL_TYPE):
            max_refs = 1 if inputs.get("image_mode") == 2 else 2
            if len(inputs.get("image_refs") or []) > max_refs:
                return "Krea 2 Edit supports at most two Reference Images."
        model_mode_int = family_handler.normalize_lanpaint_strengths(inputs)
        if inputs.get("denoising_strength", 1) < 1 and model_mode_int != 0:
            gr.Info("Denoising Strength will be ignored if Masked Denoising is not used")

    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("qwen")
