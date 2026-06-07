import os

import torch

from shared.utils.hf import build_hf_url
from .prompt_enhancer import IDEOGRAM4_PROMPT_ENHANCER, IDEOGRAM4_PROMPT_INFOS


_PROJECT_REPO = "DeepBeepMeep/Ideogram4"
_PROJECT_FOLDER = "ideogram4"
_TEXT_ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"
_VAE_REPO = "DeepBeepMeep/Flux2"
_VAE_FILENAME = "flux2_vae.safetensors"
_PRESET_CHOICES = [
    ("Quality 48", "V4_QUALITY_48"),
    ("Default 20", "V4_DEFAULT_20"),
    ("Turbo 12", "V4_TURBO_12"),
]
_DEFAULT_PRESET = "V4_DEFAULT_20"


def _model_uses_nf4(model_def):
    urls = []
    if isinstance(model_def, dict):
        urls.extend(model_def.get("URLs", []) or [])
        urls.extend(model_def.get("URLs2", []) or [])
    return any("_nf4" in str(url).lower() for url in urls)


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        text_encoder_filename = "Qwen3-VL-8B-Instruct_nf4.safetensors" if _model_uses_nf4(model_def) else "Qwen3-VL-8B-Instruct_fp8.safetensors"
        return {
            "image_outputs": True,
            "flux2": True,
            "pid_upsampler": [1],
            "pid_vae_backbone": "flux2",
            "guidance_max_phases": 0,
            "inference_steps": False,
            "lock_inference_steps": True,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [base_model_type],
            "no_negative_prompt": True,
            "no_background_removal": True,
            "skip_prompt_template": True,
            "vae_block_size": 16,
            "prompt_enhancer_button_label": "Magic Prompt",
            "prompt_infos": IDEOGRAM4_PROMPT_INFOS,
            "prompt_helper_popup_dims": [78, 100],
            "prompt_enhancer_def": {
                "selection": ["T"],
                "labels": {"T": "Ideogram JSON caption from Text Prompt"},
                "default": "",
            },
            "text_prompt_enhancer_instructions": IDEOGRAM4_PROMPT_ENHANCER,
            "image_prompt_enhancer_instructions": IDEOGRAM4_PROMPT_ENHANCER,
            "text_prompt_enhancer_max_tokens": 2048,
            "image_prompt_enhancer_max_tokens": 2048,
            "model_modes": {
                "choices": _PRESET_CHOICES,
                "default": _DEFAULT_PRESET,
                "label": "Preset",
                "image_modes": [1],
            },
            "text_encoder_folder": _TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(_PROJECT_REPO, _TEXT_ENCODER_FOLDER, text_encoder_filename),
            ],
        }

    @staticmethod
    def query_supported_types():
        return ["ideogram4"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "ideogram4"

    @staticmethod
    def query_family_infos():
        return {"ideogram4": (140, "Ideogram")}

    @staticmethod
    def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
        from .prompt_helper import render_prompt_helper

        return render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id)

    @staticmethod
    def get_prompt_helper_css():
        from .prompt_helper import get_prompt_helper_css

        return get_prompt_helper_css()

    @staticmethod
    def get_prompt_helper_javascript():
        from .prompt_helper import get_prompt_helper_javascript

        return get_prompt_helper_javascript()

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("flux", sub_family="flux2")

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-ideogram4",
            type=str,
            default=None,
            help=f"Path to a directory that contains Ideogram 4 LoRAs (default: {os.path.join(lora_root, 'ideogram4')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_ideogram4", None) or os.path.join(lora_root, "ideogram4")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": _VAE_REPO,
                "sourceFolderList": [""],
                "fileList": [[_VAE_FILENAME]],
            },
            {
                "repoId": _PROJECT_REPO,
                "sourceFolderList": [_PROJECT_FOLDER, _TEXT_ENCODER_FOLDER],
                "fileList": [
                    ["ideogram4_transformer_config.json", "ideogram4_unconditional_transformer_config.json"],
                    ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
                ],
            }
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
        **kwargs,
    ):
        from .ideogram4_main import model_factory

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": pipe_processor.conditional_transformer,
            "transformer2": pipe_processor.unconditional_transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.autoencoder,
        }
        pipe = {
            "pipe": pipe,
            "coTenantsMap": {
                "transformer": ["transformer2"],
                "transformer2": ["transformer"],
            },
        }
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "model_mode": _DEFAULT_PRESET,
            "batch_size": 1,
        })

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("model_mode", None) not in {value for _, value in _PRESET_CHOICES}:
            old_solver = ui_defaults.get("sample_solver", None)
            ui_defaults["model_mode"] = old_solver if old_solver in {value for _, value in _PRESET_CHOICES} else _DEFAULT_PRESET
        ui_defaults.setdefault("image_mode", 1)
