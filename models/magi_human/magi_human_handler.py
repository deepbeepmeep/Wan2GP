import os
from typing import Any, Dict, Tuple

import torch

from shared.utils.hf import build_hf_url
from .prompt_enhancers import MAGI_HUMAN_ENHANCED_PROMPT


MAGI_HUMAN_REPO = "DeepBeepMeep/MagiHuman"
TEXT_ENCODER_FOLDER = "t5gemma-9b-9b-ul2"
TEXT_ENCODER_BF16 = "t5gemma-9b-9b-ul2_bf16.safetensors"
TEXT_ENCODER_QUANTO = "t5gemma-9b-9b-ul2_quanto_bf16_int8.safetensors"
DISTILL_ARCH = "magi_human_distill"
BASE_ARCH = "magi_human"
SR_MODEL_DEFAULTS = {
    "sr_cfg_number": 1,
    "sr_num_inference_steps": 5,
    "sr_noise_value": 220,
    "sr_video_txt_guidance_scale": 3.5,
    "use_cfg_trick": True,
    "cfg_trick_start_frame": 13,
    "cfg_trick_value": 2.0,
    "using_sde_flag": False,
    "sr_audio_noise_scale": 0.7,
}


class family_handler:
    @staticmethod
    def query_supported_types():
        return [BASE_ARCH, DISTILL_ARCH]

    @staticmethod
    def query_family_maps() -> Tuple[Dict[str, str], Dict[str, list]]:
        return {DISTILL_ARCH: BASE_ARCH}, {BASE_ARCH: [DISTILL_ARCH]}

    @staticmethod
    def query_model_family():
        return "magi_human"

    @staticmethod
    def query_family_infos():
        return {"magi_human": (62, "Magi Human")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-magi-human",
            type=str,
            default=None,
            help=f"Path to a directory that contains Magi Human LoRAs (default: {os.path.join(lora_root, 'magi_human')})",
        )
        parser.add_argument(
            "--lora-dir-magi-human-distill",
            type=str,
            default=None,
            help=f"Path to a directory that contains Magi Human Distill LoRAs (default: {os.path.join(lora_root, 'magi_human_distill')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if base_model_type == BASE_ARCH:
            return getattr(args, "lora_dir_magi_human", None) or os.path.join(lora_root, "magi_human")
        return getattr(args, "lora_dir_magi_human_distill", None) or os.path.join(lora_root, "magi_human_distill")

    @staticmethod
    def query_model_def(base_model_type: str, model_def: Dict[str, Any]):
        is_distill = base_model_type == DISTILL_ARCH
        extra_model_def = {
            "returns_audio": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_guide_label": "Driving Audio",
            "audio_guide_window_slicing": True,
            "audio_prompt_type_sources": {
                "selection": ["", "A"],
                "labels": {"": "Generate Video & Soundtrack based on Text Prompt", "A": "Generate Video based on Soundtrack and Text Prompt"},
                "show_label": False,
            },
            "multimedia_generation": True,
            "sample_solvers": [("UniPC", "unipc")],
            "audio_guidance": not is_distill,
            "guidance_max_phases": 0 if is_distill else 1,
            "lock_inference_steps": is_distill,
            "no_negative_prompt": is_distill,
            "profiles_dir": [base_model_type],
            "group": "magi_human",
            "fps": 25,
            "frames_minimum": 26,
            "latent_size": 4,
            "frames_steps": 4,
            "sliding_window": True,
            "sliding_window_defaults": {
                "overlap_min": 1,
                "overlap_max": 1,
                "overlap_step": 1,
                "overlap_default": 1,
                "window_min": 25,
                "window_max": 251,
                "window_step": 4,
                "window_default": 101,
            },
            "image_prompt_types_allowed": "SVL",
            "multiple_images_as_text_prompts": True,
            "multiple_submodels": False,
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(MAGI_HUMAN_REPO, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16),
                build_hf_url(MAGI_HUMAN_REPO, TEXT_ENCODER_FOLDER, TEXT_ENCODER_QUANTO),
            ],
            "text_prompt_enhancer_instructions": MAGI_HUMAN_ENHANCED_PROMPT,
            "video_prompt_enhancer_instructions": MAGI_HUMAN_ENHANCED_PROMPT,
            "config_file": f"models/magi_human/configs/{base_model_type}.json",
            "vae_block_size": 32,
            "guidance_max_phases": 1,
            "visible_phases": 0 if is_distill else 1,
        }
        extra_model_def.update(model_def)
        if "URLs2" in extra_model_def:
            for key, value in SR_MODEL_DEFAULTS.items():
                extra_model_def.setdefault(key, value)
            extra_model_def.update({
                "multiple_submodels": True,
                "guidance_max_phases": 2,
                "lock_guidance_phases": True,
            })
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": MAGI_HUMAN_REPO,
                "sourceFolderList": [TEXT_ENCODER_FOLDER, "stable-audio-open-1.0", "turbo_vae"],
                "fileList": [
                    ["config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
                    ["model_config.json", "model.safetensors"],
                    ["TurboV3-Wan22-TinyShallow_7_7.json", "TurboV3-Wan22-TinyShallow_7_7.safetensors"],
                ],
            },
            {
                "repoId": "DeepBeepMeep/Wan2.2",
                "sourceFolderList": [""],
                "fileList": [["Wan2.2_VAE.safetensors"]],
            },
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
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
        from .magi_human_model import MagiHumanModel

        magi_model = MagiHumanModel(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": magi_model.transformer,
            "text_encoder": magi_model.text_encoder.model,
            "vae": magi_model.vae.model,
            "audio_vae": magi_model.audio_vae.vae_model,
            "turbo_vae": magi_model.turbo_vae,
        }
        if magi_model.transformer2 is not None:
            pipe["transformer2"] = magi_model.transformer2
        return magi_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        inputs["sliding_window_overlap"] = 1
        if base_model_type != DISTILL_ARCH:
            return
        inputs["guidance_scale"] = 1.0
        inputs["audio_guidance_scale"] = 1.0
        inputs["num_inference_steps"] = 8

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "sample_solver": "unipc",
            "flow_shift": 5.0,
            "multi_prompts_gen_type": "FG",
            "image_prompt_type": "S",
            "audio_prompt_type": "",
            "video_length": 101,
            "sliding_window_size": 101,
            "sliding_window_overlap": 1,
            "sliding_window_discard_last_frames": 0,
        })
        if "URLs2" in model_def:
            ui_defaults["guidance_phases"] = 2
        if base_model_type == BASE_ARCH:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "audio_guidance_scale": 5.0,
                "num_inference_steps": 32,
            })
        else:
            ui_defaults.update({
                "guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "num_inference_steps": 8,
            })

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("wan", "ti2v_2_2")
