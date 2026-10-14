# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from models.help_shared import specialities

from shared.utils.hf import build_hf_url
from .prompts import K6_DEEPY_INFOS, K6_DEEPY_PROMPT_INFOS, K6_IMAGE_ENHANCER_INSTRUCTIONS, K6_INFOS, K6_PROMPT_INFOS, K6_TEXT_ENHANCER_INSTRUCTIONS

PRO_DISTILL_ARCH = "k6_pro_distill"
LITE_DISTILL_ARCH = "k6_lite_distill"
TEXT_ENCODER_FOLDER = "Qwen2.5-VL-7B-Instruct"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [PRO_DISTILL_ARCH, LITE_DISTILL_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "kandinsky6"

    @staticmethod
    def query_family_infos():
        return {"kandinsky6": (51, "Kandinsky 6")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("hunyuan")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "device_explicit": True,
            "cuda_graphs": ["transformer"],
            "dtype": "bf16",
            "tiny_vae_architecture": "hunyuan",
            "size": "large" if base_model_type == PRO_DISTILL_ARCH else "lighter",
            "accelerated": "native",
            "returns_audio": True,
            "multimedia_generation": True,
            "fps": 24,
            "frames_minimum": 5,
            "frames_steps": 4,
            "frames_selection_maximum": 241,
            "sliding_window": False,
            "sliding_window_defaults": {"window_default": 121},  # one generation: the trained 5 s length is the recommended maximum
            "image_prompt_types_allowed": "TS",
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "inference_steps": True,
            "profiles_dir": [base_model_type],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/Qwen_image", TEXT_ENCODER_FOLDER, "Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Qwen_image", TEXT_ENCODER_FOLDER, "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
            ],
            "infos": K6_INFOS,
            "specialities": specialities("audio-video generation"),
            "prompt_infos": K6_PROMPT_INFOS,
            "deepy_infos": K6_DEEPY_INFOS,
            "deepy_prompt_infos": K6_DEEPY_PROMPT_INFOS,
            "prompt_enhancer_video_duration": True,
            "text_prompt_enhancer_instructions": K6_TEXT_ENHANCER_INSTRUCTIONS,
            "video_prompt_enhancer_instructions": K6_IMAGE_ENHANCER_INSTRUCTIONS,
            "text_prompt_enhancer_max_tokens": 1024,
            "video_prompt_enhancer_max_tokens": 1024,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": "DeepBeepMeep/Qwen_image",
                "sourceFolderList": [TEXT_ENCODER_FOLDER],
                "fileList": [["config.json", "merges.txt", "tokenizer_config.json", "vocab.json"]],
            },
            {
                "repoId": "DeepBeepMeep/HunyuanVideo",
                "sourceFolderList": ["clip_vit_large_patch14", ""],
                "fileList": [
                    ["text_config.json", "merges.txt", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                    ["hunyuan_video_VAE_fp32.safetensors", "hunyuan_video_VAE_config.json"],
                ],
            },
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": ["mmaudio", "bigvgan_v2_44khz_128band_512x"],
                "fileList": [["v1-44.pth"], ["config.json", "bigvgan_generator.pt"]],
            },
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False, text_encoder_quantization=None, dtype=torch.bfloat16,
                   VAE_dtype=torch.float32, mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None, text_encoder_filename=None, **kwargs):
        from .kandinsky6_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=save_quantized)
        pipe = {
            "transformer": model.transformer,
            "text_encoder": model.text_encoder.model,
            "text_encoder_2": model.text_encoder.clip,
            "vae": model.vae,
            "audio_vae": model.audio_vae,
        }
        return model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 10,
            "guidance_scale": 1.0,
            "video_length": 121,
            "resolution": "832x480",
            "image_prompt_type": "",
        })
