import os

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import HEARTMULA_LYRIC_PROMPT

HEARTMULA_VERSION = "3B"


def _get_heartmula_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 1,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["heartmula_oss_3b"],
        "alt_prompt": {
            "label": "Keywords / Tags",
            "placeholder": "piano,happy,wedding",
            "lines": 2,
        },
        "text_prompt_enhancer_instructions": HEARTMULA_LYRIC_PROMPT,
        "duration_slider": {
            "label": "Duration of the Song (in seconds)",
            "min": 30,
            "max": 240,
            "increment": 0.1,
            "default": 120,
        },
        "top_k_slider": True,
        "heartmula_cfg_scale": 1.5,
        "heartmula_topk": 50,
        "heartmula_max_audio_length_ms": 120000,
        "heartmula_codec_guidance_scale": 1.25,
        "heartmula_codec_steps": 10,
        "heartmula_codec_version": "",
        "compile": False,  # ["transformer", "transformer2"]
    }


def _get_heartmula_download_def(model_def):
    codec_version = (model_def or {}).get("heartmula_codec_version", "")
    codec_suffix = f"_{codec_version}" if codec_version else ""
    repo_id = "DeepBeepMeep/TTS"
    gen_files = [
        "gen_config.json",
        "tokenizer.json",
        f"codec_config{codec_suffix}.json",
        f"HeartMula_codec{codec_suffix}.safetensors",
    ]
    return [
        {
            "repoId": repo_id,
            "sourceFolderList": ["HeartMula"],
            "fileList": [gen_files],
        },
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["heartmula_oss_3b"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (200, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-heart_mula",
            type=str,
            default=None,
            help=f"Path to a directory that contains Heart Mula settings (default: {os.path.join(lora_root, 'heart_mula')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_heart_mula", None) or os.path.join(lora_root, "heart_mula")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_heartmula_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_heartmula_download_def(model_def or {})

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None,
        VAE_dtype=None,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        profile=0,
        **kwargs,
    ):
        from .HeartMula.pipeline import HeartMuLaPipeline

        ckpt_root = fl.get_download_location()
        weights_candidate = None
        if isinstance(model_filename, (list, tuple)):
            if len(model_filename) > 0:
                weights_candidate = model_filename[0]
        else:
            weights_candidate = model_filename
        heartmula_weights_path = None
        if weights_candidate:
            heartmula_weights_path = fl.locate_file(
                weights_candidate, error_if_none=False
            )
            if heartmula_weights_path is None:
                heartmula_weights_path = weights_candidate
        pipeline = HeartMuLaPipeline(
            ckpt_root=ckpt_root,
            device=torch.device("cpu"),
            version=HEARTMULA_VERSION,
            VAE_dtype=VAE_dtype,
            heartmula_weights_path=heartmula_weights_path,
            cfg_scale=model_def.get("heartmula_cfg_scale", 1.5),
            topk=model_def.get("heartmula_topk", 50),
            max_audio_length_ms=model_def.get("heartmula_max_audio_length_ms", 120000),
            codec_steps=model_def.get("heartmula_codec_steps", 10),
            codec_guidance_scale=model_def.get("heartmula_codec_guidance_scale", 1.25),
            codec_version=model_def.get("heartmula_codec_version", ""),
        )

        pipeline.mula.decoder[0].layers._compile_me = False
        pipeline.mula.backbone.layers._compile_me = False
        pipe = {
            "transformer": pipeline.mula,
            "transformer2": pipeline.mula.decoder[0],
            "codec": pipeline.codec,
        }
        pipe = {
            "pipe": pipe,
            "coTenantsMap": {
                "transformer": ["transformer2"],
                "transformer2": ["transformer"],
            },
        }

        if int(profile) in (2, 4, 5):
            pipe["budgets"] = {"transformer2": 200}

        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""

        defaults = {
            "audio_prompt_type": "",
        }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

        if settings_version < 2.44:
            ui_defaults["guidance_scale"] = model_def.get("heartmula_cfg_scale", 1.5)
            ui_defaults["top_k"] = model_def.get("heartmula_topk", 50)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "alt_prompt": "piano,happy,wedding",
                "repeat_generation": 1,
                "duration_seconds": duration_def.get("default", 120),
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "temperature": 1.0,
                "guidance_scale": model_def.get("heartmula_cfg_scale", 1.5),
                "top_k": model_def.get("heartmula_topk", 50),
                "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        alt_prompt = inputs.get("alt_prompt", "")
        if alt_prompt is None or len(str(alt_prompt).strip()) == 0:
            return "Keywords prompt cannot be empty for HeartMuLa."
        if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
            return "HeartMuLa does not support reference audio yet."
        return None
