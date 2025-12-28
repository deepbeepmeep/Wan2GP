import os
from shared.utils import files_locator as fl
import gradio as gr
try:
    from .mtl_tts import SUPPORTED_LANGUAGES as _SUPPORTED_LANGUAGES
except ImportError:  # pragma: no cover - fallback when package missing during startup
    _SUPPORTED_LANGUAGES = {
        "ar": "Arabic",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fi": "Finnish",
        "fr": "French",
        "he": "Hebrew",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "sv": "Swedish",
        "sw": "Swahili",
        "tr": "Turkish",
        "zh": "Chinese",
    }

LANGUAGE_CHOICES = [
    (f"{name} ({code})", code) for code, name in sorted(_SUPPORTED_LANGUAGES.items(), key=lambda item: item[1])
]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["chatterbox"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        # The numeric weight controls ordering in the family dropdown.
        return {"tts": (70, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-tts",
            type=str,
            default=os.path.join("loras", "tts"),
            help="Path to a directory that contains TTS settings"
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        return args.lora_dir_tts

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "audio_only": True,
            "image_outputs": False,
            "sliding_window": False,
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "image_prompt_types_allowed": "",
            "profiles_dir": ["chatterbox"],
            "audio_guide_label": "Voice to Replicate",
            "model_modes": {
                "choices": LANGUAGE_CHOICES,
                "default": "en",
                "label": "Language",
            },
        }
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization):
        mandatory_files = [
            "ve.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ]
        return {
            "repoId": "ResembleAI/chatterbox",
            "sourceFolderList": [""],
            "targetFolderList": ["chatterbox"],
            "fileList": [mandatory_files],
        }

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
        override_text_encoder = None,
    ):
        from .pipeline import ChatterboxPipeline

        ckpt_root = fl.get_download_location()
        pipeline = ChatterboxPipeline(ckpt_root=ckpt_root, device ="cpu")
        pipe = {"ve": pipeline.model.ve, "s3gen": pipeline.model.s3gen, "t3": pipeline.model.t3 , "conds": pipeline.model.conds}
        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        defaults = {
            "audio_prompt_type": "A",
            "model_mode": "en",
        }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "audio_prompt_type": "A",
                "model_mode": "en",
                "repeat_generation": 1,
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "chatterbox_cfg_weight": 0.5,
                "chatterbox_exaggeration": 0.5,
                "chatterbox_temperature": 0.8,
                "chatterbox_repetition_penalty": 2.0,
                "chatterbox_min_p": 0.05,
                "chatterbox_top_p": 1.0,
            }
        )


    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if len(one_prompt) > 300:
            gr.Info("It is recommended to use a prompt that has less than 300 characters, otherwise you may get unexpected results.")
