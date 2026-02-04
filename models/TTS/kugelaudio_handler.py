import os

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_OR_DIALOGUE_PROMPT


KUGELAUDIO_REPO_ID = "DeepBeepMeep/TTS"
KUGELAUDIO_ASSET_DIR = "kugelaudio"
KUGELAUDIO_TOKENIZER_DIR = "kugelaudio_text_tokenizer"

KUGELAUDIO_CONFIG_NAME = "config.json"
KUGELAUDIO_GENERATION_CONFIG_NAME = "generation_config.json"
KUGELAUDIO_TOKENIZER_FILES = [
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "preprocessor_config.json",
]

KUGELAUDIO_DURATION_SLIDER = {
    "label": "Max duration (seconds)",
    "min": 1,
    "max": 600,
    "increment": 1,
    "default": 20,
}


def _get_kugelaudio_model_def():
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
        "profiles_dir": ["kugelaudio_0_open"],
        "duration_slider": dict(KUGELAUDIO_DURATION_SLIDER),
        "pause_between_sentences": True,
        "any_audio_prompt": True,
        "audio_guide_label": "Reference voice (optional)",
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": {
            "selection": ["", "A", "AB"],
            "labels": {
                "": "Text only",
                "A": "Voice cloning (1 reference audio)",
                "AB": "Voice cloning (2 reference audios)",
            },
            "letters_filter": "AB",
            "default": "",
        },
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_OR_DIALOGUE_PROMPT,
        "compile": False,
    }


def _get_kugelaudio_download_def():
    return [
        {
            "repoId": KUGELAUDIO_REPO_ID,
            "sourceFolderList": [KUGELAUDIO_TOKENIZER_DIR],
            "fileList": [KUGELAUDIO_TOKENIZER_FILES],
        },
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["kugelaudio_0_open"]

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
            "--lora-dir-kugelaudio",
            type=str,
            default=None,
            help=f"Path to a directory that contains KugelAudio settings (default: {os.path.join(lora_root, 'kugelaudio')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_kugelaudio", None) or os.path.join(lora_root, "kugelaudio")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_kugelaudio_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_kugelaudio_download_def()

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
        from .kugelaudio.pipeline import KugelAudioPipeline

        weights_path = model_filename[0] 
        pipeline = KugelAudioPipeline(
            model_weights_path=weights_path,
            ckpt_root=fl.get_download_location(),
            device=torch.device("cpu"),
        )

        pipe = {
            "transformer": pipeline.model,
        }

        if save_quantized and weights_path:
            from wgp import save_quantized_model

            local_config_path = os.path.join(
                os.path.dirname(__file__), "kugelaudio", "configs", "kugelaudio", KUGELAUDIO_CONFIG_NAME
            )
            if os.path.isfile(local_config_path):
                config_path = local_config_path
            else:
                config_path = fl.locate_file(
                    os.path.join(KUGELAUDIO_ASSET_DIR, KUGELAUDIO_CONFIG_NAME),
                    error_if_none=False,
                )
                if config_path is None:
                    config_path = os.path.join(KUGELAUDIO_ASSET_DIR, KUGELAUDIO_CONFIG_NAME)
            save_quantized_model(
                pipeline.model,
                model_type,
                weights_path,
                dtype or torch.bfloat16,
                config_path,
            )

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

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "prompt": "Hello! This is KugelAudio speaking in a clear, friendly voice.",
                "repeat_generation": 1,
                "duration_seconds": duration_def.get("default", 60),
                "pause_seconds": 0.5,
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "temperature": 1.0,
                "guidance_scale": 3.0,
                "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for KugelAudio."
        text = str(one_prompt)
        if "Speaker" in text or "speaker" in text:
            if "A" not in audio_prompt_type or "B" not in audio_prompt_type:
                return "Multi-speaker prompts require two reference voice audio samples. Provide a voice sample or remove Speaker tags."
        if "B" in audio_prompt_type:
            if inputs.get("audio_guide") is None or inputs.get("audio_guide2") is None:
                return "Two-voice cloning requires two reference audio files."
            if "Speaker 1:" not in text and "speaker 1:" not in text:
                return "Two-voice cloning requires prompt lines with Speaker 0: and Speaker 1:."
        return None
