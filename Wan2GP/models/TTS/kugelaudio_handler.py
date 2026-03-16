import os
import re

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


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
KUGELAUDIO_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
KUGELAUDIO_AUTO_SPLIT_MIN_SECONDS = 5.0
KUGELAUDIO_AUTO_SPLIT_MAX_SECONDS = 90.0
KUGELAUDIO_CUSTOM_SETTINGS = [
    {
        "id": KUGELAUDIO_AUTO_SPLIT_SETTING_ID,
        "label": "Auto Split Every s (5-90, optional), to avoid Acceleration Effect. Empty Lines will force anyway Manual Splits.",
        "name": "Auto Split Every s",
        "type": "float",
    },
]


def _configure_diffusion_compile_targets(model):
    for _, submodule in model.named_modules():
        submodule._compile_me = False

    prediction_head = getattr(model, "prediction_head", None)
    if prediction_head is None:
        prediction_head = getattr(getattr(model, "model", None), "prediction_head", None)
    if prediction_head is None:
        raise RuntimeError("KugelAudio diffusion head is missing; cannot configure compile targets.")

    layers = getattr(prediction_head, "layers", None)
    if layers is not None:
        for layer in layers:
            layer._compile_me = True
    final_layer = getattr(prediction_head, "final_layer", None)
    if final_layer is not None:
        final_layer._compile_me = True


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
        "lm_engines": ["cg"],
        "duration_slider": dict(KUGELAUDIO_DURATION_SLIDER),
        "custom_settings": [one.copy() for one in KUGELAUDIO_CUSTOM_SETTINGS],
        "preserve_empty_prompt_lines": True,
        "pause_between_sentences": True,
        "any_audio_prompt": True,
        "audio_guide_label": "Reference voice (optional)",
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": {
            "selection": ["", "A", "AB"],
            "labels": {
                "": "Text only",
                "A": "Voice cloning (1 reference audio)",
                "AB": "Voice cloning (2 reference audios: Speaker 1 and Speaker 2)",
            },
            "letters_filter": "AB",
            "default": "",
        },
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_PROMPT,
        "text_prompt_enhancer_instructions1": TTS_QWEN3_DIALOGUE_PROMPT,
        "text_prompt_enhancer_max_tokens": 512,
        "text_prompt_enhancer_max_tokens1": 512,
        "prompt_enhancer_def": {
            "selection": ["T", "T1"],
            "labels": {
                "T": "A Speech based on current Prompt",
                "T1": "A Dialogue between two People based on current Prompt",
            },
            "default": "T",
        },
        "prompt_enhancer_button_label": "Write",
        "compile": ["transformer"],
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
        lm_decoder_engine="legacy",
        **kwargs,
    ):
        from .kugelaudio.pipeline import KugelAudioPipeline

        weights_path = model_filename[0] 
        pipeline = KugelAudioPipeline(
            model_weights_path=weights_path,
            ckpt_root=fl.get_download_location(),
            device=torch.device("cpu"),
            lm_decoder_engine=lm_decoder_engine,
        )
        if lm_decoder_engine == "cg":
            pipeline.model._budget = 0
            language_model = getattr(getattr(pipeline.model, "model", None), "language_model", None)
            if language_model is not None:
                language_model._budget = 0
        _configure_diffusion_compile_targets(pipeline.model)

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
            speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:", text, flags=re.IGNORECASE))
            if not speaker_matches:
                return (
                    "Two-voice cloning requires prompt lines with Speaker 1: and Speaker 2: "
                    "(or any two numeric speaker IDs). For headless settings, keep "
                    "'multi_prompts_gen_type' = 2 so dialogue lines stay in one prompt."
                )
            speaker_ids = sorted({int(m.group(1)) for m in speaker_matches})
            if len(speaker_ids) != 2:
                return (
                    "Two-voice cloning requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:. "
                    "For headless settings, keep 'multi_prompts_gen_type' = 2."
                )
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        custom_settings = inputs.get("custom_settings", None)
        if custom_settings is None:
            return None
        if not isinstance(custom_settings, dict):
            return "Custom settings must be a dictionary."

        raw_value = custom_settings.get(KUGELAUDIO_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                custom_settings.pop(KUGELAUDIO_AUTO_SPLIT_SETTING_ID, None)
                inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
                return None

        try:
            if isinstance(raw_value, bool):
                raise ValueError()
            auto_split_seconds = float(raw_value)
        except Exception:
            return (
                f"Auto Split Every s must be a number between "
                f"{int(KUGELAUDIO_AUTO_SPLIT_MIN_SECONDS)} and {int(KUGELAUDIO_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        if (
            auto_split_seconds < KUGELAUDIO_AUTO_SPLIT_MIN_SECONDS
            or auto_split_seconds > KUGELAUDIO_AUTO_SPLIT_MAX_SECONDS
        ):
            return (
                f"Auto Split Every s must be between "
                f"{int(KUGELAUDIO_AUTO_SPLIT_MIN_SECONDS)} and {int(KUGELAUDIO_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        custom_settings[KUGELAUDIO_AUTO_SPLIT_SETTING_ID] = auto_split_seconds
        inputs["custom_settings"] = custom_settings
        return None
