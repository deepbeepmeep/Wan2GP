import json
import os
import re

import torch
import whisper
from accelerate import init_empty_weights

from mmgp import offload

from shared.deepy.transcription import (
    WHISPER_MEDIUM_CONFIG_FILENAME,
    WHISPER_MEDIUM_FOLDER,
    WHISPER_MEDIUM_REPO,
    WHISPER_MEDIUM_WEIGHTS_FILENAME,
)
from shared.mps import mps_device_or
from shared.utils import files_locator as fl

from .omnivoice.pipeline import (
    OMNIVOICE_ASSET_DIR,
    OMNIVOICE_AUDIO_TOKENIZER_DIR,
    OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS,
    OMNIVOICE_AUTO_END_TRIM_FLAG,
    OMNIVOICE_AUTO_SPLIT_MAX_SECONDS,
    OMNIVOICE_AUTO_SPLIT_MIN_SECONDS,
    OMNIVOICE_AUTO_SPLIT_SETTING_ID,
    OMNIVOICE_CONFIG_NAME,
    OMNIVOICE_DEFAULT_VOICE_INSTRUCTION,
    normalize_omnivoice_voice_instruction,
)
from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


OMNIVOICE_REPO_ID = "DeepBeepMeep/TTS"
OMNIVOICE_MAIN_FILENAME = "omnivoice_bf16.safetensors"
OMNIVOICE_QUANT_FILENAME = "omnivoice_quanto_bf16_int8.safetensors"
OMNIVOICE_TOKENIZER_FILES = [
    OMNIVOICE_CONFIG_NAME,
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
OMNIVOICE_AUDIO_TOKENIZER_FILES = [
    "config.json",
    "preprocessor_config.json",
    OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS,
]
OMNIVOICE_WHISPER_FILES = [
    WHISPER_MEDIUM_CONFIG_FILENAME,
    WHISPER_MEDIUM_WEIGHTS_FILENAME,
]
OMNIVOICE_LANGUAGE_CHOICES = [
    ("Auto", "auto"),
    ("English", "english"),
    ("Chinese", "chinese"),
    ("French", "french"),
    ("German", "german"),
    ("Italian", "italian"),
    ("Japanese", "japanese"),
    ("Korean", "korean"),
    ("Portuguese", "portuguese"),
    ("Spanish", "spanish"),
    ("Arabic", "arabic"),
    ("Hindi", "hindi"),
    ("Russian", "russian"),
]
OMNIVOICE_DURATION_SLIDER = {
    "label": "Max duration (seconds)",
    "min": 1,
    "max": 600,
    "increment": 1,
    "default": 30,
}
OMNIVOICE_AUDIO_PROMPT_TYPE_SOURCES = {
    "selection": ["", "A", "AB"],
    "labels": {
        "": "Auto or voice design",
        "A": "Voice cloning (1 reference audio)",
        "AB": "Voice cloning dialogue (Speaker 1 and Speaker 2)",
    },
    "letters_filter": "AB",
    "default": "",
}
OMNIVOICE_AUDIO_PROMPT_TYPE_CUSTOM_OPTION = {
    "label": "Auto Detect Segment End",
    "flag": OMNIVOICE_AUTO_END_TRIM_FLAG,
}
OMNIVOICE_CUSTOM_SETTINGS = [
    {
        "id": OMNIVOICE_AUTO_SPLIT_SETTING_ID,
        "label": "Auto Split Every s (5-90, optional), may reduce VRAM requirements for very long speeches.",
        "name": "Auto Split Every s",
        "type": "float",
    },
]


def _detach_whisper_alignment_heads(whisper_model):
    alignment_heads = getattr(whisper_model, "alignment_heads", None)
    if alignment_heads is not None and getattr(alignment_heads, "layout", None) == torch.sparse_coo:
        whisper_model._buffers.pop("alignment_heads", None)
        object.__setattr__(whisper_model, "alignment_heads", alignment_heads)


def _load_omnivoice_whisper_medium():
    model_dir = fl.locate_folder(WHISPER_MEDIUM_FOLDER)
    config_path = os.path.join(model_dir, WHISPER_MEDIUM_CONFIG_FILENAME)
    weights_path = fl.locate_file(os.path.join(WHISPER_MEDIUM_FOLDER, WHISPER_MEDIUM_WEIGHTS_FILENAME))
    with open(config_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)
    dims = whisper.model.ModelDimensions(**dict(config.get("dims", {}) or {}))
    with init_empty_weights(include_buffers=False):
        whisper_model = whisper.model.Whisper(dims)
    whisper_model._buffers.pop("alignment_heads", None)
    offload.load_model_data(whisper_model, weights_path, default_dtype=torch.float32, writable_tensors=False)
    whisper_model.to(dtype=torch.float32)
    alignment_heads = str(config.get("alignment_heads", "") or "").strip()
    if len(alignment_heads) > 0:
        whisper_model.set_alignment_heads(alignment_heads.encode("ascii"))
    _detach_whisper_alignment_heads(whisper_model)
    whisper_model.eval().requires_grad_(False)
    whisper_model._model_dtype = torch.float32
    return whisper_model


def _get_omnivoice_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 1,
        "no_negative_prompt": True,
        "inference_steps": True,
        "temperature": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["omnivoice"],
        "duration_slider": dict(OMNIVOICE_DURATION_SLIDER),
        "model_modes": {
            "choices": list(OMNIVOICE_LANGUAGE_CHOICES),
            "default": "auto",
            "label": "Language",
        },
        "alt_prompt": {
            "label": "Voice instruction / reference transcript(s)",
            "placeholder": "Auto mode: optional voice tags such as female\nVoice clone: optional transcript; blank uses Whisper\nDialogue: Speaker 1: optional transcript\nSpeaker 2: optional transcript",
            "lines": 4,
        },
        "preserve_empty_prompt_lines": True,
        "pause_between_sentences": True,
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": dict(OMNIVOICE_AUDIO_PROMPT_TYPE_SOURCES),
        "audio_prompt_type_custom_option": dict(OMNIVOICE_AUDIO_PROMPT_TYPE_CUSTOM_OPTION),
        "custom_settings": [one.copy() for one in OMNIVOICE_CUSTOM_SETTINGS],
        "audio_guide_label": "Speaker 1 reference voice",
        "audio_guide2_label": "Speaker 2 reference voice",
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
        "compile": False,
    }


def _get_omnivoice_download_def():
    return [
        {
            "repoId": OMNIVOICE_REPO_ID,
            "sourceFolderList": [OMNIVOICE_ASSET_DIR, OMNIVOICE_AUDIO_TOKENIZER_DIR],
            "fileList": [OMNIVOICE_TOKENIZER_FILES, OMNIVOICE_AUDIO_TOKENIZER_FILES],
        },
        {
            "repoId": WHISPER_MEDIUM_REPO,
            "sourceFolderList": [WHISPER_MEDIUM_FOLDER],
            "fileList": [OMNIVOICE_WHISPER_FILES],
        }
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["omnivoice"]

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
            "--lora-dir-omnivoice",
            type=str,
            default=None,
            help=f"Path to a directory that contains OmniVoice settings (default: {os.path.join(lora_root, 'omnivoice')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_omnivoice", None) or os.path.join(lora_root, "omnivoice")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_omnivoice_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_omnivoice_download_def()

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
        from .omnivoice.pipeline import OmniVoicePipeline

        weights_path = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        pipeline = OmniVoicePipeline(
            model_weights_path=weights_path,
            ckpt_root=fl.get_download_location(),
            device=mps_device_or(torch.device("cpu")),
            dtype=dtype or torch.bfloat16,
        )
        whisper_model = _load_omnivoice_whisper_medium()
        pipeline.set_whisper_model(whisper_model)
        pipe = {
            "transformer": pipeline.model,
            "audio_tokenizer": pipeline.audio_tokenizer,
            "whisper": whisper_model,
        }

        if save_quantized and weights_path:
            from wgp import save_quantized_model

            config_path = fl.locate_file(os.path.join(OMNIVOICE_ASSET_DIR, OMNIVOICE_CONFIG_NAME))
            save_quantized_model(pipeline.model, model_type, weights_path, dtype or torch.bfloat16, config_path)

        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("model_mode", "auto")
        ui_defaults.setdefault("alt_prompt", "")
        ui_defaults["alt_prompt"] = normalize_omnivoice_voice_instruction(str(ui_defaults.get("alt_prompt") or ""))
        ui_defaults.setdefault("pause_seconds", 0.2)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "model_mode": "auto",
                "prompt": "The lights are already on, so we can start whenever you are ready.",
                "alt_prompt": OMNIVOICE_DEFAULT_VOICE_INSTRUCTION,
                "repeat_generation": 1,
                "duration_seconds": duration_def.get("default", 30),
                "pause_seconds": 0.2,
                "video_length": 0,
                "num_inference_steps": 32,
                "negative_prompt": "",
                "temperature": 0.1,
                "guidance_scale": 2.0,
                "multi_prompts_gen_type": "FG",
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for OmniVoice."
        audio_prompt_type = str(inputs.get("audio_prompt_type", "") or "").upper()
        text = str(one_prompt)
        has_speaker_syntax = re.search(r"Speaker\s*\d+\s*:", text, flags=re.IGNORECASE) is not None
        if "A" in audio_prompt_type and "B" not in audio_prompt_type and inputs.get("audio_guide") is None:
            return "OmniVoice voice cloning requires a reference audio file."
        if "B" in audio_prompt_type:
            if inputs.get("audio_guide") is None or inputs.get("audio_guide2") is None:
                return "OmniVoice dialogue mode requires two reference audio files."
            speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:", text, flags=re.IGNORECASE))
            if not speaker_matches:
                return "OmniVoice dialogue mode requires prompt lines using Speaker 1: and Speaker 2:."
            speaker_ids = sorted({int(m.group(1)) for m in speaker_matches})
            if len(speaker_ids) != 2:
                return "OmniVoice dialogue mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:."
        elif has_speaker_syntax:
            return "Speaker-tag dialogue requires OmniVoice two-speaker mode."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        custom_settings = inputs.get("custom_settings", None)
        if custom_settings is None:
            return None
        if not isinstance(custom_settings, dict):
            return "Custom settings must be a dictionary."

        raw_value = custom_settings.get(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                custom_settings.pop(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
                inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
                return None

        try:
            if isinstance(raw_value, bool):
                raise ValueError()
            auto_split_seconds = float(raw_value)
        except Exception:
            return (
                f"Auto Split Every s must be a number between "
                f"{int(OMNIVOICE_AUTO_SPLIT_MIN_SECONDS)} and {int(OMNIVOICE_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        if auto_split_seconds < OMNIVOICE_AUTO_SPLIT_MIN_SECONDS or auto_split_seconds > OMNIVOICE_AUTO_SPLIT_MAX_SECONDS:
            return (
                f"Auto Split Every s must be between "
                f"{int(OMNIVOICE_AUTO_SPLIT_MIN_SECONDS)} and {int(OMNIVOICE_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        custom_settings[OMNIVOICE_AUTO_SPLIT_SETTING_ID] = auto_split_seconds
        inputs["custom_settings"] = custom_settings
        return None
