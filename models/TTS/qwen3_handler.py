import json
import os
from pathlib import Path
from typing import Optional

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_PROMPT


QWEN3_TTS_VARIANTS = {
    "qwen3_tts_customvoice": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "config_file": "qwen3_tts_customvoice.json",
    },
    "qwen3_tts_voicedesign": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "config_file": "qwen3_tts_voicedesign.json",
    },
    "qwen3_tts_base": {
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "config_file": "qwen3_tts_base.json",
    },
}

QWEN3_TTS_GENERATION_CONFIG = "qwen3_tts_generation_config.json"
_QWEN3_CONFIG_DIR = Path(__file__).resolve().parent / "qwen3" / "configs"

QWEN3_TTS_TEXT_TOKENIZER_DIR = "qwen3_tts_text_tokenizer"
QWEN3_TTS_SPEECH_TOKENIZER_DIR = "qwen3_tts_tokenizer_12hz"
QWEN3_TTS_SPEECH_TOKENIZER_WEIGHTS = "qwen3_tts_tokenizer_12hz.safetensors"
QWEN3_TTS_REPO = "DeepBeepMeep/TTS"
QWEN3_TTS_TEXT_TOKENIZER_FILES = [
    "merges.txt",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]
QWEN3_TTS_SPEECH_TOKENIZER_FILES = [
    "config.json",
    "configuration.json",
    "preprocessor_config.json",
    QWEN3_TTS_SPEECH_TOKENIZER_WEIGHTS,
]

QWEN3_TTS_LANG_FALLBACK = [
    "auto",
    "chinese",
    "english",
    "japanese",
    "korean",
    "german",
    "french",
    "russian",
    "portuguese",
    "spanish",
    "italian",
]
QWEN3_TTS_SPEAKER_FALLBACK = [
    "serena",
    "vivian",
    "uncle_fu",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
    "eric",
    "dylan",
]
QWEN3_TTS_SPEAKER_META = {
    "vivian": {
        "style": "Bright, slightly edgy young female voice",
        "language": "Chinese",
    },
    "serena": {
        "style": "Warm, gentle young female voice",
        "language": "Chinese",
    },
    "uncle_fu": {
        "style": "Seasoned male voice with a low, mellow timbre",
        "language": "Chinese",
    },
    "dylan": {
        "style": "Youthful Beijing male voice with a clear, natural timbre",
        "language": "Chinese (Beijing Dialect)",
    },
    "eric": {
        "style": "Lively Chengdu male voice with a slightly husky brightness",
        "language": "Chinese (Sichuan Dialect)",
    },
    "ryan": {
        "style": "Dynamic male voice with strong rhythmic drive",
        "language": "English",
    },
    "aiden": {
        "style": "Sunny American male voice with a clear midrange",
        "language": "English",
    },
    "ono_anna": {
        "style": "Playful Japanese female voice with a light, nimble timbre",
        "language": "Japanese",
    },
    "sohee": {
        "style": "Warm Korean female voice with rich emotion",
        "language": "Korean",
    },
}
QWEN3_TTS_DURATION_SLIDER = {
    "label": "Max duration (seconds)",
    "min": 1,
    "max": 240,
    "increment": 1,
    "default": 20,
}


def _format_qwen3_label(value: str) -> str:
    return value.replace("_", " ").title()


def _format_qwen3_speaker_label(name: str) -> str:
    label = _format_qwen3_label(name)
    meta = QWEN3_TTS_SPEAKER_META.get(name.lower())
    if not meta:
        return label
    parts = []
    style = meta.get("style", "")
    language = meta.get("language", "")
    if style:
        parts.append(style)
    if language:
        parts.append(language)
    if not parts:
        return label
    return f"{label} ({'; '.join(parts)})"


def get_qwen3_config_path(base_model_type: str) -> Optional[str]:
    variant = QWEN3_TTS_VARIANTS.get(base_model_type)
    if variant is None:
        return None
    config_path = _QWEN3_CONFIG_DIR / variant["config_file"]
    return str(config_path) if config_path.is_file() else None


def get_qwen3_generation_config_path() -> Optional[str]:
    config_path = _QWEN3_CONFIG_DIR / QWEN3_TTS_GENERATION_CONFIG
    return str(config_path) if config_path.is_file() else None


def load_qwen3_config(base_model_type: str) -> Optional[dict]:
    config_path = get_qwen3_config_path(base_model_type)
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_qwen3_languages(base_model_type: str) -> list[str]:
    config = load_qwen3_config(base_model_type)
    if config is None:
        return list(QWEN3_TTS_LANG_FALLBACK)
    lang_map = config.get("talker_config", {}).get("codec_language_id", {})
    languages = [name for name in lang_map.keys() if "dialect" not in name.lower()]
    languages = ["auto"] + sorted({name.lower() for name in languages})
    return languages


def get_qwen3_speakers(base_model_type: str) -> list[str]:
    config = load_qwen3_config(base_model_type)
    if config is None:
        return list(QWEN3_TTS_SPEAKER_FALLBACK)
    speakers = list(config.get("talker_config", {}).get("spk_id", {}).keys())
    speakers = sorted({name.lower() for name in speakers})
    return speakers or list(QWEN3_TTS_SPEAKER_FALLBACK)


def get_qwen3_language_choices(base_model_type: str) -> list[tuple[str, str]]:
    return [(_format_qwen3_label(lang), lang) for lang in get_qwen3_languages(base_model_type)]


def get_qwen3_speaker_choices(base_model_type: str) -> list[tuple[str, str]]:
    return [(_format_qwen3_speaker_label(name), name) for name in get_qwen3_speakers(base_model_type)]


def get_qwen3_model_def(base_model_type: str) -> dict:
    common = {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": [base_model_type],
        "duration_slider": dict(QWEN3_TTS_DURATION_SLIDER),
        "top_k_slider": True,
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_PROMPT,
        "compile": False,
        "parent_model_type": "qwen3_tts_base",
    }
    if base_model_type == "qwen3_tts_customvoice":
        speakers = get_qwen3_speakers(base_model_type)
        default_speaker = speakers[0] if speakers else ""
        return {
            **common,
            "model_modes": {
                "choices": get_qwen3_speaker_choices(base_model_type),
                "default": default_speaker,
                "label": "Speaker",
            },
            "alt_prompt": {
                "label": "Instruction (optional)",
                "placeholder": "calm, friendly, slightly husky",
                "lines": 2,
            },
        }
    if base_model_type == "qwen3_tts_voicedesign":
        return {
            **common,
            "model_modes": {
                "choices": get_qwen3_language_choices(base_model_type),
                "default": "auto",
                "label": "Language",
            },
            "alt_prompt": {
                "label": "Voice instruction",
                "placeholder": "young female, warm tone, clear articulation",
                "lines": 2,
            },
        }
    if base_model_type == "qwen3_tts_base":
        return {
            **common,
            "model_modes": {
                "choices": get_qwen3_language_choices(base_model_type),
                "default": "auto",
                "label": "Language",
            },
            "alt_prompt": {
                "label": "Reference transcript (optional)",
                "placeholder": "Okay. Yeah. I respect you, but you blew it.",
                "lines": 3,
            },
            "any_audio_prompt": True,
            "audio_guide_label": "Reference voice",
        }
    return common


def get_qwen3_duration_default() -> int:
    return int(QWEN3_TTS_DURATION_SLIDER.get("default", 20))


def get_qwen3_download_def(base_model_type: str) -> list[dict]:
    return [
        {
            "repoId": QWEN3_TTS_REPO,
            "sourceFolderList": [QWEN3_TTS_TEXT_TOKENIZER_DIR],
            "fileList": [QWEN3_TTS_TEXT_TOKENIZER_FILES],
        },
        {
            "repoId": QWEN3_TTS_REPO,
            "sourceFolderList": [QWEN3_TTS_SPEECH_TOKENIZER_DIR],
            "fileList": [QWEN3_TTS_SPEECH_TOKENIZER_FILES],
        },
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return list(QWEN3_TTS_VARIANTS)

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
            "--lora-dir-qwen3-tts",
            type=str,
            default=None,
            help=f"Path to a directory that contains Qwen3 TTS settings (default: {os.path.join(lora_root, 'qwen3_tts')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_qwen3_tts", None) or os.path.join(lora_root, "qwen3_tts")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return get_qwen3_model_def(base_model_type)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return get_qwen3_download_def(base_model_type)

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
        from .qwen3.pipeline import Qwen3TTSPipeline

        ckpt_root = fl.get_download_location()
        weights_candidate = None
        if isinstance(model_filename, (list, tuple)):
            if len(model_filename) > 0:
                weights_candidate = model_filename[0]
        else:
            weights_candidate = model_filename
        weights_path = None
        if weights_candidate:
            weights_path = fl.locate_file(weights_candidate, error_if_none=False)
            if weights_path is None:
                weights_path = weights_candidate

        pipeline = Qwen3TTSPipeline(
            model_weights_path=weights_path,
            base_model_type=base_model_type,
            ckpt_root=ckpt_root,
            device=torch.device("cpu"),
        )

        pipe = {"transformer": pipeline.model}
        if getattr(pipeline, "speech_tokenizer", None) is not None:
            pipe["speech_tokenizer"] = pipeline.speech_tokenizer.model
        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""

        if base_model_type == "qwen3_tts_customvoice":
            speakers = get_qwen3_speakers(base_model_type)
            defaults = {
                "audio_prompt_type": "",
                "model_mode": speakers[0] if speakers else "",
            }
        elif base_model_type == "qwen3_tts_voicedesign":
            defaults = {
                "audio_prompt_type": "",
                "model_mode": "auto",
            }
        elif base_model_type == "qwen3_tts_base":
            defaults = {
                "audio_prompt_type": "A",
                "model_mode": "auto",
            }
        else:
            defaults = {
                "audio_prompt_type": "",
                "model_mode": "auto",
            }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

        if settings_version < 2.44:
            if model_def.get("top_k_slider", False):
                ui_defaults["top_k"] = 50

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == "qwen3_tts_customvoice":
            speakers = get_qwen3_speakers(base_model_type)
            default_speaker = speakers[0] if speakers else ""
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "model_mode": default_speaker,
                    "alt_prompt": "",
                    "duration_seconds": get_qwen3_duration_default(),
                    "repeat_generation": 1,
                    "video_length": 0,
                    "num_inference_steps": 0,
                    "negative_prompt": "",
                    "temperature": 0.9,
                    "top_k": 50,
                    "multi_prompts_gen_type": 2,
                }
            )
            return

        if base_model_type == "qwen3_tts_voicedesign":
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "model_mode": "auto",
                    "alt_prompt": "young female, warm tone, clear articulation",
                    "duration_seconds": get_qwen3_duration_default(),
                    "repeat_generation": 1,
                    "video_length": 0,
                    "num_inference_steps": 0,
                    "negative_prompt": "",
                    "temperature": 0.9,
                    "top_k": 50,
                    "multi_prompts_gen_type": 2,
                }
            )
            return

        if base_model_type == "qwen3_tts_base":
            ui_defaults.update(
                {
                    "audio_prompt_type": "A",
                    "model_mode": "auto",
                    "alt_prompt": "",
                    "duration_seconds": get_qwen3_duration_default(),
                    "repeat_generation": 1,
                    "video_length": 0,
                    "num_inference_steps": 0,
                    "negative_prompt": "",
                    "temperature": 0.9,
                    "top_k": 50,
                    "multi_prompts_gen_type": 2,
                }
            )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if base_model_type == "qwen3_tts_customvoice":
            if one_prompt is None or len(str(one_prompt).strip()) == 0:
                return "Prompt text cannot be empty for Qwen3 CustomVoice."
            speaker = inputs.get("model_mode", "")
            if not speaker:
                return "Please select a speaker for Qwen3 CustomVoice."
            speakers = get_qwen3_speakers(base_model_type)
            if speaker.lower() not in speakers:
                return f"Unsupported speaker '{speaker}'."
            return None

        if base_model_type == "qwen3_tts_voicedesign":
            if one_prompt is None or len(str(one_prompt).strip()) == 0:
                return "Prompt text cannot be empty for Qwen3 VoiceDesign."
            return None

        if base_model_type == "qwen3_tts_base":
            if one_prompt is None or len(str(one_prompt).strip()) == 0:
                return "Prompt text cannot be empty for Qwen3 Base voice clone."
            if inputs.get("audio_guide") is None:
                return "Qwen3 Base requires a reference audio clip."
            return None

        return None
