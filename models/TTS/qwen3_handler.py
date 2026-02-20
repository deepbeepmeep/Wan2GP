import json
import os
import re
from pathlib import Path
from typing import Optional

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


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
    "max": 600,
    "increment": 1,
    "default": 20,
}
QWEN3_TTS_AUDIO_PROMPT_TYPE_SOURCES = {
    "selection": ["A", "AB"],
    "labels": {
        "A": "Voice cloning of 1 speaker",
        "AB": "Voice cloning of 2 speakers (Speaker 1 and Speaker 2)",
    },
    "letters_filter": "AB",
    "default": "A",
}
QWEN3_TTS_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
QWEN3_TTS_AUTO_SPLIT_MIN_SECONDS = 5.0
QWEN3_TTS_AUTO_SPLIT_MAX_SECONDS = 90.0
QWEN3_TTS_CUSTOM_SETTINGS = [
    {
        "id": QWEN3_TTS_AUTO_SPLIT_SETTING_ID,
        "label": "Auto Split Every s (5-90, optional), may reduce VRAM requiremens for very long speeches.",
        "name": "Auto Split Every s",
        "type": "float",
    },
]


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
        "text_prompt_enhancer_max_tokens": 512,
        "prompt_enhancer_button_label": "Write",
        "compile": False,
        "parent_model_type": "qwen3_tts_base",
        "lm_engines": ["cg"],
            "prompt_enhancer_def": {
                "selection": ["T", "T1"] if base_model_type == "qwen3_tts_base" else ["T"],
                "labels": {
                    "T": "A Speech based on current Prompt",
                    "T1": "A Dialogue between two People based on current Prompt",
                },
                "default": "T",
        },
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
                "label": "Reference transcript(s) (optional, two-speaker: one per line)",
                "placeholder": "Speaker 1 reference transcript\nSpeaker 2 reference transcript",
                "lines": 3,
            },
            "pause_between_sentences": True,
            "preserve_empty_prompt_lines": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_prompt_type_sources": dict(QWEN3_TTS_AUDIO_PROMPT_TYPE_SOURCES),
            "custom_settings": [one.copy() for one in QWEN3_TTS_CUSTOM_SETTINGS],
            "text_prompt_enhancer_instructions1": TTS_QWEN3_DIALOGUE_PROMPT,
            "text_prompt_enhancer_max_tokens1": 512,
            "audio_guide_label": "Speaker 1 reference voice",
            "audio_guide2_label": "Speaker 2 reference voice",
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
        lm_decoder_engine="legacy",
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
            lm_decoder_engine=lm_decoder_engine,
        )
        if str(lm_decoder_engine).strip().lower() in ("cg", "cudagraph"):
            pipeline.model._budget = 0
            talker = getattr(pipeline.model, "talker", None)
            if talker is not None:
                talker._budget = 0
                code_predictor = getattr(talker, "code_predictor", None)
                if code_predictor is not None:
                    code_predictor._budget = 0

        pipe = {"transformer": pipeline.model}
        if getattr(pipeline, "speech_tokenizer", None) is not None:
            pipe["speech_tokenizer"] = pipeline.speech_tokenizer.model
        if save_quantized and weights_path:
            from wgp import save_quantized_model

            config_path = get_qwen3_config_path(base_model_type)
            if config_path is None:
                config_candidate = os.path.join("qwen3", "configs", f"{base_model_type}.json")
                config_path = fl.locate_file(config_candidate, error_if_none=False) or config_candidate
            save_quantized_model(pipeline.model, model_type, weights_path, dtype or torch.bfloat16, config_path)
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
                "pause_seconds": 0.5,
            }
        else:
            defaults = {
                "audio_prompt_type": "",
                "model_mode": "auto",
            }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)
        if base_model_type == "qwen3_tts_base":
            audio_prompt_type = str(ui_defaults.get("audio_prompt_type", "A") or "A").upper()
            if audio_prompt_type not in ("A", "AB"):
                ui_defaults["audio_prompt_type"] = "A"

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
                    "pause_seconds": 0.5,
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
            audio_prompt_type = str(inputs.get("audio_prompt_type", "A") or "A").upper()
            if audio_prompt_type not in ("A", "AB"):
                return "Unsupported audio prompt mode for Qwen3 Base. Use one speaker or two speakers."
            if inputs.get("audio_guide") is None:
                return "Qwen3 Base requires Speaker 1 reference audio."
            prompt_text = str(one_prompt)
            has_speaker_syntax = re.search(r"Speaker\s*\d+\s*:", prompt_text, flags=re.IGNORECASE) is not None
            if audio_prompt_type == "AB":
                if inputs.get("audio_guide2") is None:
                    return "Two-speaker mode requires Speaker 2 reference audio."
                speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:", prompt_text, flags=re.IGNORECASE))
                if not speaker_matches:
                    return (
                        "Two-speaker mode requires prompt lines using Speaker 1: and Speaker 2: "
                        "(or any two numeric speaker IDs). For headless settings, keep "
                        "'multi_prompts_gen_type' = 2 so dialogue lines stay in one prompt."
                    )
                speaker_ids = sorted({int(m.group(1)) for m in speaker_matches})
                if len(speaker_ids) != 2:
                    return (
                        "Two-speaker mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:. "
                        "For headless settings, keep 'multi_prompts_gen_type' = 2."
                    )
            elif has_speaker_syntax:
                return "Speaker-tag dialogue requires two-speaker mode (set audio prompt mode to Dialogue)."
            return None

        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type != "qwen3_tts_base":
            return None
        custom_settings = inputs.get("custom_settings", None)
        if custom_settings is None:
            return None
        if not isinstance(custom_settings, dict):
            return "Custom settings must be a dictionary."

        raw_value = custom_settings.get(QWEN3_TTS_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                custom_settings.pop(QWEN3_TTS_AUTO_SPLIT_SETTING_ID, None)
                inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
                return None

        try:
            if isinstance(raw_value, bool):
                raise ValueError()
            auto_split_seconds = float(raw_value)
        except Exception:
            return (
                f"Auto Split Every s must be a number between "
                f"{int(QWEN3_TTS_AUTO_SPLIT_MIN_SECONDS)} and {int(QWEN3_TTS_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        if (
            auto_split_seconds < QWEN3_TTS_AUTO_SPLIT_MIN_SECONDS
            or auto_split_seconds > QWEN3_TTS_AUTO_SPLIT_MAX_SECONDS
        ):
            return (
                f"Auto Split Every s must be between "
                f"{int(QWEN3_TTS_AUTO_SPLIT_MIN_SECONDS)} and {int(QWEN3_TTS_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        custom_settings[QWEN3_TTS_AUTO_SPLIT_SETTING_ID] = auto_split_seconds
        inputs["custom_settings"] = custom_settings
        return None
