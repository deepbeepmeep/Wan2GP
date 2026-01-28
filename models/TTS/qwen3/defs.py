from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


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
_QWEN3_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

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
TTS_MONOLOGUE_PROMPT = (
    "You are a speechwriting assistant. Generate a single-speaker monologue "
    "for a text-to-speech model based on the user prompt. Output only the "
    "monologue text. Do not include explanations, bullet lists, or stage "
    "directions. Keep a consistent tone and point of view. Use natural, "
    "spoken sentences with clear punctuation for pauses. Aim for a short "
    "monologue (4-8 sentences) unless the prompt asks for a different length.\n\n"
    "Example:\n"
    "I never thought a small town would teach me so much about patience. "
    "Every morning the same faces pass the bakery window, and I know their "
    "stories without a word. The bell over the door rings, the coffee steams, "
    "and time slows down just enough to breathe. Some days I miss the noise of "
    "the city, but most days I am grateful for the quiet. It lets me hear "
    "myself think, and that has become its own kind of music."
)
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
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": [base_model_type],
        "duration_slider": dict(QWEN3_TTS_DURATION_SLIDER),
        "top_k_slider": True,
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_PROMPT,
        "compile": False,
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
