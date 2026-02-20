import os
import re

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import HEARTMULA_LYRIC_PROMPT


ACE_STEP_REPO_ID = "DeepBeepMeep/TTS"
ACE_STEP_REPO_FOLDER = "ace_step"

ACE_STEP_TRANSFORMER_CONFIG_NAME = "ace_step_v1_transformer_config.json"
ACE_STEP_DCAE_WEIGHTS_NAME = "ace_step_v1_music_dcae_f8c8_bf16.safetensors"
ACE_STEP_DCAE_CONFIG_NAME = "ace_step_v1_dcae_config.json"
ACE_STEP_VOCODER_WEIGHTS_NAME = "ace_step_v1_music_vocoder_bf16.safetensors"
ACE_STEP_VOCODER_CONFIG_NAME = "ace_step_v1_vocoder_config.json"
ACE_STEP_TEXT_ENCODER_NAME = "umt5_base_bf16.safetensors"
ACE_STEP_TEXT_ENCODER_FOLDER = "umt5_base"

ACE_STEP_TEXT_ENCODER_URL = (
    f"https://huggingface.co/{ACE_STEP_REPO_ID}/resolve/main/"
    f"{ACE_STEP_TEXT_ENCODER_FOLDER}/{ACE_STEP_TEXT_ENCODER_NAME}"
)

ACE_STEP15_REPO_ID = "DeepBeepMeep/TTS"
ACE_STEP15_REPO_FOLDER = "ace_step15"
ACE_STEP15_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "ace_step15", "configs")

ACE_STEP15_TRANSFORMER_CONFIG_NAME = "ace_step_v1_5_transformer_config.json"
ACE_STEP15_VAE_WEIGHTS_NAME = "ace_step_v1_5_audio_vae_bf16.safetensors"
ACE_STEP15_VAE_CONFIG_NAME = "ace_step_v1_5_audio_vae_config.json"
ACE_STEP15_TEXT_ENCODER_2_FOLDER = "Qwen3-Embedding-0.6B"
ACE_STEP15_TEXT_ENCODER_2_NAME = "model.safetensors"
ACE_STEP15_LM_FOLDER = "acestep-5Hz-lm-1.7B"
ACE_STEP15_SILENCE_LATENT_NAME = "silence_latent.pt"

ACE_STEP15_TRANSFORMER_VARIANTS = {
    "base": "ace_step_v1_5_transformer_config_base.json",
    "sft": "ace_step_v1_5_transformer_config_sft.json",
    "turbo": "ace_step_v1_5_transformer_config_turbo.json",
    "turbo_shift1": "ace_step_v1_5_transformer_config_turbo_shift1.json",
    "turbo_shift3": "ace_step_v1_5_transformer_config_turbo_shift3.json",
    "turbo_continuous": "ace_step_v1_5_transformer_config_turbo_continuous.json",
}

def _ace_step15_lm_weights_name(lm_folder):
    folder_name = os.path.basename(os.path.normpath(str(lm_folder)))
    return f"{folder_name}_bf16.safetensors"

ACE_STEP_DURATION_SLIDER = {
    "label": "Duration (seconds)",
    "min": 5,
    "max": 240,
    "increment": 1,
    "default": 20,
}

ACE_STEP15_DURATION_SLIDER = {
    "label": "Duration (seconds)",
    "min": 5,
    "max": 360,
    "increment": 1,
    "default": 20,
}

ACE_STEP_BPM_MIN = 30
ACE_STEP_BPM_MAX = 300
ACE_STEP_BPM_HINT = f"Use an integer from {ACE_STEP_BPM_MIN} to {ACE_STEP_BPM_MAX} (leave empty for N/A)."
ACE_STEP_TIME_SIGNATURE_VALUES = {2, 3, 4, 6}
ACE_STEP_TIME_SIGNATURE_HINT = "Use a single digit supported by ACE: 2, 3, 4, or 6 (leave empty for N/A)."
ACE_STEP_KEYSCALE_HINT = (
    "Use <NOTE><ACCIDENTAL> <MODE> where NOTE is A/B/C/D/E/F/G, "
    "ACCIDENTAL is optional (# or b, Unicode ♯/♭ also accepted), "
    "and MODE is major or minor. "
    "Short form <NOTE><ACCIDENTAL>m is also accepted. Leave empty for N/A."
)
ACE_STEP15_VALID_LANGUAGES = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
]
ACE_STEP15_VALID_LANGUAGE_SET = set(ACE_STEP15_VALID_LANGUAGES)
ACE_STEP15_LANGUAGE_CODES_TEXT = ", ".join(ACE_STEP15_VALID_LANGUAGES)
ACE_STEP15_CUSTOM_SETTINGS = [
    {
        "id": "bpm",
        "label": f"BPM ({ACE_STEP_BPM_MIN}-{ACE_STEP_BPM_MAX})",
        "name": "BPM",
        "type": "int",
    },
    {
        "id": "keyscale",
        "label": "KeyScale (C major, F# minor, ...)",
        "name": "KeyScale",
        "type": "text",
    },
    {
        "id": "timesignature",
        "label": "Time Signature (2,3,4,6)",
        "name": "Time Signature",
        "type": "int",
    },
    {
        "id": "language",
        "label": "Language (ISO code, empty = auto/en)",
        "name": "Language",
        "type": "text",
        "default": "",
    },
]
ACE_STEP15_MODEL_MODES = {
    "choices": [
        ("Generate Audio Codes based on Lyrics for better Semantic Understanding", 0),
        ("+ Compute empty Bpm, Keyscale, Time Signature, Language using Lyrics & Music Caption", 1),
        ("++ Refine Caption", 2),
        ("++ Determine Best Song Duration based on Lyrics & Music Caption", 4),
        ("+++ Refine Caption & Determine Best Song Duration based on Lyrics & Music Caption", 3),
    ],
    "default": 0,
    "label": "LM Chain Of Thought Preprocessing",
}
ACE_STEP15_SETTING_ALIASES = {
    "bpm": "bpm",
    "keyscale": "keyscale",
    "key_scale": "keyscale",
    "timesignature": "timesignature",
    "time_signature": "timesignature",
    "language": "language",
    "lang": "language",
    "language_code": "language",
}
ACE_STEP_V1_SAMPLE_SOLVERS = [
    ("Euler", "euler"),
    ("Heun", "heun"),
    ("PingPong", "pingpong"),
]


def _normalize_ace_setting_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def _resolve_ace_setting_id(setting_def):
    raw_name = setting_def.get("id") or setting_def.get("param") or setting_def.get("name") or ""
    normalized_name = _normalize_ace_setting_name(raw_name)
    return ACE_STEP15_SETTING_ALIASES.get(normalized_name, normalized_name)


def _normalize_keyscale_value(value):
    if value is None:
        return None, None
    keyscale = str(value).strip()
    if len(keyscale) == 0:
        return None, None
    lowered = keyscale.lower()
    if lowered in {"n/a", "na", "none"}:
        return None, None
    keyscale = keyscale.replace("\u266f", "#").replace("\u266d", "b")

    short_minor = re.fullmatch(r"([A-Ga-g])\s*([#b]?)\s*[mM]", keyscale)
    if short_minor:
        note = short_minor.group(1).upper()
        accidental = short_minor.group(2)
        return f"{note}{accidental} minor", None

    full = re.fullmatch(r"([A-Ga-g])\s*([#b]?)\s*(major|minor|maj|min)", keyscale, flags=re.IGNORECASE)
    if not full:
        return None, ACE_STEP_KEYSCALE_HINT
    note = full.group(1).upper()
    accidental = full.group(2)
    mode = full.group(3).lower()
    if mode == "maj":
        mode = "major"
    elif mode == "min":
        mode = "minor"
    return f"{note}{accidental} {mode}", None


def _get_model_path(model_def, key, default):
    if not model_def:
        return default
    value = model_def.get(key, None)
    if value is None or value == "":
        model_block = model_def.get("model", {}) if isinstance(model_def, dict) else {}
        value = model_block.get(key, None)
    return value or default

def _ace_step_ckpt_file(filename):
    rel_path = os.path.join(ACE_STEP_REPO_FOLDER, filename)
    return fl.locate_file(rel_path, error_if_none=False) or rel_path


def _ace_step_ckpt_dir(dirname):
    rel_path = os.path.join(ACE_STEP_REPO_FOLDER, dirname)
    return fl.locate_folder(rel_path, error_if_none=False) or rel_path


def _ckpt_dir(dirname):
    return fl.locate_folder(dirname, error_if_none=False) or dirname


def _ace_step15_ckpt_file(filename):
    rel_path = os.path.join(ACE_STEP15_REPO_FOLDER, filename)
    return fl.locate_file(rel_path, error_if_none=False) or rel_path


def _ace_step15_ckpt_dir(dirname):
    rel_path = os.path.join(ACE_STEP15_REPO_FOLDER, dirname)
    return fl.locate_folder(rel_path, error_if_none=False) or rel_path


def _ace_step15_lm_ckpt_file(filename):
    return fl.locate_file(filename, error_if_none=False) or filename


def _ace_step15_lm_ckpt_dir(dirname):
    return fl.locate_folder(dirname, error_if_none=False) or dirname


def _ace_step15_config_path(filename):
    return os.path.join(ACE_STEP15_CONFIG_DIR, filename)


def _is_ace_step15(base_model_type):
    return base_model_type == "ace_step_v1_5"


def _ace_step15_has_lm_definition(model_def):
    text_encoder_urls = _get_model_path(model_def, "text_encoder_URLs", None)
    if isinstance(text_encoder_urls, str):
        return len(text_encoder_urls.strip()) > 0
    if isinstance(text_encoder_urls, (list, tuple)):
        return any(isinstance(one, str) and len(one.strip()) > 0 for one in text_encoder_urls)
    return False


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ace_step_v1", "ace_step_v1_5"]

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
            "--lora-dir-ace-step",
            type=str,
            default=None,
            help=f"Path to a directory that contains Ace Step settings (default: {os.path.join(lora_root, 'ace_step')})",
        )
        parser.add_argument(
            "--lora-dir-ace-step15",
            dest="lora_ace_step15",
            type=str,
            default=None,
            help=f"Path to a directory that contains Ace Step 1.5 settings (default: {os.path.join(lora_root, 'ace_step_v1_5')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if _is_ace_step15(base_model_type):
            return getattr(args, "lora_ace_step15", None) or os.path.join(lora_root, "ace_step_v1_5")
        return getattr(args, "lora_ace_step", None) or os.path.join(lora_root, "ace_step")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        if _is_ace_step15(base_model_type):
            extra_model_def = {
                "audio_only": True,
                "image_outputs": False,
                "sliding_window": False,
                "guidance_max_phases": 0,
                "lock_inference_steps": True,
                "no_negative_prompt": True,
                "image_prompt_types_allowed": "",
                "profiles_dir": ["ace_step_v1_5"],
                "text_encoder_folder": _get_model_path(model_def, "text_encoder_folder", ACE_STEP15_LM_FOLDER),
                "inference_steps": True,
                "temperature": True,
                "top_p_slider": True,
                "top_k_slider": True,
                "any_audio_prompt": True,
                "audio_guide_label": "Source Audio",
                "audio_guide2_label": "Reference Timbre",
                "audio_scale_name": "Source Audio Strength",
                "audio_prompt_choices": True,
                "enabled_audio_lora": True,
                "lm_engines": ["vllm"],
                "prompt_class": "Lyrics",
                "prompt_description": "Lyrics / Prompt (Write [Instrumental] for Instrumental Generation only)",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "B", "AB"],
                    "labels": {
                        "": "Text (Lyrics) 2 Audio",
                        "A": "Cover Mode of Source Audio (need to provide original Lyrics and set a Source Audio Strength)",
                        "B": "Transfer Reference Audio Timbre",
                        "AB": "Cover Mode of Source Audio + Transfer Reference Audio Timbre",
                    },
                    "default": "",
                    "label": "Audio Task",
                    "letters_filter": "AB",
                },
                "alt_prompt": {
                    "label": "Music Caption (Describe the style, genre, instruments, and mood)",
                    "name": "Music Caption",
                    "placeholder": "disco",
                    "lines": 2,
                },
                "duration_slider": dict(ACE_STEP15_DURATION_SLIDER),
                "custom_settings": [one.copy() for one in ACE_STEP15_CUSTOM_SETTINGS],
                "text_prompt_enhancer_instructions": HEARTMULA_LYRIC_PROMPT,
                "text_prompt_enhancer_max_tokens": 512,
                "prompt_enhancer_button_label": "Compose Lyrics",
            }
            if _ace_step15_has_lm_definition(model_def):
                extra_model_def["model_modes"] = ACE_STEP15_MODEL_MODES.copy()
            return extra_model_def
        return {
            "audio_only": True,
            "image_outputs": False,
            "sliding_window": False,
            "guidance_max_phases": 1,
            "no_negative_prompt": True,
            "image_prompt_types_allowed": "",
            "profiles_dir": ["ace_step_v1"],
            "text_encoder_URLs": [ACE_STEP_TEXT_ENCODER_URL],
            "text_encoder_folder": ACE_STEP_TEXT_ENCODER_FOLDER,
            "inference_steps": True,
            "sample_solvers": ACE_STEP_V1_SAMPLE_SOLVERS,
            "temperature": False,
            "any_audio_prompt": True,
            "audio_guide_label": "Source Audio",
            "audio_scale_name": "Prompt Audio Strength",
            "audio_prompt_choices": True,
            "enabled_audio_lora": True,
            "audio_prompt_type_sources": {
                "selection": ["", "A"],
                "labels": {
                    "": "No Source Audio",
                    "A": "Remix Audio (need to provide original lyrics and set an Audio Prompt strength)",
                },
                "default": "",
                "label": "Source Audio Mode",
                "letters_filter": "A",
            },
            "alt_prompt": {
                "label": "Genres / Tags",
                "placeholder": "disco",
                "lines": 2,
            },
            "duration_slider": dict(ACE_STEP_DURATION_SLIDER),
            "text_prompt_enhancer_instructions": HEARTMULA_LYRIC_PROMPT,
            "prompt_enhancer_button_label": "Compose Lyrics",
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if _is_ace_step15(base_model_type):
            enable_lm = _ace_step15_has_lm_definition(model_def)
            text_encoder_2_folder = _get_model_path(model_def, "ACE_STEP15_TEXT_ENCODER_2_FOLDER", ACE_STEP15_TEXT_ENCODER_2_FOLDER)
            base_files = [
                ACE_STEP15_VAE_WEIGHTS_NAME,
                ACE_STEP15_SILENCE_LATENT_NAME,
            ]
            text_encoder_2_files = [
                ACE_STEP15_TEXT_ENCODER_2_NAME,
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]
            source_folders = [
                ACE_STEP15_REPO_FOLDER,
                text_encoder_2_folder,
            ]
            file_lists = [
                base_files,
                text_encoder_2_files,
            ]
            target_folders = [None, None]
            if enable_lm:
                lm_folder = _get_model_path(model_def, "text_encoder_folder", ACE_STEP15_LM_FOLDER)
                lm_files = [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "merges.txt",
                    "vocab.json",
                    "chat_template.jinja",
                ]
                source_folders.append(lm_folder)
                file_lists.append(lm_files)
                target_folders.append(None)
            return {
                "repoId": ACE_STEP15_REPO_ID,
                "sourceFolderList": source_folders,
                "targetFolderList": target_folders,
                "fileList": file_lists,
            }
        text_encoder_folder = _get_model_path(model_def, "text_encoder_folder", ACE_STEP_TEXT_ENCODER_FOLDER)
        base_files = [
            ACE_STEP_TRANSFORMER_CONFIG_NAME,
            ACE_STEP_DCAE_WEIGHTS_NAME,
            ACE_STEP_DCAE_CONFIG_NAME,
            ACE_STEP_VOCODER_WEIGHTS_NAME,
            ACE_STEP_VOCODER_CONFIG_NAME,
        ]
        tokenizer_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        return {
            "repoId": ACE_STEP_REPO_ID,
            "sourceFolderList": [
                ACE_STEP_REPO_FOLDER,
                text_encoder_folder,
            ],
            "targetFolderList": [None, None],
            "fileList": [base_files, tokenizer_files],
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
        text_encoder_filename=None,
        profile=0,
        lm_decoder_engine="legacy",
        **kwargs,
    ):
        transformer_weights = None
        if isinstance(model_filename, (list, tuple)):
            transformer_weights = model_filename[0] if model_filename else None
        else:
            transformer_weights = model_filename

        if _is_ace_step15(base_model_type):
            from .ace_step15.pipeline_ace_step15 import ACEStep15Pipeline

            transformer_variant = _get_model_path(model_def, "ace_step15_transformer_variant", "")
            transformer_config_name = ACE_STEP15_TRANSFORMER_CONFIG_NAME
            if transformer_variant:
                transformer_config_name = ACE_STEP15_TRANSFORMER_VARIANTS.get(
                    str(transformer_variant).lower(),
                    transformer_config_name,
                )
            transformer_config = _get_model_path(model_def, "ace_step15_transformer_config", _ace_step15_config_path(transformer_config_name))
            vae_weights = _get_model_path(model_def, "ace_step15_vae_weights", _ace_step15_ckpt_file(ACE_STEP15_VAE_WEIGHTS_NAME))
            vae_config = _get_model_path(model_def, "ace_step15_vae_config", _ace_step15_config_path(ACE_STEP15_VAE_CONFIG_NAME))

            text_encoder_2_folder = _get_model_path(model_def, "ACE_STEP15_TEXT_ENCODER_2_FOLDER", ACE_STEP15_TEXT_ENCODER_2_FOLDER)
            text_encoder_2_weights = _get_model_path(
                model_def,
                "ace_step15_text_encoder_2_weights",
                fl.locate_file(os.path.join(text_encoder_2_folder, ACE_STEP15_TEXT_ENCODER_2_NAME), error_if_none=False)
                or os.path.join(text_encoder_2_folder, ACE_STEP15_TEXT_ENCODER_2_NAME),
            )
            pre_text_tokenizer_dir = _get_model_path(model_def, "ace_step15_pre_text_tokenizer_dir", _ckpt_dir(text_encoder_2_folder))

            enable_lm = bool(text_encoder_filename)
            ignore_lm_cache_seed = bool(_get_model_path(model_def, "ace_step15_lm_cache_ignore_seed", False))
            lm_folder = _get_model_path(model_def, "text_encoder_folder", ACE_STEP15_LM_FOLDER)
            lm_weights = text_encoder_filename
            lm_tokenizer_dir = _get_model_path(model_def, "ace_step15_lm_tokenizer_dir", _ace_step15_lm_ckpt_dir(lm_folder))
            silence_latent = _get_model_path(model_def, "ace_step15_silence_latent", _ace_step15_ckpt_file(ACE_STEP15_SILENCE_LATENT_NAME))
            if enable_lm:
                lm_weight_name = os.path.basename(str(lm_weights)) if lm_weights else ""
                print(f"[ace_step15] LM engine='{lm_decoder_engine}' | LM weights='{lm_weight_name}'")

            pipeline = ACEStep15Pipeline(
                transformer_weights_path=transformer_weights,
                transformer_config_path=transformer_config,
                vae_weights_path=vae_weights,
                vae_config_path=vae_config,
                text_encoder_2_weights_path=text_encoder_2_weights,
                text_encoder_2_tokenizer_dir=pre_text_tokenizer_dir,
                lm_weights_path=lm_weights,
                lm_tokenizer_dir=lm_tokenizer_dir,
                silence_latent_path=silence_latent,
                enable_lm=enable_lm,
                ignore_lm_cache_seed=ignore_lm_cache_seed,
                lm_decoder_engine=lm_decoder_engine,
                dtype=dtype or torch.bfloat16,
            )

            pipe = {
                "transformer": pipeline.ace_step_transformer,
                "text_encoder_2": pipeline.text_encoder_2,
                "codec": pipeline.audio_vae,
            }
            if text_encoder_filename and pipeline.lm_model is not None:
                pipe["text_encoder"] = pipeline.lm_model

            pipe = { "pipe": pipe, "coTenantsMap": {}, }

            if save_quantized and transformer_weights:
                from wgp import save_quantized_model

                save_quantized_model(
                    pipeline.ace_step_transformer,
                    model_type,
                    transformer_weights,
                    dtype or torch.bfloat16,
                    transformer_config,
                )

            return pipeline, pipe
        else:
            from .ace_step.pipeline_ace_step import ACEStepPipeline

            transformer_config = _get_model_path(model_def, "ace_step_transformer_config", _ace_step_ckpt_file(ACE_STEP_TRANSFORMER_CONFIG_NAME))
            dcae_weights = _get_model_path(model_def, "ace_step_dcae_weights", _ace_step_ckpt_file(ACE_STEP_DCAE_WEIGHTS_NAME))
            dcae_config = _get_model_path(model_def, "ace_step_dcae_config", _ace_step_ckpt_file(ACE_STEP_DCAE_CONFIG_NAME))
            vocoder_weights = _get_model_path(model_def, "ace_step_vocoder_weights", _ace_step_ckpt_file(ACE_STEP_VOCODER_WEIGHTS_NAME))
            vocoder_config = _get_model_path(model_def, "ace_step_vocoder_config", _ace_step_ckpt_file(ACE_STEP_VOCODER_CONFIG_NAME))
            text_encoder_folder = _get_model_path(model_def, "text_encoder_folder", ACE_STEP_TEXT_ENCODER_FOLDER)
            text_encoder_weights = text_encoder_filename or _get_model_path(model_def, "ace_step_text_encoder_weights", os.path.join(text_encoder_folder, ACE_STEP_TEXT_ENCODER_NAME))
            tokenizer_dir = _get_model_path(model_def, "ace_step_tokenizer_dir", _ckpt_dir(text_encoder_folder))

            pipeline = ACEStepPipeline(
                transformer_weights_path=transformer_weights,
                transformer_config_path=transformer_config,
                dcae_weights_path=dcae_weights,
                dcae_config_path=dcae_config,
                vocoder_weights_path=vocoder_weights,
                vocoder_config_path=vocoder_config,
                text_encoder_weights_path=text_encoder_weights,
                text_encoder_tokenizer_dir=tokenizer_dir,
                dtype=dtype or torch.bfloat16,
            )

            pipe = {
                "transformer": pipeline.ace_step_transformer,
                "text_encoder": pipeline.text_encoder_model,
                "codec": pipeline.music_dcae,
            }
            if save_quantized and transformer_weights:
                from wgp import get_model_def, save_quantized_model

                save_quantized_model(
                    pipeline.ace_step_transformer,
                    model_type,
                    transformer_weights,
                    dtype or torch.bfloat16,
                    transformer_config,
                )
            return pipeline, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        if _is_ace_step15(base_model_type):
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "prompt": "[Verse]\\nNeon rain on the city line\\n"
                    "You hum the tune and I fall in time\\n"
                    "[Chorus]\\nHold me close and keep the time",
                    "alt_prompt": "dreamy synth-pop, shimmering pads, soft vocals",
                    "duration_seconds": duration_def.get("default", 60),
                    "repeat_generation": 1,
                    "video_length": 0,
                    "num_inference_steps": 8,
                    "negative_prompt": "",
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 0,
                    "guidance_scale": 1.0,
                    "multi_prompts_gen_type": 2,
                    "audio_scale": 0.5,
                }
            )
            # default_custom_settings = {}
            # for setting_def in model_def.get("custom_settings", []):
            #     setting_id = _resolve_ace_setting_id(setting_def)
            #     default_value = setting_def.get("default", None)
            #     if default_value is None:
            #         continue
            #     if isinstance(default_value, str) and len(default_value.strip()) == 0:
            #         continue
            #     default_custom_settings[setting_id] = default_value
            # ui_defaults["custom_settings"] = default_custom_settings if len(default_custom_settings) > 0 else None
            return
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "prompt": "[Verse]\\nNeon rain on the city line\\n"
                "You hum the tune and I fall in time\\n"
                "[Chorus]\\nHold me close and keep the time",
                "alt_prompt": "dreamy synth-pop, shimmering pads, soft vocals",
                "sample_solver": ui_defaults.get("sample_solver", ui_defaults.get("scheduler_type", "euler")),
                "duration_seconds": duration_def.get("default", 60),
                "repeat_generation": 1,
                "video_length": 0,
                "num_inference_steps": 60,
                "negative_prompt": "",
                "temperature": 1.0,
                "guidance_scale": 7.0,
                "multi_prompts_gen_type": 2,
                "audio_scale": 0.5,
            }
        )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if _is_ace_step15(base_model_type):
            if settings_version < 2.51:
                # ACE-Step 1.5 implicit LM defaults are: top-p 0.9, top-k disabled.
                ui_defaults["top_p"] = 0.9
                ui_defaults["top_k"] = 0
            else:
                ui_defaults.setdefault("top_p", 0.9)
                ui_defaults.setdefault("top_k", 0)
            return
        if ui_defaults.get("sample_solver", "") in ("", None):
            legacy_scheduler = ui_defaults.get("scheduler_type", "")
            if legacy_scheduler in {"euler", "heun", "pingpong"}:
                ui_defaults["sample_solver"] = legacy_scheduler

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Lyrics prompt cannot be empty for ACE-Step."
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        if "A" in audio_prompt_type and inputs.get("audio_guide") is None:
            return "Reference audio is required for Only Lyrics or Remix modes."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if not _is_ace_step15(base_model_type):
            return None

        raw_custom_settings = inputs.get("custom_settings", None)
        if raw_custom_settings is None:
            return None
        if not isinstance(raw_custom_settings, dict):
            return "Custom settings must be a dictionary."

        canonical_custom_settings = {}
        for raw_key, raw_value in raw_custom_settings.items():
            canonical_key = ACE_STEP15_SETTING_ALIASES.get(_normalize_ace_setting_name(raw_key), _normalize_ace_setting_name(raw_key))
            if len(canonical_key) == 0:
                continue
            canonical_custom_settings[canonical_key] = raw_value

        validated_custom_settings = {}
        for setting_def in model_def.get("custom_settings", []):
            setting_id = _resolve_ace_setting_id(setting_def)
            raw_value = canonical_custom_settings.get(setting_id, None)
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                raw_value = raw_value.strip()
                if len(raw_value) == 0:
                    continue

            if setting_id == "bpm":
                try:
                    if isinstance(raw_value, bool):
                        raise ValueError()
                    if isinstance(raw_value, int):
                        bpm_value = raw_value
                    elif isinstance(raw_value, float):
                        if not raw_value.is_integer():
                            raise ValueError()
                        bpm_value = int(raw_value)
                    else:
                        bpm_as_float = float(str(raw_value).strip())
                        if not bpm_as_float.is_integer():
                            raise ValueError()
                        bpm_value = int(bpm_as_float)
                except Exception:
                    return f"Invalid BPM. {ACE_STEP_BPM_HINT}"
                if bpm_value < ACE_STEP_BPM_MIN or bpm_value > ACE_STEP_BPM_MAX:
                    return f"Invalid BPM. {ACE_STEP_BPM_HINT}"
                validated_custom_settings["bpm"] = bpm_value
                continue

            if setting_id == "timesignature":
                timesig_value = None
                if isinstance(raw_value, bool):
                    return f"Invalid Time Signature. {ACE_STEP_TIME_SIGNATURE_HINT}"
                if isinstance(raw_value, int):
                    timesig_value = raw_value
                elif isinstance(raw_value, float):
                    if not raw_value.is_integer():
                        return f"Invalid Time Signature. {ACE_STEP_TIME_SIGNATURE_HINT}"
                    timesig_value = int(raw_value)
                else:
                    time_text = str(raw_value).strip()
                    if len(time_text) == 0 or time_text.lower() in {"n/a", "na", "none"}:
                        timesig_value = None
                    else:
                        compact = time_text.replace(" ", "")
                        compact_lower = compact.lower()
                        if compact_lower in {"2/4", "3/4", "4/4", "6/8"}:
                            timesig_value = int(compact_lower.split("/", 1)[0])
                        elif compact in {"2", "3", "4", "6"}:
                            timesig_value = int(compact)
                        else:
                            return f"Invalid Time Signature. {ACE_STEP_TIME_SIGNATURE_HINT}"
                if timesig_value is not None and timesig_value not in ACE_STEP_TIME_SIGNATURE_VALUES:
                    return f"Invalid Time Signature. {ACE_STEP_TIME_SIGNATURE_HINT}"
                if timesig_value is not None:
                    validated_custom_settings["timesignature"] = timesig_value
                continue

            if setting_id == "keyscale":
                normalized_keyscale, keyscale_error = _normalize_keyscale_value(raw_value)
                if keyscale_error is not None:
                    return f"Invalid KeyScale. {keyscale_error}"
                if normalized_keyscale is not None:
                    validated_custom_settings["keyscale"] = normalized_keyscale
                continue

            if setting_id == "language":
                language_value = str(raw_value).strip().lower()
                if len(language_value) == 0:
                    continue
                if language_value not in ACE_STEP15_VALID_LANGUAGE_SET:
                    return f"Invalid Language code '{raw_value}'. Available codes: {ACE_STEP15_LANGUAGE_CODES_TEXT}"
                validated_custom_settings["language"] = language_value
                continue

        for key, value in canonical_custom_settings.items():
            if key in validated_custom_settings:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if len(value) == 0:
                    continue
            validated_custom_settings[key] = value

        inputs["custom_settings"] = validated_custom_settings if len(validated_custom_settings) > 0 else None
        return None
