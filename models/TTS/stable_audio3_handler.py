import os
from importlib.util import find_spec

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

from .stable_audio3_prompt_enhancers import get_stable_audio3_prompt_enhancer


STABLE_AUDIO3_REPO_ID = "DeepBeepMeep/TTS"
STABLE_AUDIO3_SMALL = "stable_audio3_small"
STABLE_AUDIO3_MEDIUM = "stable_audio3_medium"
STABLE_AUDIO3_TEXT_ENCODER_FOLDER = "t5gemma-b-b-ul2"
STABLE_AUDIO3_TEXT_ENCODER_BF16 = "t5gemma-b-b-ul2_bf16.safetensors"
STABLE_AUDIO3_SMALL_CONFIG = "stable_audio3_small_config.json"
STABLE_AUDIO3_MEDIUM_CONFIG = "stable_audio3_medium_config.json"
STABLE_AUDIO3_SAME_S_WEIGHTS = "stable_audio3_same_s_bf16.safetensors"
STABLE_AUDIO3_SAME_L_WEIGHTS = "stable_audio3_same_l_bf16.safetensors"
STABLE_AUDIO3_TOKENIZER_FILES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
STABLE_AUDIO3_AUDIO_MODE_CHOICES = ["", "AE", "AI", "AC"]
STABLE_AUDIO3_AUDIO_MODE_LABELS = {
    "": "Text to audio",
    "AE": "Audio to audio edit",
    "AI": "Inpaint source audio",
    "AC": "Continue source audio",
}
STABLE_AUDIO3_SAMPLE_SOLVERS = [
    ("PingPong", "pingpong"),
    ("Euler", "euler"),
    ("DPM++", "dpmpp"),
    ("RK4", "rk4"),
]
STABLE_AUDIO3_CUSTOM_SETTINGS = [
    {"id": "inpaint_start_seconds", "label": "Inpaint Start Seconds", "name": "Inpaint Start Seconds", "type": "float", "default": 0.0, "min": 0.0, "inc": 0.1},
    {"id": "inpaint_end_seconds", "label": "Inpaint End Seconds", "name": "Inpaint End Seconds", "type": "float", "default": 10.0, "min": 0.0, "inc": 0.1},
]


def _root_checkpoint_path(filename):
    return fl.locate_file(filename, error_if_none=False) or filename


def _config_path(base_model_type):
    return os.path.join(os.path.dirname(__file__), "stable_audio3", "configs", _config_name(base_model_type))


def _asset_weights(base_model_type):
    return STABLE_AUDIO3_SAME_L_WEIGHTS if base_model_type == STABLE_AUDIO3_MEDIUM else STABLE_AUDIO3_SAME_S_WEIGHTS


def _config_name(base_model_type):
    return STABLE_AUDIO3_MEDIUM_CONFIG if base_model_type == STABLE_AUDIO3_MEDIUM else STABLE_AUDIO3_SMALL_CONFIG


def _max_duration(base_model_type):
    return 380 if base_model_type == STABLE_AUDIO3_MEDIUM else 120


def _duration_slider(base_model_type):
    return {"label": "Duration (seconds)", "min": 1, "max": _max_duration(base_model_type), "increment": 1, "default": 30}


def _custom_settings(base_model_type):
    return [dict(one, max=float(_max_duration(base_model_type))) for one in STABLE_AUDIO3_CUSTOM_SETTINGS]


def _mode_from_audio_prompt_type(audio_prompt_type):
    audio_prompt_type = str(audio_prompt_type or "").upper()
    if "A" not in audio_prompt_type:
        return "text"
    if "E" in audio_prompt_type:
        return "audio_to_audio"
    if "I" in audio_prompt_type:
        return "inpaint"
    if "C" in audio_prompt_type:
        return "continue"
    return "text"


def _flash_attention2_available():
    if find_spec("flash_attn") is None:
        return False
    try:
        import flash_attn
    except Exception:
        return False
    return callable(getattr(flash_attn, "flash_attn_varlen_func", None))


def _medium_flash_attention_error():
    return "Stable Audio 3 Medium requires Flash Attention 2 for SAME-L sliding-window attention. Install flash-attn or use Stable Audio 3 Small Music/SFX."


def _require_medium_flash_attention(base_model_type):
    if base_model_type == STABLE_AUDIO3_MEDIUM and not _flash_attention2_available():
        raise RuntimeError(_medium_flash_attention_error())


def _model_id(base_model_type, model_def):
    default_model_id = "medium" if base_model_type == STABLE_AUDIO3_MEDIUM else "small-music"
    return str((model_def or {}).get("stable_audio3_model_id") or default_model_id)


class family_handler:
    @staticmethod
    def query_supported_types():
        return [STABLE_AUDIO3_SMALL, STABLE_AUDIO3_MEDIUM]

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
        parser.add_argument("--lora-dir-stable-audio3-small", type=str, default=None, help=f"Path to Stable Audio 3 Small LoRAs (default: {os.path.join(lora_root, STABLE_AUDIO3_SMALL)})")
        parser.add_argument("--lora-dir-stable-audio3-medium", type=str, default=None, help=f"Path to Stable Audio 3 Medium LoRAs (default: {os.path.join(lora_root, STABLE_AUDIO3_MEDIUM)})")

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if base_model_type == STABLE_AUDIO3_MEDIUM:
            return getattr(args, "lora_dir_stable_audio3_medium", None) or os.path.join(lora_root, STABLE_AUDIO3_MEDIUM)
        return getattr(args, "lora_dir_stable_audio3_small", None) or os.path.join(lora_root, STABLE_AUDIO3_SMALL)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        prompt_enhancer_instructions, prompt_enhancer_max_tokens, prompt_enhancer_button_label = get_stable_audio3_prompt_enhancer(_model_id(base_model_type, model_def))
        return {
            "audio_only": True,
            "image_outputs": False,
            "sliding_window": False,
            "guidance_max_phases": 1,
            "image_prompt_types_allowed": "",
            "supports_early_stop": True,
            "profiles_dir": [base_model_type],
            "text_encoder_URLs": [build_hf_url(STABLE_AUDIO3_REPO_ID, STABLE_AUDIO3_TEXT_ENCODER_FOLDER, STABLE_AUDIO3_TEXT_ENCODER_BF16)],
            "text_encoder_folder": STABLE_AUDIO3_TEXT_ENCODER_FOLDER,
            "text_prompt_enhancer_instructions": prompt_enhancer_instructions,
            "text_prompt_enhancer_max_tokens": prompt_enhancer_max_tokens,
            "prompt_enhancer_button_label": prompt_enhancer_button_label,
            "prompt_enhancer_choices_allowed": ["T"],
            "inference_steps": True,
            "sample_solvers": STABLE_AUDIO3_SAMPLE_SOLVERS,
            "temperature": False,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "enabled_audio_lora": True,
            "audio_guide_label": "Source audio",
            "audio_scale_name": "Edit Noise Level",
            "audio_prompt_type_sources": {
                "selection": STABLE_AUDIO3_AUDIO_MODE_CHOICES,
                "labels": STABLE_AUDIO3_AUDIO_MODE_LABELS,
                "default": "",
                "label": "Source Audio",
                "letters_filter": "AEIC",
            },
            "duration_slider": _duration_slider(base_model_type),
            "custom_settings": _custom_settings(base_model_type),
            "compile": False,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return {
            "repoId": STABLE_AUDIO3_REPO_ID,
            "sourceFolderList": ["", STABLE_AUDIO3_TEXT_ENCODER_FOLDER],
            "fileList": [[_asset_weights(base_model_type)], STABLE_AUDIO3_TOKENIZER_FILES],
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
        _require_medium_flash_attention(base_model_type)

        from .stable_audio3.pipeline import StableAudio3Pipeline

        transformer_weights = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        config_path = _config_path(base_model_type)
        autoencoder_weights = _root_checkpoint_path(_asset_weights(base_model_type))
        tokenizer_dir = fl.locate_folder(STABLE_AUDIO3_TEXT_ENCODER_FOLDER)
        pipeline = StableAudio3Pipeline(
            transformer_weights,
            config_path,
            autoencoder_weights,
            text_encoder_filename,
            tokenizer_dir,
            model_id=_model_id(base_model_type, model_def),
            max_duration=_max_duration(base_model_type),
            dtype=dtype or torch.bfloat16,
        )

        prompt_conditioner = pipeline.model.conditioner.conditioners["prompt"] if "prompt" in pipeline.model.conditioner.conditioners else None
        pipe = {
            "transformer": pipeline.main_model,
            "codec": pipeline.model.pretransform,
        }
        if prompt_conditioner is not None and hasattr(prompt_conditioner, "model"):
            pipe["text_encoder"] = prompt_conditioner.model

        if save_quantized and transformer_weights:
            from wgp import save_quantized_model

            save_quantized_model(pipeline.main_model, model_type, transformer_weights, dtype or torch.bfloat16, config_path)

        return pipeline, {"pipe": pipe, "coTenantsMap": {}}

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("sample_solver", "pingpong")

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "prompt": "An anthemic pop rock instrumental with bright guitars, punchy drums, and a nostalgic festival chorus.",
                "duration_seconds": 30,
                "repeat_generation": 1,
                "video_length": 0,
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "negative_prompt": "poor quality, distorted, noisy",
                "sample_solver": "pingpong",
                "audio_scale": 0.9,
                "multi_prompts_gen_type": "FG",
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for Stable Audio 3."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type == STABLE_AUDIO3_MEDIUM and not _flash_attention2_available():
            return _medium_flash_attention_error()
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        mode = _mode_from_audio_prompt_type(audio_prompt_type)
        if mode in ("audio_to_audio", "inpaint", "continue") and inputs.get("audio_guide") is None:
            return "Stable Audio 3 source-audio modes require a source audio file."
        if mode == "audio_to_audio" and inputs.get("audio_scale") is not None:
            try:
                inputs["audio_scale"] = float(inputs["audio_scale"])
            except (TypeError, ValueError):
                return "Stable Audio 3 Edit Noise Level must be a number."
            if not 0 <= inputs["audio_scale"] <= 1:
                return "Stable Audio 3 Edit Noise Level must be between 0 and 1."
        custom_settings = inputs.get("custom_settings", None)
        if isinstance(custom_settings, dict):
            for key in ("inpaint_start_seconds", "inpaint_end_seconds"):
                value = custom_settings.get(key, None)
                if value is None or value == "":
                    continue
                try:
                    custom_settings[key] = float(value)
                except (TypeError, ValueError):
                    return f"Stable Audio 3 custom setting '{key}' must be a number."
        return None
