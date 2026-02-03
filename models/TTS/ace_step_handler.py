import os

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

ACE_STEP_DURATION_SLIDER = {
    "label": "Duration (seconds)",
    "min": 5,
    "max": 240,
    "increment": 1,
    "default": 20,
}

def _get_model_path(model_def, key, default):
    if not model_def:
        return default
    value = model_def.get(key, default)
    return value or default

def _ace_step_ckpt_file(filename):
    rel_path = os.path.join(ACE_STEP_REPO_FOLDER, filename)
    return fl.locate_file(rel_path, error_if_none=False) or rel_path


def _ace_step_ckpt_dir(dirname):
    rel_path = os.path.join(ACE_STEP_REPO_FOLDER, dirname)
    return fl.locate_folder(rel_path, error_if_none=False) or rel_path


def _ckpt_dir(dirname):
    return fl.locate_folder(dirname, error_if_none=False) or dirname


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ace_step_v1"]

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

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_ace_step", None) or os.path.join(lora_root, "ace_step")

    @staticmethod
    def query_model_def(base_model_type, model_def):
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
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
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
                ACE_STEP_TEXT_ENCODER_FOLDER,
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
        **kwargs,
    ):
        from .ace_step.pipeline_ace_step import ACEStepPipeline

        transformer_weights = None
        if isinstance(model_filename, (list, tuple)):
            transformer_weights = model_filename[0] if model_filename else None
        else:
            transformer_weights = model_filename

        transformer_config = _get_model_path(model_def, "ace_step_transformer_config", _ace_step_ckpt_file(ACE_STEP_TRANSFORMER_CONFIG_NAME))
        dcae_weights = _get_model_path(model_def, "ace_step_dcae_weights", _ace_step_ckpt_file(ACE_STEP_DCAE_WEIGHTS_NAME))
        dcae_config = _get_model_path(model_def, "ace_step_dcae_config", _ace_step_ckpt_file(ACE_STEP_DCAE_CONFIG_NAME))
        vocoder_weights = _get_model_path(model_def, "ace_step_vocoder_weights", _ace_step_ckpt_file(ACE_STEP_VOCODER_WEIGHTS_NAME))
        vocoder_config = _get_model_path(model_def, "ace_step_vocoder_config", _ace_step_ckpt_file(ACE_STEP_VOCODER_CONFIG_NAME))
        text_encoder_folder = _get_model_path(model_def, "text_encoder_folder", ACE_STEP_TEXT_ENCODER_FOLDER)
        text_encoder_weights = text_encoder_filename or _get_model_path(
            model_def,
            "ace_step_text_encoder_weights",
            os.path.join(text_encoder_folder, ACE_STEP_TEXT_ENCODER_NAME),
        )
        tokenizer_dir = _get_model_path(
            model_def,
            "ace_step_tokenizer_dir",
            _ckpt_dir(text_encoder_folder),
        )

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
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "prompt": "[Verse]\\nNeon rain on the city line\\n"
                "You hum the tune and I fall in time\\n"
                "[Chorus]\\nHold me close and keep the time",
                "alt_prompt": "dreamy synth-pop, shimmering pads, soft vocals",
                "scheduler_type": "euler",
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
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Lyrics prompt cannot be empty for ACE-Step."
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        if "A" in audio_prompt_type and inputs.get("audio_guide") is None:
            return "Reference audio is required for Only Lyrics or Remix modes."
        return None
