import os

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from models.TTS.prompt_enhancers import MINIMAX_MUSIC3_CAPTION_PROMPT, MINIMAX_MUSIC3_LYRIC_PROMPT


ARCHITECTURE = "minimax_music3"
REPO_ID = "DeepBeepMeep/TTS"
PROJECT_FOLDER = "MiniMax-Music3"
TEXT_ENCODER_FOLDER = "MiniMaxMusic3-Qwen3"
TEXT_ENCODER_BF16 = "MiniMaxMusic3-Qwen3_bf16.safetensors"
TEXT_ENCODER_INT8 = "MiniMaxMusic3-Qwen3_int8_convrot.safetensors"

PROJECT_FILES = [
    "rvq_depth_decoder_config.json",
    "rvq_depth_decoder_bf16.safetensors",
    "rvq_depth_decoder_int8_convrot.safetensors",
    "condition_encoder_config.json",
    "condition_encoder_fp32.safetensors",
    "vocoder_config.json",
    "vocoder_fp32.safetensors",
    "scheduler_config.json",
]
TOKENIZER_FILES = ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]


def _asset_path(filename: str) -> str:
    return fl.locate_file(os.path.join(PROJECT_FOLDER, filename))


def _source_config(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"music": (2195, "Music"), "tts": (2200, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument("--lora-dir-minimax-music3", type=str, default=None, help=f"Path to MiniMax Music 3 LoRAs (default: {os.path.join(lora_root, ARCHITECTURE)})")

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_minimax_music3", None) or os.path.join(lora_root, ARCHITECTURE)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": "music",
            "audio_only": True,
            "image_outputs": False,
            "sliding_window": False,
            "guidance_max_phases": 1,
            "no_negative_prompt": True,
            "inference_steps": True,
            "temperature": False,
            "image_prompt_types_allowed": "",
            "supports_early_stop": True,
            "profiles_dir": [ARCHITECTURE],
            "text_encoder_URLs": [
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16),
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8),
            ],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "lm_engines": ["cg", "vllm"],
            "compile": False,
            "prompt_class": "Lyrics",
            "prompt_enhancer_button_label": "Enhance",
            "prompt_enhancer_def": {
                "selection": ["T1", "L2O", "B2O", "T1,L2O", "T1,B2O"],
                "labels": {
                    "T1": "Lyrics",
                    "L2O": "Music Description",
                    "B2O": "Music Description using Existing Lyrics and Description Directions",
                    "T1,L2O": "Lyrics and Music Description independently",
                    "T1,B2O": "Lyrics then Music Description using Enhanced Lyrics and Description Directions",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions1": MINIMAX_MUSIC3_LYRIC_PROMPT,
            "text_prompt_enhancer_max_tokens1": 1536,
            "text_prompt_enhancer_instructions2": MINIMAX_MUSIC3_CAPTION_PROMPT,
            "text_prompt_enhancer_max_tokens2": 1024,
            "alt_prompt_inherits_prompt_paragraphs": True,
            "alt_prompt": {
                "label": "Music Description",
                "placeholder": "Genre, mood, vocals, instrumentation, tempo, and arrangement.",
                "lines": 5,
            },
            "duration_slider": {
                "label": "Maximum Song Duration (seconds)",
                "min": 1,
                "max": 300,
                "increment": 1,
                "default": 30,
            },
            "infos": (
                "MiniMax Music 3 generates complete 44.1 kHz stereo songs from a music description and structured "
                "lyrics. The autoregressive stage may end before the selected maximum duration. Commercial use is "
                "subject to the MiniMax-Music3 Community License, including its attribution and revenue terms."
            ),
            "prompt_infos": (
                "Put section tags such as [verse], [chorus], [bridge], and [outro] on their own lines. "
                "Describe genre, mood, vocal character, instruments, tempo, and arrangement in Music Description."
            ),
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": REPO_ID,
            "sourceFolderList": [PROJECT_FOLDER, TEXT_ENCODER_FOLDER],
            "fileList": [PROJECT_FILES, TOKENIZER_FILES],
        }]

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
        from .pipeline import MiniMaxMusic3Pipeline

        flow_weights = model_filename[0]
        rvq_weights = {
            "bf16": _asset_path("rvq_depth_decoder_bf16.safetensors"),
            "int8": _asset_path("rvq_depth_decoder_int8_convrot.safetensors"),
        }
        asset_paths = {
            "text_encoder_config": _source_config("text_encoder_config.json"),
            "transformer_config": _source_config("transformer_config.json"),
            "rvq_config": _asset_path("rvq_depth_decoder_config.json"),
            "rvq_weights": rvq_weights["int8" if lm_decoder_engine in ("cg", "vllm") else "bf16"],
            "condition_config": _asset_path("condition_encoder_config.json"),
            "condition_weights": _asset_path("condition_encoder_fp32.safetensors"),
            "vocoder_config": _asset_path("vocoder_config.json"),
            "vocoder_weights": _asset_path("vocoder_fp32.safetensors"),
            "scheduler_dir": os.path.dirname(_asset_path("scheduler_config.json")),
            "tokenizer_dir": os.path.dirname(fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "tokenizer.json"))),
        }
        fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "tokenizer_config.json"))
        fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "chat_template.jinja"))

        dtype = dtype or torch.bfloat16
        pipeline = MiniMaxMusic3Pipeline(flow_weights, text_encoder_filename, asset_paths, dtype, lm_decoder_engine=lm_decoder_engine)
        if lm_decoder_engine in ("cg", "vllm"):
            pipeline.text_encoder._budget = 0
            pipeline.rvq_depth_decoder._budget = 0
        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(pipeline.transformer, model_type, flow_weights, dtype, asset_paths["transformer_config"], submodel_no=1)

        pipe = {
            "transformer": pipeline.transformer,
            "text_encoder": pipeline.text_encoder,
            "rvq_depth_decoder": pipeline.rvq_depth_decoder,
            "condition_encoder": pipeline.condition_encoder,
            "vocoder": pipeline.vocoder,
        }
        offload_def = {
            "pipe": pipe,
            "coTenantsMap": {
                "transformer": ["condition_encoder"],
                "text_encoder": ["rvq_depth_decoder"],
                "rvq_depth_decoder": ["text_encoder"],
                "condition_encoder": ["transformer"],
            },
        }
        if int(profile) in (2, 4, 5):
            offload_def["budgets"] = {"transformer": 400, "text_encoder": 800}
        return pipeline, offload_def

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("alt_prompt", "")
        ui_defaults.setdefault("audio_prompt_type", "")

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "audio_prompt_type": "",
            "prompt": "[Verse]\nNeon rain on the city line\n"
            "You hum the tune and I fall in time\n"
            "[Chorus]\nHold me close and keep the time",
            "alt_prompt": "dreamy synth-pop, shimmering pads, soft vocals",
            "repeat_generation": 1,
            "duration_seconds": 30,
            "video_length": 0,
            "num_inference_steps": 30,
            "guidance_scale": 1.7,
            "negative_prompt": "",
            "multi_prompts_gen_type": "FG",
        })

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if not str(one_prompt or "").strip():
            return "Lyrics cannot be empty for MiniMax Music 3."
        if not str(inputs.get("alt_prompt") or "").strip():
            return "Music Description cannot be empty for MiniMax Music 3."
        if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
            return "MiniMax Music 3 does not support reference audio."
        return None
