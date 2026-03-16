import os
import re

import torch

from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


INDEX_TTS2_REPO_ID = "DeepBeepMeep/TTS"
INDEX_TTS2_FOLDER = "index_tts2"
INDEX_TTS2_MAIN_GPT_FILENAME = "index_tts2_gpt_fp16.safetensors"
INDEX_TTS2_QWEN_EMO_FOLDER = "qwen0.6bemo4-merge"
INDEX_TTS2_BIGVGAN_FOLDER = "bigvgan_v2_22khz_80band_256x"
INDEX_TTS2_W2V_BERT_FOLDER = "w2v-bert-2.0"
INDEX_TTS2_BIGVGAN_FILES = [
    "config.json",
    "bigvgan_generator.pt",
]
INDEX_TTS2_W2V_BERT_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model_fp16.safetensors",
]
INDEX_TTS2_SHOW_LOAD_LOGS = False
INDEX_TTS2_ROOT_FILES = [
    "bpe.model",
    "feat1.pt",
    "feat2.pt",
    "s2mel.safetensors",
    "wav2vec2bert_stats.pt",
    "campplus_cn_common.bin",
    "index_tts2_semantic_codec.safetensors",
]
INDEX_TTS2_QWEN_EMO_FILES = [
    "Modelfile",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
INDEX_TTS2_DURATION_SLIDER = {
    "label": "Max duration (seconds)",
    "min": 1,
    "max": 600,
    "increment": 1,
    "default": 25,
}
INDEX_TTS2_AUDIO_PROMPT_TYPES = {
    "selection": ["A", "AB", "AB2"],
    "labels": {
        "A": "Voice cloning (1 reference audio)",
        "AB": "Voice + emotion (2 reference audios)",
        "AB2": "Dialogue (2 speaker reference audios)",
    },
    "letters_filter": "AB2",
    "default": "A",
}
INDEX_TTS2_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
INDEX_TTS2_AUTO_SPLIT_MIN_SECONDS = 5.0
INDEX_TTS2_AUTO_SPLIT_MAX_SECONDS = 90.0
INDEX_TTS2_CUSTOM_SETTINGS = []


def _get_index_tts2_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": True,
        "top_p_slider": True,
        "top_k_slider": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["index_tts2"],
        "duration_slider": dict(INDEX_TTS2_DURATION_SLIDER),
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": dict(INDEX_TTS2_AUDIO_PROMPT_TYPES),
        "custom_settings": [one.copy() for one in INDEX_TTS2_CUSTOM_SETTINGS],
        "preserve_empty_prompt_lines": True,
        "pause_between_sentences": True,
        "audio_guide_label": "Speaker reference voice",
        "audio_guide2_label": "Speaker 2 voice / emotion reference (optional)",
        "alt_prompt": {
            "label": "Default Emotion Instruction (if none, emotion will be detected or set manually for each sentence)",
            "name": "Default Emotion Instruction",
            "placeholder": "happy,angry,sad,afraid,disgusted,melancholic,surprised,calm",
            "lines": 2,
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
        "lm_engines": ["legacy", "cg", "vllm"],
        "compile": False,
    }


def _get_index_tts2_download_def():
    return {
        "repoId": INDEX_TTS2_REPO_ID,
        # IndexTTS2 configs are bundled with source code in models/TTS/index_tts2/configs.
        "sourceFolderList": [
            INDEX_TTS2_FOLDER,
            INDEX_TTS2_QWEN_EMO_FOLDER,
            INDEX_TTS2_BIGVGAN_FOLDER,
            INDEX_TTS2_W2V_BERT_FOLDER,
        ],
        "fileList": [
            INDEX_TTS2_ROOT_FILES,
            INDEX_TTS2_QWEN_EMO_FILES,
            INDEX_TTS2_BIGVGAN_FILES,
            INDEX_TTS2_W2V_BERT_FILES,
        ],
    }


def _resolve_w2v_bert_dir():
    located = fl.locate_folder(INDEX_TTS2_W2V_BERT_FOLDER, error_if_none=False)
    if located is not None:
        return located
    fallback = os.path.join(fl.get_download_location(), INDEX_TTS2_W2V_BERT_FOLDER)
    if os.path.isdir(fallback):
        return fallback
    return None


def _ensure_w2v_bert_fp16_file():
    w2v_dir = _resolve_w2v_bert_dir()
    if w2v_dir is None:
        raise FileNotFoundError(
            f"IndexTTS2 semantic folder '{INDEX_TTS2_W2V_BERT_FOLDER}' is missing. "
            "WanGP must download it from DeepBeepMeep/TTS."
        )
    # fp32_path = os.path.join(w2v_dir, "model.safetensors")
    # if not os.path.isfile(fp32_path):
    #     raise FileNotFoundError(
    #         f"IndexTTS2 semantic model file is missing at '{fp32_path}'. "
    #         "Expected DeepBeepMeep/TTS/w2v-bert-2.0/model.safetensors."
    #     )
    fp16_path = os.path.join(w2v_dir, "model_fp16.safetensors")
    if os.path.isfile(fp16_path):
        return fp16_path
    from mmgp import offload

    src = safetensors2.l .load_ (fp16_path, device="cpu")
    # dst = {}
    # for key, value in src.items():
    #     if torch.is_floating_point(value) and value.dtype != torch.float16:
    #         dst[key] = value.to(torch.float16).contiguous()
    #     else:
    #         dst[key] = value.contiguous()
    # save_file(dst, fp16_path, metadata={"format": "pt", "dtype": "float16"})
    return fp16_path


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["index_tts2"]

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
            "--lora-dir-index-tts2",
            type=str,
            default=None,
            help=f"Path to a directory that contains IndexTTS2 settings (default: {os.path.join(lora_root, 'index_tts2')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_index_tts2", None) or os.path.join(lora_root, "index_tts2")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_index_tts2_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_index_tts2_download_def()

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
        from .index_tts2.pipeline import IndexTTS2Pipeline

        # _ensure_w2v_bert_fp16_file()
        weights_candidate = None
        if isinstance(model_filename, (list, tuple)):
            if len(model_filename) > 0:
                weights_candidate = model_filename[0]
        else:
            weights_candidate = model_filename
        gpt_weights_path = None
        if weights_candidate:
            gpt_weights_path = fl.locate_file(weights_candidate, error_if_none=False) or weights_candidate
        if gpt_weights_path is None:
            gpt_weights_path = fl.locate_file(INDEX_TTS2_MAIN_GPT_FILENAME, error_if_none=False)
        if gpt_weights_path is not None:
            gpt_name = os.path.basename(gpt_weights_path)
            if "_quanto_" in gpt_name:
                non_quanto_name = gpt_name.replace("_quanto_fp16_int8", "_fp16").replace("_quanto_int8", "")
                non_quanto_path = fl.locate_file(non_quanto_name, error_if_none=False)
                if non_quanto_path is not None:
                    gpt_weights_path = non_quanto_path
        if gpt_weights_path is None:
            raise FileNotFoundError(
                f"IndexTTS2 main transformer file '{INDEX_TTS2_MAIN_GPT_FILENAME}' is missing. "
                "It must be provided in defaults model.URLs."
            )

        runtime_device = torch.device("cpu")
        pipeline = IndexTTS2Pipeline(
            ckpt_root=fl.get_download_location(),
            device=runtime_device,
            gpt_weights_path=gpt_weights_path,
            show_load_logs=INDEX_TTS2_SHOW_LOAD_LOGS,
            lm_decoder_engine=lm_decoder_engine,
        )
        if torch.cuda.is_available():
            pipeline.model.device = "cuda:0"

        pipe = {
            "transformer": pipeline.model.gpt,
            "transformer2": pipeline.model.s2mel,
            "vocoder": pipeline.model.bigvgan,
            "semantic_model": pipeline.model.semantic_model,
            "campplus_model": pipeline.model.campplus_model,
            "qwen_emo_model": pipeline.model.qwen_emo.model,
        }
        if str(lm_decoder_engine).strip().lower() in ("cg", "cudagraph", "vllm"):
            pipe["transformer"]._budget = 0

        load_def = {
            "pipe": pipe,
            "coTenantsMap": {},
        }
        if int(profile) in (2, 4, 5):
            load_def["budgets"] = {"transformer2": 250}

        if save_quantized and gpt_weights_path:
            from mmgp import offload

            quant_filename = os.path.basename(gpt_weights_path)
            if "quanto" not in quant_filename:
                if "_fp16" in quant_filename:
                    quant_filename = quant_filename.replace("_fp16", "_quanto_fp16_int8")
                else:
                    dot_pos = quant_filename.rfind(".")
                    if dot_pos >= 0:
                        quant_filename = f"{quant_filename[:dot_pos]}_quanto_int8{quant_filename[dot_pos:]}"
                    else:
                        quant_filename = f"{quant_filename}_quanto_int8.safetensors"
            if fl.locate_file(quant_filename, error_if_none=False) is None:
                quant_path = os.path.join(fl.get_download_location(), quant_filename)
                offload.save_model(pipeline.model.gpt, quant_path, do_quantize=True, config_file_path=None)

        return pipeline, load_def

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""
        defaults = {
            "audio_prompt_type": "A",
        }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        ui_defaults.update(
            {
                "prompt": "[fear] At the very beginning I was so afraid to speak.\n[sadness] Nobody would talk to me. I felt so alone.\n[disgust] They would just ignore me and pretend that I didnt exist\n[happy] By chance I discovered this wonderful App, and now everything is different.\n[anger] I have a new voice and now everybody will have no choice but to listen to my words !!!",
                "audio_prompt_type": "A",
                "alt_prompt": "",
                "repeat_generation": 1,
                "duration_seconds": duration_def.get("default", 25),
                "pause_seconds": 0.2,
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 30,
                "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for IndexTTS2."
        if inputs.get("audio_guide") is None:
            return "IndexTTS2 requires one reference voice audio file."
        raw_audio_prompt_type = str(inputs.get("audio_prompt_type", "A") or "A").upper()
        if "2" in raw_audio_prompt_type:
            audio_prompt_type = "2"
        elif "B" in raw_audio_prompt_type:
            audio_prompt_type = "AB"
        elif "A" in raw_audio_prompt_type:
            audio_prompt_type = "A"
        else:
            return "Unsupported audio prompt mode for IndexTTS2."
        prompt_text = str(one_prompt)
        has_speaker_syntax = re.search(r"Speaker\s*\d+\s*:", prompt_text, flags=re.IGNORECASE) is not None
        if audio_prompt_type == "AB" and inputs.get("audio_guide2") is None:
            return "Emotion mode requires a second reference audio file."
        if audio_prompt_type == "2":
            if inputs.get("audio_guide2") is None:
                return "Two-speaker mode requires a second speaker reference audio file."
            speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:", prompt_text, flags=re.IGNORECASE))
            if not speaker_matches:
                return (
                    "Two-speaker mode requires prompt lines using Speaker 1: and Speaker 2: "
                    "(or any two numeric speaker IDs). For headless settings, keep "
                    "'multi_prompts_gen_type' = 2 so dialogue lines stay in one prompt."
                )
            speaker_ids = sorted({int(match.group(1)) for match in speaker_matches})
            if len(speaker_ids) != 2:
                return (
                    "Two-speaker mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:. "
                    "For headless settings, keep 'multi_prompts_gen_type' = 2."
                )
        elif has_speaker_syntax:
            return "Speaker-tag dialogue requires two-speaker mode (set audio prompt mode to Dialogue)."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        duration = inputs.get("duration_seconds", 0)
        try:
            duration = float(duration)
        except Exception:
            return "Max duration must be a number."
        if duration <= 0:
            return "Max duration must be greater than 0."
        custom_settings = inputs.get("custom_settings", None)
        if custom_settings is None:
            return None
        if not isinstance(custom_settings, dict):
            return "Custom settings must be a dictionary."
        raw_value = custom_settings.get(INDEX_TTS2_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                custom_settings.pop(INDEX_TTS2_AUTO_SPLIT_SETTING_ID, None)
                inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
                return None
        try:
            if isinstance(raw_value, bool):
                raise ValueError()
            auto_split_seconds = float(raw_value)
        except Exception:
            return (
                f"Auto Split Every s must be a number between "
                f"{int(INDEX_TTS2_AUTO_SPLIT_MIN_SECONDS)} and {int(INDEX_TTS2_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )
        if (
            auto_split_seconds < INDEX_TTS2_AUTO_SPLIT_MIN_SECONDS
            or auto_split_seconds > INDEX_TTS2_AUTO_SPLIT_MAX_SECONDS
        ):
            return (
                f"Auto Split Every s must be between "
                f"{int(INDEX_TTS2_AUTO_SPLIT_MIN_SECONDS)} and {int(INDEX_TTS2_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )
        custom_settings[INDEX_TTS2_AUTO_SPLIT_SETTING_ID] = auto_split_seconds
        inputs["custom_settings"] = custom_settings
        return None
