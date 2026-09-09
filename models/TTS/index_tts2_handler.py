import os
import re

import torch

from shared.mps import mps_device_or
from shared.utils import files_locator as fl

from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


INDEX_TTS2_REPO_ID = "DeepBeepMeep/TTS"
INDEX_TTS2_FOLDER = "index_tts2"
INDEX_TTS2_MAIN_GPT_FILENAME = "index_tts2_gpt_fp16.safetensors"
INDEX_TTS25_ARCHITECTURE = "index_tts25"
INDEX_TTS25_FOLDER = "index_tts25"
INDEX_TTS25_MAIN_GPT_FILENAME = "IndexTTS25_GPT_bf16.safetensors"
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
INDEX_TTS25_CUSTOM_SETTINGS = [
    {
        "id": "speech_speed",
        "label": "Speech Speed (0.5-2.0)",
        "name": "Speech Speed",
        "type": "float",
        "default": 1.0,
        "min": 0.5,
        "max": 2.0,
        "inc": 0.05,
    },
    {
        "id": "text_normalization",
        "label": "Text Normalization",
        "name": "Text Normalization",
        "type": "dropdown",
        "default": "Yes",
        "choices": [("Yes", "Yes"), ("No", "No")],
    },
]
INDEX_TTS25_ROOT_FILES = [
    "feat1.pt",
    "feat2.pt",
    "index_tts25_codec_fp32.safetensors",
    "index_tts25_s2mel_fp32.safetensors",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
    "campplus_cn_common.bin",
]


def _get_index_tts2_model_def(base_model_type):
    is_v25 = base_model_type == INDEX_TTS25_ARCHITECTURE
    model_def = {
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
        "profiles_dir": [base_model_type],
        "duration_slider": dict(INDEX_TTS2_DURATION_SLIDER),
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": dict(INDEX_TTS2_AUDIO_PROMPT_TYPES),
        "custom_settings": [one.copy() for one in (INDEX_TTS25_CUSTOM_SETTINGS if is_v25 else INDEX_TTS2_CUSTOM_SETTINGS)],
        "preserve_empty_prompt_lines": True,
        "pause_between_sentences": True,
        "audio_guide_label": "Speaker reference voice",
        "audio_guide2_label": "Speaker 2 voice / emotion reference (optional)",
        "alt_prompt": {
            "label": "Default Emotion Instruction (empty preserves emotion from the speaker reference; [] overrides per segment)",
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
    if is_v25:
        model_def.update({
            "model_modes": {
                "label": "Language",
                "choices": [("English", "EN"), ("Chinese", "ZH"), ("Chinese / English Mixed", "ZHEN"), ("Japanese", "JA"), ("Spanish", "ES"), ("Arabic", "AR")],
                "default": "EN",
            },
            "infos": (
                "### Improvements over IndexTTS 2.0\n\n"
                "- Supports **Chinese, English, Japanese, Spanish, and Arabic**.\n"
                "- Supports **cross-lingual voice cloning**.\n"
                "- Uses a faster **25 Hz semantic-token path**.\n"
                "- Adds native speech-speed control and Pinyin, CMU phoneme, and Kana pronunciation annotations.\n\n"
                "### Language\n\n"
                "Language is required by IndexTTS 2.5 because it conditions both the text prefix and a learned "
                "language embedding. Select the language matching the text; **EN** is the default.\n\n"
                "### Emotion control\n\n"
                "- Put an emotion instruction in square brackets immediately before the text it controls, for "
                "example: `[happy]` or `[calm and reassuring]`.\n"
                "- An inline emotion remains active until another bracketed instruction appears.\n"
                "- **Default Emotion Instruction** applies to segments without an inline emotion.\n"
                "- If neither is provided, WanGP preserves emotion from the speaker reference.\n\n"
                "### Speech Speed\n\n"
                "- **1.0** is normal speed.\n"
                "- Values above **1.0** speak faster.\n"
                "- Values below **1.0** speak slower.\n\n"
                "### Text Normalization\n\n"
                "Text Normalization expands supported written forms such as numbers and symbols before tokenization. "
                "Disable it when the prompt is already written exactly as it should be spoken or when testing precise "
                "pronunciation markup."
            ),
            "prompt_infos": (
                "Emotion example: [fear] I heard something outside. [calm] It was only the wind. "
                "Pronunciation syntax: Chinese <行|XING2>, English <minute|M IH1 . N AH0 T>, Japanese <上手|じょうず>."
            ),
        })
    return model_def


def _get_index_tts2_download_def(base_model_type):
    if base_model_type == INDEX_TTS25_ARCHITECTURE:
        qwen_files = [name for name in INDEX_TTS2_QWEN_EMO_FILES if name != "model.safetensors"] + ["model_int8_convrot.safetensors"]
        return {
            "repoId": INDEX_TTS2_REPO_ID,
            "sourceFolderList": [INDEX_TTS25_FOLDER, INDEX_TTS2_QWEN_EMO_FOLDER, INDEX_TTS2_BIGVGAN_FOLDER, INDEX_TTS2_W2V_BERT_FOLDER],
            "fileList": [INDEX_TTS25_ROOT_FILES, qwen_files, INDEX_TTS2_BIGVGAN_FILES, INDEX_TTS2_W2V_BERT_FILES],
        }
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
        return ["index_tts2", INDEX_TTS25_ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (2200, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-index-tts2",
            type=str,
            default=None,
            help=f"Path to a directory that contains IndexTTS2 settings (default: {os.path.join(lora_root, 'index_tts2')})",
        )
        parser.add_argument("--lora-dir-index-tts25", type=str, default=None, help=f"Path to IndexTTS 2.5 LoRAs (default: {os.path.join(lora_root, INDEX_TTS25_ARCHITECTURE)})")

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if base_model_type == INDEX_TTS25_ARCHITECTURE:
            return getattr(args, "lora_dir_index_tts25", None) or os.path.join(lora_root, INDEX_TTS25_ARCHITECTURE)
        return getattr(args, "lora_dir_index_tts2", None) or os.path.join(lora_root, "index_tts2")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_index_tts2_model_def(base_model_type)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_index_tts2_download_def(base_model_type)

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

        is_v25 = base_model_type == INDEX_TTS25_ARCHITECTURE
        main_gpt_filename = INDEX_TTS25_MAIN_GPT_FILENAME if is_v25 else INDEX_TTS2_MAIN_GPT_FILENAME
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
            gpt_weights_path = fl.locate_file(main_gpt_filename, error_if_none=False)
        if gpt_weights_path is not None:
            gpt_name = os.path.basename(gpt_weights_path)
            if "_quanto_" in gpt_name:
                non_quanto_name = gpt_name.replace("_quanto_fp16_int8", "_fp16").replace("_quanto_int8", "")
                non_quanto_path = fl.locate_file(non_quanto_name, error_if_none=False)
                if non_quanto_path is not None:
                    gpt_weights_path = non_quanto_path
        if gpt_weights_path is None:
            raise FileNotFoundError(
                f"IndexTTS main transformer file '{main_gpt_filename}' is missing. "
                "It must be provided in defaults model.URLs."
            )

        runtime_device = mps_device_or(torch.device("cpu"))
        pipeline = IndexTTS2Pipeline(
            ckpt_root=fl.get_download_location(),
            device=runtime_device,
            gpt_weights_path=gpt_weights_path,
            show_load_logs=INDEX_TTS2_SHOW_LOAD_LOGS,
            lm_decoder_engine=lm_decoder_engine,
            architecture=base_model_type,
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
        if is_v25:
            pipe["semantic_codec"] = pipeline.model.semantic_codec
        if str(lm_decoder_engine).strip().lower() in ("cg", "cudagraph", "vllm"):
            pipe["transformer"]._budget = 0

        load_def = {
            "pipe": pipe,
            "coTenantsMap": {},
        }
        if int(profile) in (2, 4, 5):
            load_def["budgets"] = {"transformer2": 250}

        if save_quantized and gpt_weights_path:
            from wgp import save_quantized_model

            gpt_config_path = os.path.join(pipeline.model_dir, "configs", "gpt_runtime_config.json")
            save_quantized_model(pipeline.model.gpt, model_type, gpt_weights_path, dtype or torch.float16, gpt_config_path, submodel_no=1)

        return pipeline, load_def

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""
        defaults = {
            "audio_prompt_type": "A",
        }
        if base_model_type == INDEX_TTS25_ARCHITECTURE:
            defaults["model_mode"] = "EN"
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
                "multi_prompts_gen_type": "FG",
            }
        )
        if base_model_type == INDEX_TTS25_ARCHITECTURE:
            ui_defaults.update({
                "model_mode": "EN",
                "prompt": "[fear] At the very beginning I was so afraid to speak.\n"
                "[sadness] Nobody would talk to me. I felt so alone.\n"
                "[disgust] They would just ignore me and pretend that I didnt exist\n"
                "[happy] By chance I discovered this wonderful App, and now everything is different.\n"
                "[anger] I have a new voice and now everybody will have no choice but to listen to my words !!!",
            })

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
                    "'multi_prompts_gen_type' = 'FG' so dialogue lines stay in one prompt."
                )
            speaker_ids = sorted({int(match.group(1)) for match in speaker_matches})
            if len(speaker_ids) != 2:
                return (
                    "Two-speaker mode requires exactly two speaker IDs. Use Speaker 1: and Speaker 2:. "
                    "For headless settings, keep 'multi_prompts_gen_type' = 'FG'."
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
        if base_model_type == INDEX_TTS25_ARCHITECTURE:
            if not isinstance(custom_settings, dict):
                return "IndexTTS 2.5 custom settings are missing."
            language = str(inputs.get("model_mode", "EN") or "EN").upper()
            if language not in ("EN", "ZH", "ZHEN", "JA", "ES", "AR"):
                return "Unsupported IndexTTS 2.5 language."
            try:
                speech_speed = float(custom_settings.get("speech_speed", 1.0))
            except Exception:
                return "Speech Speed must be a number between 0.5 and 2.0."
            if speech_speed < 0.5 or speech_speed > 2.0:
                return "Speech Speed must be between 0.5 and 2.0."
            normalization = str(custom_settings.get("text_normalization", "Yes"))
            if normalization not in ("Yes", "No"):
                return "Text Normalization must be Yes or No."
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
