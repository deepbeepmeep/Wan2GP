import os

import gradio as gr
import torch

from shared.utils import files_locator as fl
from .qwen3 import defs as qwen3_defs


_FALLBACK_SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def _get_supported_languages() -> dict:
    try:
        from .chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    except Exception:
        return _FALLBACK_SUPPORTED_LANGUAGES
    return SUPPORTED_LANGUAGES


def _get_language_choices() -> list[tuple[str, str]]:
    languages = _get_supported_languages()
    return [
        (f"{name} ({code})", code)
        for code, name in sorted(languages.items(), key=lambda item: item[1])
    ]

HEARTMULA_VERSION = "3B"

YUE_STAGE1_COT_REPO = "m-a-p/YuE-s1-7B-anneal-en-cot"
YUE_STAGE1_ICL_REPO = "m-a-p/YuE-s1-7B-anneal-en-icl"
YUE_STAGE2_REPO = "m-a-p/YuE-s2-1B-general"
YUE_STAGE1_FILES = [
    "config.json",
]
YUE_STAGE2_FILES = [
    "config.json",
]


def _get_heartmula_local_config_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "HeartMula",
        "config",
        f"heartmula_{HEARTMULA_VERSION}.json",
    )


def _get_chatterbox_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "image_prompt_types_allowed": "",
        "profiles_dir": ["chatterbox"],
        "audio_guide_label": "Voice to Replicate",
        "model_modes": {
            "choices": _get_language_choices(),
            "default": "en",
            "label": "Language",
        },
        "any_audio_prompt": True,
        "chatterbox_controls": True,
        "text_prompt_enhancer_instructions": qwen3_defs.TTS_MONOLOGUE_PROMPT,
    }


def _get_heartmula_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 1,
        "no_negative_prompt": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["heartmula_oss_3b"],
        "alt_prompt": {
            "label": "Keywords / Tags",
            "placeholder": "piano,happy,wedding",
            "lines": 2,
        },
        "text_prompt_enhancer_instructions": (
            "You are a lyric-writing assistant. Generate a clean song lyric prompt "
            "for a text-to-song model. Output only the lyric text with optional "
            "section headers in square brackets (e.g., [Verse], [Chorus], [Bridge], "
            "[Intro], [Outro]). Do not include explanations, bullet lists, or tags. "
            "Keep a consistent theme, POV, and rhyme or rhythm where natural. Use "
            "short lines that are easy to sing.\n\n"
            "Example:\n"
            "[Verse]\n"
            "Morning light through the window pane\n"
            "I hum a tune to chase the rain\n"
            "Steady steps on a quiet street\n"
            "Heart and rhythm, gentle beat\n"
        ),
        "duration_slider": {
            "label": "Duration of the Song (in seconds)",
            "min": 30,
            "max": 240,
            "increment": 0.1,
            "default": 120,
        },
        "top_k_slider": True,
        "heartmula_cfg_scale": 1.5,
        "heartmula_topk": 50,
        "heartmula_max_audio_length_ms": 120000,
        "heartmula_codec_guidance_scale": 1.25,
        "heartmula_codec_steps": 10,
        "heartmula_codec_version": "",
        "compile": False, #["transformer", "transformer2"]
    }


def _get_yue_model_def(model_def):
    use_audio_prompt = bool(model_def.get("yue_audio_prompt", False))
    yue_def = {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "image_prompt_types_allowed": "",
        "profiles_dir": ["yue"],
        "alt_prompt": {
            "label": "Genres / Tags",
            "placeholder": "pop, dreamy, warm vocal, female, nostalgic",
            "lines": 2,
        },
        "yue_max_new_tokens": 3000,
        "yue_run_n_segments": 2,
        "yue_stage2_batch_size": 4,
        "yue_segment_duration": 6,
        "yue_prompt_start_time": 0.0,
        "yue_prompt_end_time": 30.0,
    }
    if use_audio_prompt:
        yue_def.update(
            {
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "audio_guide_label": "Vocal prompt",
                "audio_guide2_label": "Instrumental prompt",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "AB"],
                    "labels": {
                        "": "Lyrics only",
                        "A": "Mixed audio prompt",
                        "AB": "Vocal + Instrumental prompts",
                    },
                    "letters_filter": "AB",
                    "default": "",
                },
            }
        )
    return yue_def


def _get_chatterbox_download_def():
    mandatory_files = [
        "ve.safetensors",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ]
    return {
        "repoId": "ResembleAI/chatterbox",
        "sourceFolderList": [""],
        "targetFolderList": ["chatterbox"],
        "fileList": [mandatory_files],
    }


def _get_heartmula_download_def(model_def):
    codec_version = (model_def or {}).get("heartmula_codec_version", "")
    codec_suffix = f"_{codec_version}" if codec_version else ""
    repo_id = "DeepBeepMeep/TTS"
    gen_files = [
        "gen_config.json",
        "tokenizer.json",
        f"codec_config{codec_suffix}.json",
        f"HeartMula_codec{codec_suffix}.safetensors",
    ]
    return [
        {
            "repoId": repo_id,
            "sourceFolderList": ["HeartMula"],
            "fileList": [gen_files],
        },
    ]


def _get_yue_download_def(model_def):
    use_audio_prompt = bool(model_def.get("yue_audio_prompt", False))
    stage1_repo = YUE_STAGE1_ICL_REPO if use_audio_prompt else YUE_STAGE1_COT_REPO
    stage1_folder = os.path.basename(stage1_repo)
    stage2_folder = os.path.basename(YUE_STAGE2_REPO)
    xcodec_root = "xcodec_mini_infer"
    xcodec_source_folders = [
        "final_ckpt",
        "decoders",
        "models",
        "modules",
        "quantization",
        "RepCodec",
        "descriptaudiocodec",
        "vocos",
        "semantic_ckpts/hf_1_325000",
    ]
    xcodec_files = [
        ["config.yaml", "ckpt_00360000.pth"],
        ["config.yaml", "decoder_131000.pth", "decoder_151000.pth"],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    return [
        {
            "repoId": stage1_repo,
            "sourceFolderList": [""],
            "targetFolderList": [stage1_folder],
            "fileList": [YUE_STAGE1_FILES],
        },
        {
            "repoId": YUE_STAGE2_REPO,
            "sourceFolderList": [""],
            "targetFolderList": [stage2_folder],
            "fileList": [YUE_STAGE2_FILES],
        },
        {
            "repoId": stage1_repo,
            "sourceFolderList": [""],
            "targetFolderList": ["mm_tokenizer_v0.2_hf"],
            "fileList": [["tokenizer.model"]],
        },
        {
            "repoId": "m-a-p/xcodec_mini_infer",
            "sourceFolderList": [""],
            "targetFolderList": [xcodec_root],
            "fileList": [["vocoder.py", "post_process_audio.py"]],
        },
        {
            "repoId": "m-a-p/xcodec_mini_infer",
            "sourceFolderList": xcodec_source_folders,
            "targetFolderList": [xcodec_root] * len(xcodec_source_folders),
            "fileList": xcodec_files,
        },
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [
            "chatterbox",
            "heartmula_oss_3b",
            "yue",
            *list(qwen3_defs.QWEN3_TTS_VARIANTS),
        ]

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
    def register_lora_cli_args(parser):
        parser.add_argument(
            "--lora-dir-tts",
            type=str,
            default=os.path.join("loras", "tts"),
            help="Path to a directory that contains TTS settings",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args):
        return args.lora_dir_tts

    @staticmethod
    def query_model_def(base_model_type, model_def):
        if base_model_type == "heartmula_oss_3b":
            return _get_heartmula_model_def()
        if base_model_type == "yue":
            return _get_yue_model_def(model_def)
        if base_model_type in qwen3_defs.QWEN3_TTS_VARIANTS:
            return qwen3_defs.get_qwen3_model_def(base_model_type)
        return _get_chatterbox_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if base_model_type == "heartmula_oss_3b":
            return _get_heartmula_download_def(model_def or {})
        if base_model_type == "yue":
            return _get_yue_download_def(model_def or {})
        if base_model_type in qwen3_defs.QWEN3_TTS_VARIANTS:
            return qwen3_defs.get_qwen3_download_def(base_model_type)
        return _get_chatterbox_download_def()

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
        profile = 0,
        **kwargs,
    ):
        ckpt_root = fl.get_download_location()
        if base_model_type == "heartmula_oss_3b":
            from .HeartMula.pipeline import HeartMuLaPipeline

            weights_candidate = None
            if isinstance(model_filename, (list, tuple)):
                if len(model_filename) > 0:
                    weights_candidate = model_filename[0]
            else:
                weights_candidate = model_filename
            heartmula_weights_path = None
            if weights_candidate:
                heartmula_weights_path = fl.locate_file(
                    weights_candidate, error_if_none=False
                )
                if heartmula_weights_path is None:
                    heartmula_weights_path = weights_candidate
            pipeline = HeartMuLaPipeline(
                ckpt_root=ckpt_root,
                device=torch.device("cpu"),
                version=HEARTMULA_VERSION,
                VAE_dtype = VAE_dtype,
                heartmula_weights_path=heartmula_weights_path,
                cfg_scale=model_def.get("heartmula_cfg_scale", 1.5),
                topk=model_def.get("heartmula_topk", 50),
                max_audio_length_ms=model_def.get("heartmula_max_audio_length_ms", 120000),
                codec_steps=model_def.get("heartmula_codec_steps", 10),
                codec_guidance_scale=model_def.get(
                    "heartmula_codec_guidance_scale", 1.25
                ),
                codec_version=model_def.get("heartmula_codec_version", ""),
            )
          
            pipeline.mula.decoder[0].layers._compile_me = False
            pipeline.mula.backbone.layers._compile_me = False
            pipe = {"transformer": pipeline.mula, "transformer2": pipeline.mula.decoder[0], "codec": pipeline.codec}
            pipe = { "pipe": pipe, "coTenantsMap" : { 
                            "transformer": ["transformer2"],
                            "transformer2": ["transformer"],                             
                        }
            }

            if int(profile) in (2,4,5):
                pipe["budgets"] = { "transformer2" : 200} 
 
            return pipeline, pipe

        if base_model_type == "yue":
            from .yue.pipeline import YuePipeline

            if isinstance(model_filename, list):
                stage1_weights = model_filename[0] if len(model_filename) > 0 else ""
                stage2_weights = model_filename[1] if len(model_filename) > 1 else ""
            else:
                stage1_weights = model_filename or ""
                stage2_weights = ""

            pipeline = YuePipeline(
                stage1_weights_path=stage1_weights,
                stage2_weights_path=stage2_weights,
                use_audio_prompt=bool(model_def.get("yue_audio_prompt", False)),
                max_new_tokens=model_def.get("yue_max_new_tokens", 200),
                run_n_segments=model_def.get("yue_run_n_segments", 1),
                stage2_batch_size=model_def.get("yue_stage2_batch_size", 10),
                segment_duration=model_def.get("yue_segment_duration", 6),
                prompt_start_time=model_def.get("yue_prompt_start_time", 0.0),
                prompt_end_time=model_def.get("yue_prompt_end_time", 30.0),
            )
 
            pipe = {
                "transformer": pipeline.model_stage1,
                "transformer2": pipeline.model_stage2,
                "codec_model": pipeline.codec_model,
                "vocoder_vocal": pipeline.vocoder_vocal,
                "vocoder_inst": pipeline.vocoder_inst,
            }
            return pipeline, pipe

        if base_model_type in qwen3_defs.QWEN3_TTS_VARIANTS:
            from .qwen3.pipeline import Qwen3TTSPipeline

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

        from .chatterbox.pipeline import ChatterboxPipeline

        pipeline = ChatterboxPipeline(ckpt_root=ckpt_root, device="cpu")
        pipe = {
            "ve": pipeline.model.ve,
            "s3gen": pipeline.model.s3gen,
            "t3": pipeline.model.t3,
            "conds": pipeline.model.conds,
        }
        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if "alt_prompt" not in ui_defaults:
            ui_defaults["alt_prompt"] = ""

        if base_model_type == "heartmula_oss_3b":
            defaults = {
                "audio_prompt_type": "",
            }
        elif base_model_type == "yue":
            defaults = {
                "audio_prompt_type": "",
            }
        elif base_model_type == "qwen3_tts_customvoice":
            speakers = qwen3_defs.get_qwen3_speakers(base_model_type)
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
                "audio_prompt_type": "A",
                "model_mode": "en",
            }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)
        if settings_version < 2.44:
            if base_model_type == "heartmula_oss_3b":
                ui_defaults["guidance_scale"] = model_def.get("heartmula_cfg_scale", 1.5)
                ui_defaults["top_k"] = model_def.get("heartmula_topk", 50)
            elif base_model_type in qwen3_defs.QWEN3_TTS_VARIANTS:
                if model_def.get("top_k_slider", False):
                    ui_defaults["top_k"] = 50
            elif base_model_type != "yue":
                ui_defaults["guidance_scale"] = 1.0

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == "heartmula_oss_3b":
            duration_def = model_def.get("duration_slider", {})
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "alt_prompt": "piano,happy,wedding",
                    "repeat_generation": 1,
                    "duration_seconds": duration_def.get("default", 120),
                    "video_length": 0,
                    "num_inference_steps": 0,
                    "negative_prompt": "",
                    "temperature": 1.0,
                    "guidance_scale": model_def.get("heartmula_cfg_scale", 1.5),
                    "top_k": model_def.get("heartmula_topk", 50),
	                "multi_prompts_gen_type": 2
                }
            )
            return
        if base_model_type == "yue":
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "alt_prompt": "pop, dreamy, warm vocal, female, nostalgic",
                    "repeat_generation": 1,
                    "video_length": 0,
                    "num_inference_steps": 0,
                    "negative_prompt": "",
                    "temperature": 1.0,
	                "multi_prompts_gen_type": 2,
                }
            )
            return

        if base_model_type == "qwen3_tts_customvoice":
            speakers = qwen3_defs.get_qwen3_speakers(base_model_type)
            default_speaker = speakers[0] if speakers else ""
            ui_defaults.update(
                {
                    "audio_prompt_type": "",
                    "model_mode": default_speaker,
                    "alt_prompt": "",
                    "duration_seconds": qwen3_defs.get_qwen3_duration_default(),
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
                    "duration_seconds": qwen3_defs.get_qwen3_duration_default(),
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
                    "duration_seconds": qwen3_defs.get_qwen3_duration_default(),
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

        ui_defaults.update(
            {
                "audio_prompt_type": "A",
                "model_mode": "en",
                "repeat_generation": 1,
                "video_length": 0,
                "num_inference_steps": 0,
                "negative_prompt": "",
                "exaggeration": 0.5,
                "temperature": 0.8,
                "pace": 0.5,
                "guidance_scale": 1.0,
	            "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if base_model_type == "heartmula_oss_3b":
            alt_prompt = inputs.get("alt_prompt", "")
            if alt_prompt is None or len(str(alt_prompt).strip()) == 0:
                return "Keywords prompt cannot be empty for HeartMuLa."
            if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
                return "HeartMuLa does not support reference audio yet."
            return None
        if base_model_type == "yue":
            if one_prompt is None or len(str(one_prompt).strip()) == 0:
                return "Lyrics prompt cannot be empty for Yue."
            alt_prompt = inputs.get("alt_prompt", "")
            if alt_prompt is None or len(str(alt_prompt).strip()) == 0:
                return "Genres prompt cannot be empty for Yue."
            audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
            if model_def.get("yue_audio_prompt", False):
                if "A" in audio_prompt_type:
                    if inputs.get("audio_guide") is None:
                        return "You must provide a vocal or mixed audio prompt for Yue ICL."
                    if "B" in audio_prompt_type and inputs.get("audio_guide2") is None:
                        return "You must provide an instrumental prompt for Yue ICL."
                    start_time = float(inputs.get("yue_prompt_start_time", model_def.get("yue_prompt_start_time", 0.0)))
                    end_time = float(inputs.get("yue_prompt_end_time", model_def.get("yue_prompt_end_time", 30.0)))
                    if start_time >= end_time:
                        return "Audio prompt start time must be less than end time."
                    if end_time - start_time > 30:
                        return "Audio prompt duration should not exceed 30 seconds."
                elif inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
                    return "Select an audio prompt type for Yue ICL or clear audio prompts."
            else:
                if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
                    return "Yue base model does not support audio prompts. Please use Yue ICL."
            return None

        if base_model_type == "qwen3_tts_customvoice":
            if one_prompt is None or len(str(one_prompt).strip()) == 0:
                return "Prompt text cannot be empty for Qwen3 CustomVoice."
            speaker = inputs.get("model_mode", "")
            if not speaker:
                return "Please select a speaker for Qwen3 CustomVoice."
            speakers = qwen3_defs.get_qwen3_speakers(base_model_type)
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

        if len(one_prompt) > 300:
            gr.Info(
                "It is recommended to use a prompt that has less than 300 characters,"
                " otherwise you may get unexpected results."
            )
