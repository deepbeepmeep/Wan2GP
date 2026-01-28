import os

import gradio as gr

from shared.utils import files_locator as fl


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
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_PROMPT,
    }


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


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["chatterbox"]

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
            "--lora-dir-tts",
            type=str,
            default=None,
            help=f"Path to a directory that contains TTS settings (default: {os.path.join(lora_root, 'tts')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_tts", None) or os.path.join(lora_root, "tts")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_chatterbox_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
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
        profile=0,
        **kwargs,
    ):
        from .chatterbox.pipeline import ChatterboxPipeline

        ckpt_root = fl.get_download_location()
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

        defaults = {
            "audio_prompt_type": "A",
            "model_mode": "en",
        }
        for key, value in defaults.items():
            ui_defaults.setdefault(key, value)

        if settings_version < 2.44:
            ui_defaults["guidance_scale"] = 1.0

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
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
        if len(one_prompt) > 300:
            gr.Info(
                "It is recommended to use a prompt that has less than 300 characters,"
                " otherwise you may get unexpected results."
            )
