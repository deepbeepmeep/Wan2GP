import os
import re
import warnings

import torch

from postprocessing import seedvc
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

from .ltx2_handler import _GEMMA_FILENAME, _GEMMA_FOLDER, _GEMMA_QUANTO_FILENAME
from .prompt_enhancer import SCENEMA_DIALOGUE_PROMPT, SCENEMA_SPEECH_PROMPT


SCENEMA_REPO_ID = "DeepBeepMeep/LTX-2"
SCENEMA_ASSET_DIR = ""
SCENEMA_MAIN_FILENAME = "scenema-audio-transformer_bf16.safetensors"
SCENEMA_QUANT_FILENAME = "scenema-audio-transformer_quanto_bf16_int8.safetensors"
LTX23_AUDIO_VAE_FILENAME = "ltx-2.3-22b_audio_vae.safetensors"
LTX23_VOCODER_FILENAME = "ltx-2.3-22b_vocoder.safetensors"
LTX23_TEXT_EMBEDDING_PROJECTION_FILENAME = "ltx-2.3-22b_text_embedding_projection.safetensors"
LTX23_EMBEDDINGS_CONNECTOR_FILENAME = "ltx-2.3-22b_embeddings_connector.safetensors"
SCENEMA_WHISPER_MEDIUM_REPO = "DeepBeepMeep/Wan2.1"
SCENEMA_WHISPER_MEDIUM_DIR = "whisper_medium"
SCENEMA_WHISPER_MEDIUM_FILES = ["config.json", "model.safetensors"]
SCENEMA_KOKORO_DIR = "kokoro"
SCENEMA_KOKORO_VOICE_DIR = "kokoro/voices"
SCENEMA_KOKORO_FILES = ["config.json", "kokoro-v1_0.pth"]
SCENEMA_KOKORO_VOICE_FILES = ["af_heart.pt"]
SCENEMA_DEFAULT_PACE = 1.5
SCENEMA_DEFAULT_DURATION_SECONDS = 120
SCENEMA_MAX_DURATION_SECONDS = 30 * 60
SCENEMA_DEFAULT_CUSTOM_SETTINGS = {
    "vc_steps": 25,
    "vc_cfg_rate": 0.5,
    "pace": SCENEMA_DEFAULT_PACE,
}
SCENEMA_CUSTOM_SETTINGS = [
    {
        "id": "vc_steps",
        "label": "SeedVC Steps (default 25)",
        "name": "SeedVC Steps",
        "type": "int",
        "default": SCENEMA_DEFAULT_CUSTOM_SETTINGS["vc_steps"],
    },
    {
        "id": "vc_cfg_rate",
        "label": "SeedVC CFG Rate (default 0.5)",
        "name": "SeedVC CFG Rate",
        "type": "float",
        "default": SCENEMA_DEFAULT_CUSTOM_SETTINGS["vc_cfg_rate"],
    },
    {
        "id": "pace",
        "label": "Pace (default 1.5)",
        "name": "Pace",
        "type": "float",
        "default": SCENEMA_DEFAULT_CUSTOM_SETTINGS["pace"],
        "min": 0.2,
        "max": 3.0,
    },
]
SCENEMA_TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "config_light.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
SCENEMA_INFOS = """
## WanGP Speech Format
For one speaker, write normal text and put one performance cue in square brackets before each sentence. WanGP converts each `[cue]` into a Scenema `<action>` block.

```text
[Soft, close to the microphone] The lights are already on, so we can start whenever you are ready.
[Brighter, reassuring] Take your time. I am not going anywhere.
```

Plain speech uses only the text and action cues. Use Scenema XML when a single-speaker prompt also needs explicit `voice`, `gender`, `scene`, `shot`, or `language` properties.

## WanGP Dialogue Format

Use `Speaker N:` blocks for dialogue. Every new speaker block becomes its own audio chunk.

```text
Speaker 1{voice="A tired older man, low gravelly voice", gender="male", scene="a quiet kitchen at night"}:
[Trying to stay calm] I left the light on because I knew you would come back.

Speaker 2{voice="A younger woman, controlled but shaken", gender="female"}:
[A guarded breath before speaking] You always say that like it fixes everything.

Speaker 1:
[Softer, almost apologetic] No. But it keeps the room from feeling empty.
```

Properties set in `{...}` are remembered for that speaker and reused when the same speaker appears again. Add a new `{...}` later to override them.

## Supported Properties

- `voice`: the main voice and delivery description. Include age, timbre, accent, energy, microphone distance, or emotional color here.
- `gender`: `male` or `female`. It complements `voice`; for stronger control, also mention the gender in `voice`.
- `scene`: acoustic or cinematic context, such as `a quiet office late at night`.
- `shot`: `closeup`, `wide`, or `scene`. `closeup` is best for speech-first TTS.
- `language`: language code such as `en`, `fr`, or `it`.
- `speaker`: XML-only speaker id, used as `<speak speaker="2">...`.

## Voice References

The voice dropdown uses SeedVC for references. `Speaker 1 reference using SeedVC` applies the first audio reference. `Two Speakers references using SeedVC` applies the first reference to Speaker 1 and the second reference to Speaker 2. Additional speakers are supported, but only the first two can use uploaded reference audio.
"""


def _get_scenema_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": False,
        "lock_inference_steps": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["scenema_audio"],
        "duration_slider": {
            "label": "Max Duration (seconds)",
            "min": 1,
            "max": SCENEMA_MAX_DURATION_SECONDS,
            "increment": 0.5,
            "default": SCENEMA_DEFAULT_DURATION_SECONDS,
        },
        "profile_type": "video",
        "preserve_empty_prompt_lines": True,
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": {
            "selection": ["", "A2", "AB2"],
            "labels": {
                "": "Text or <speak> XML",
                "A2": "Speaker 1 reference using SeedVC",
                "AB2": "Two Speakers references using SeedVC",
            },
            "letters_filter": "AB2",
            "custom_flags": {"2": "SeedVC"},
            "default": "",
        },
        "audio_guide_label": "Speaker 1 reference voice (optional for multi-speaker)",
        "audio_guide2_label": "Speaker 2 reference voice (optional)",
        "custom_settings": [one.copy() for one in SCENEMA_CUSTOM_SETTINGS],
        "infos": SCENEMA_INFOS,
        "prompt_description": "Speech text or Scenema <speak> XML",
        "text_prompt_enhancer_instructions": SCENEMA_SPEECH_PROMPT,
        "text_prompt_enhancer_instructions1": SCENEMA_DIALOGUE_PROMPT,
        "text_prompt_enhancer_max_tokens": 768,
        "text_prompt_enhancer_max_tokens1": 1024,
        "prompt_enhancer_def": {
            "selection": ["T", "T1"],
            "labels": {
                "T": "A Speech with Action Cues",
                "T1": "A Dialogue with Action Cues",
            },
            "default": "T",
        },
        "prompt_enhancer_button_label": "Write",
        "compile": False,
        "text_encoder_folder": _GEMMA_FOLDER,
        "text_encoder_URLs": [
            build_hf_url("DeepBeepMeep/LTX-2", _GEMMA_FOLDER, _GEMMA_FILENAME),
            build_hf_url("DeepBeepMeep/LTX-2", _GEMMA_FOLDER, _GEMMA_QUANTO_FILENAME),
        ],
        "dtype": "bf16",
    }


def _get_scenema_download_def():
    return [
        {
            "repoId": SCENEMA_REPO_ID,
            "sourceFolderList": [""],
            "fileList": [[LTX23_AUDIO_VAE_FILENAME, LTX23_VOCODER_FILENAME, LTX23_TEXT_EMBEDDING_PROJECTION_FILENAME, LTX23_EMBEDDINGS_CONNECTOR_FILENAME]],
        },
        {
            "repoId": "DeepBeepMeep/LTX-2",
            "sourceFolderList": [_GEMMA_FOLDER],
            "fileList": [SCENEMA_TOKENIZER_FILES],
        },
        {
            "repoId": SCENEMA_WHISPER_MEDIUM_REPO,
            "sourceFolderList": [SCENEMA_WHISPER_MEDIUM_DIR],
            "fileList": [SCENEMA_WHISPER_MEDIUM_FILES],
        },
        {
            "repoId": SCENEMA_REPO_ID,
            "sourceFolderList": [SCENEMA_KOKORO_DIR, SCENEMA_KOKORO_VOICE_DIR],
            "fileList": [SCENEMA_KOKORO_FILES, SCENEMA_KOKORO_VOICE_FILES],
        },
    ] + seedvc.query_download_def()


def _load_alignment_whisper():
    from shared.deepy.transcription import _load_whisper_medium

    alignment_whisper = _load_whisper_medium(torch.device("cpu"))
    alignment_heads = alignment_whisper.alignment_heads
    del alignment_whisper._buffers["alignment_heads"]
    object.__setattr__(alignment_whisper, "alignment_heads", alignment_heads)
    for module in alignment_whisper.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module._lock_dtype = torch.float32
    alignment_whisper._offload_hooks = ["transcribe"]
    alignment_whisper._model_dtype = torch.float16
    alignment_whisper.eval().requires_grad_(False)
    return alignment_whisper


def _load_kokoro_pipeline():
    from preprocessing.kokoro import KPipeline

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*")
            kokoro_pipeline = KPipeline(lang_code="a", device="cpu", repo_id=fl.locate_folder(SCENEMA_KOKORO_DIR))
    except Exception as exc:
        raise RuntimeError(f"Kokoro TTS is required for Scenema Audio duration estimation. Error: {exc}") from exc
    kokoro_model = getattr(kokoro_pipeline, "model", None)
    if kokoro_model is None:
        raise RuntimeError("Kokoro TTS is required for Scenema Audio duration estimation.")
    kokoro_model._model_dtype = torch.float32
    return kokoro_pipeline


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["scenema_audio"]

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
            "--lora-dir-scenema-audio",
            type=str,
            default=None,
            help=f"Path to a directory that contains Scenema Audio LoRAs (default: {os.path.join(lora_root, 'scenema_audio')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_scenema_audio", None) or os.path.join(lora_root, "scenema_audio")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_scenema_model_def()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_scenema_download_def()

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        inputs.update(
            {
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "audio_cfg_scale": 1.0,
                "alt_guidance_scale": 1.0,
                "alt_scale": 0.0,
            }
        )
        custom_settings = inputs.get("custom_settings", None)
        if isinstance(custom_settings, dict):
            vc_steps = custom_settings.get("vc_steps", SCENEMA_DEFAULT_CUSTOM_SETTINGS["vc_steps"])
            vc_cfg_rate = custom_settings.get("vc_cfg_rate", SCENEMA_DEFAULT_CUSTOM_SETTINGS["vc_cfg_rate"])
            pace = custom_settings.get("pace", SCENEMA_DEFAULT_CUSTOM_SETTINGS["pace"])
            try:
                if int(vc_steps) <= 0:
                    return "Scenema Audio SeedVC Steps must be greater than 0."
            except Exception:
                return "Scenema Audio SeedVC Steps must be an integer."
            try:
                if float(vc_cfg_rate) <= 0:
                    return "Scenema Audio SeedVC CFG Rate must be greater than 0."
            except Exception:
                return "Scenema Audio SeedVC CFG Rate must be a number."
            try:
                if float(pace) <= 0:
                    return "Scenema Audio pace must be greater than 0."
            except Exception:
                return "Scenema Audio pace must be a number."
            custom_settings["vc_steps"] = int(vc_steps)
            custom_settings["vc_cfg_rate"] = float(vc_cfg_rate)
            custom_settings["pace"] = float(pace)
            inputs["custom_settings"] = custom_settings
        return None

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for Scenema Audio."
        audio_prompt_type = str(inputs.get("audio_prompt_type", "") or "").upper()
        if "A" in audio_prompt_type and "B" not in audio_prompt_type and inputs.get("audio_guide") is None:
            return "Scenema Audio reference voice mode requires a reference audio file."
        if "B" in audio_prompt_type and re.search(r"(?is)(Speaker\s*\d+|<speak[^>]*\bspeaker\s*=)", str(one_prompt)) is None:
            return "Scenema Audio multi-speaker mode requires SpeakerN text or <speak speaker=\"N\"> XML blocks."
        return None

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        profile=0,
        **kwargs,
    ):
        from .scenema_audio import ScenemaAudioPipeline

        weights_path = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if not text_encoder_filename:
            raise ValueError("Scenema Audio requires the LTX2 Gemma text encoder.")
        audio_vae_path = fl.locate_file(LTX23_AUDIO_VAE_FILENAME)
        vocoder_path = fl.locate_file(LTX23_VOCODER_FILENAME)
        text_projection_path = fl.locate_file(LTX23_TEXT_EMBEDDING_PROJECTION_FILENAME)
        text_connector_path = fl.locate_file(LTX23_EMBEDDINGS_CONNECTOR_FILENAME)
        config_path = os.path.join(os.path.dirname(__file__), "configs", "ltx2_22b_config.json")
        alignment_whisper = _load_alignment_whisper()
        kokoro_pipeline = _load_kokoro_pipeline()
        kokoro_model = kokoro_pipeline.model
        seedvc_model = seedvc.get_model(dtype=torch.float16)
        seedvc_pipe = seedvc.get_pipe(profile_no=profile, model=seedvc_model)

        pipeline = ScenemaAudioPipeline(
            model_weights_path=weights_path,
            audio_vae_path=audio_vae_path,
            vocoder_path=vocoder_path,
            text_projection_path=text_projection_path,
            text_connector_path=text_connector_path,
            gemma_path=text_encoder_filename,
            config_path=config_path,
            alignment_whisper=alignment_whisper,
            kokoro_pipeline=kokoro_pipeline,
            seedvc=seedvc_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=dtype or torch.bfloat16,
        )
        pipe = {
            "transformer": pipeline.model,
            "text_encoder": pipeline.text_encoder,
            "text_embedding_projection": pipeline.text_embedding_projection,
            "text_embeddings_connector": pipeline.text_embeddings_connector,
            "audio_encoder": pipeline.audio_encoder,
            "audio_decoder": pipeline.audio_decoder,
            "vocoder": pipeline.vocoder,
            "alignment_whisper": alignment_whisper,
            "kokoro": kokoro_model,
        }
        pipe.update(seedvc_pipe)
        pipe = {"pipe": pipe, "coTenantsMap": seedvc.get_cotenants_map(seedvc_pipe)}

        if save_quantized and weights_path:
            from wgp import save_quantized_model

            quantized_transformer = getattr(pipeline.model, "velocity_model", pipeline.model)
            save_quantized_model(quantized_transformer, model_type, weights_path, dtype or torch.bfloat16, config_path)

        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        audio_prompt_type = str(ui_defaults.get("audio_prompt_type", "") or "").upper()
        ui_defaults["audio_prompt_type"] = "AB2" if "2" in audio_prompt_type and "B" in audio_prompt_type else "A2" if "2" in audio_prompt_type and "A" in audio_prompt_type else ""
        ui_defaults["alt_prompt"] = ""
        ui_defaults.setdefault("duration_seconds", model_def.get("duration_slider", {}).get("default", SCENEMA_DEFAULT_DURATION_SECONDS))
        custom_settings = ui_defaults.get("custom_settings", None)
        if not isinstance(custom_settings, dict):
            custom_settings = {}
        for key, value in SCENEMA_DEFAULT_CUSTOM_SETTINGS.items():
            custom_settings.setdefault(key, value)
        ui_defaults["custom_settings"] = custom_settings

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "alt_prompt": "",
                "repeat_generation": 1,
                "duration_seconds": model_def.get("duration_slider", {}).get("default", SCENEMA_DEFAULT_DURATION_SECONDS),
                "video_length": 0,
                "num_inference_steps": 8,
                "negative_prompt": "",
                "guidance_scale": 1.0,
                "custom_settings": dict(SCENEMA_DEFAULT_CUSTOM_SETTINGS),
                "multi_prompts_gen_type": "FG",
            }
        )
