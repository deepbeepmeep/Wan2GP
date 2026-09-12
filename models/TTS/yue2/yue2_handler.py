import os
from pathlib import Path

from shared.utils import files_locator as fl
from .prompt_enhancers import LYRICS_SYSTEM_PROMPT, STYLE_SYSTEM_PROMPT


ARCHITECTURE = "yue2"
REPO_ID = "DeepBeepMeep/TTS"
TEXT_ENCODER_FOLDER = "YuE2_AR"
ASSETS = ["vae_config.json", "YuE2_VAE_bf16.safetensors"]
SCORING_CHECKPOINT = "SheetSage2_MERT2_bf16.safetensors"
PROMPT = "[Verse]\nMorning light across the bay\nWe watch the shadows drift away\n[Chorus]\nStay with me until the dawn\nLet our little song go on"
STYLE = "English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM"
INFOS = """**Turn your lyrics into a complete song** with a singing voice and accompaniment, in stereo.

- **Lyrics** supplies the words to sing. **Music Style** describes how the song should sound. Use both for your first song.
- **Melody and chords** is the recommended starting point: YuE2 plans the tune and its harmony before making the recording.
- **Melody only** plans the tune while giving the accompaniment more freedom. Use this when a supplied melody should keep its shape but the arrangement can change.
- **Direct generation** skips the written composition plan. Try it for an alternative interpretation of the same lyrics and style.

**Start with the defaults.** Write one or two short verses and a repeated chorus, choose a clear style, and allow enough time for the words. **Maximum Song Duration is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing the limit allows more time but does not force a longer song; a limit that is too short can cut it off. A very large limit is capped by the model's capacity, which also depends on the length of your lyrics and score. Change the seed to try another performance. More synthesis steps cost more time; 32 is the recommended starting point.

**Abort** cancels the generation without saving a partial song. **Early Stop** stops audio-token generation and renders what has already been composed.

**Optional prompt enhancer:** disabled by default. Choose Lyrics to turn an idea into singable words or tidy existing lyrics; choose Music Style to clarify the sound, or Lyrics then Music Style to prepare both. Review the result before generating. The enhancer does not edit your ABC score.

**Optional ABC score:** leave this empty unless you already have a compatible written melody or composition. ABC is a text format for musical notation, not a place for instructions such as “make it happier.” Supplying a score replaces the automatic plan. It requires Melody and chords or Melody only; melody-only scores must omit chord symbols. The supported score format uses Vocal and Ins voices. A score and lyrics that belong together give the model clearer guidance.

Changing the lyrics, style or score creates a **new recording**; it does not preserve the original voice or keep parts of an existing recording untouched. Choose **Source Audio > Extract Score from Source Song** to transcribe a recording with SheetSage2/MERT2. Melody and chords retains harmony; Melody only leaves the new accompaniment more freedom. Direct generation cannot use source audio. The manual ABC field is hidden while source audio is selected. You must still supply lyrics: transcription extracts musical notes, not sung words. Match the lyrics and section order to the source; transcription mistakes can affect the result. The model weights are **CC BY-NC 4.0 (non-commercial use)**.
"""
PROMPT_INFOS = """**Write lyrics, not a request to write them.** Put section labels on their own lines and leave a blank line between sections:

```text
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
```

Keep lines reasonably short and easy to sing. Repeat the actual chorus words when you want the chorus again; avoid long explanations between the lyrics.

If you want help writing lyrics from an idea, explicitly enable the Lyrics enhancer. With enhancement disabled, enter the lyrics themselves.

**Describe the sound in Music Style.** A useful description names the language, genre, main instruments, mood and kind of voice. Add a tempo if it matters:

```text
English acoustic pop, warm female vocal, fingerpicked guitar,
gentle drums, hopeful, 90 BPM
```

For a cover from source audio, enter the original lyrics separately and keep their verse/chorus order aligned with the source. The audio supplies notes, not a transcript of the words.

Start with a few compatible ideas. “Gentle acoustic ballad” and “aggressive fast metal” pull in different directions. For a different arrangement, keep the lyrics and change the style; for a different performance, change the seed. If the ending is cut off, allow more time or shorten the lyrics. A style prompt describes the character of a voice; it does not guarantee a particular singer's identity.
"""
DEEPY_INFOS = """**YuE2 song generation:** `prompt` = lyrics; `alt_prompt` = music style. Outputs 48 kHz stereo vocals and accompaniment.
- `model_mode`: **0** melody+chords (recommended), **1** melody only/free accompaniment, **2** direct generation.
- Leave `custom_settings.abc` empty normally. A supplied native ABC score replaces planning in modes 0/1; use Vocal/Ins voices and no chords for mode 1. Align the lyrics with the score.
- **`duration_seconds` is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing it does not force a longer song; too short can cut it off. Very large limits are capped by the remaining model context after lyrics/score. Start with 32 steps and guidance 1; change seed for another take.
- Abort cancels without audio. Early Stop renders existing audio tokens; during score planning it finishes the score then makes an approximately eight-second preview, capped by duration. Acoustic synthesis/decoding still finish; previews may end mid-phrase.
- Prompt enhancement is off by default: `T1` prepares lyrics, `L2O` prepares style, `T1,B2O` prepares lyrics then style informed by them. It does not modify ABC.
- `audio_prompt_type="A"` + `audio_guide` transcribes a source song into ABC with SheetSage2/MERT2 in modes 0/1, replacing any manual ABC. Empty audio mode uses automatic planning or manual ABC. Supply lyrics separately; notes are transcribed, not words. Regeneration creates a new recording; no waveform-preserving edits. **CC BY-NC 4.0 weights: non-commercial use.**
"""
DEEPY_PROMPT_INFOS = """**Lyrics:** actual short, singable lines with `[Verse]`, `[Chorus]`, `[Bridge]` on separate lines; blank lines between sections. Repeat chorus words explicitly.
**Music style:** language + genre + instruments + mood + vocal character, optionally tempo. Example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`.
Keep style instructions out of the lyrics; avoid conflicting styles. For a cover, supply source lyrics with section order and phrasing matching the recording. Leave the ABC field empty unless a compatible score is supplied; source audio overrides manual ABC. Increase duration or shorten lyrics if truncated; change seed for a new performance. A voice description does not guarantee a singer's identity.
"""


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
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, ARCHITECTURE)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": "music", "audio_only": True, "image_outputs": False, "sliding_window": False, "supports_early_stop": True,
            "guidance_max_phases": 1, "no_negative_prompt": True, "inference_steps": True,
            "temperature": True, "top_k_slider": True, "top_p_slider": True, "embedded_guidance": False,
            "image_prompt_types_allowed": "", "profiles_dir": [ARCHITECTURE], "compile": False,
            "lm_engines": ["cg", "vllm"], "prompt_class": "Lyrics", "prompt_enhancer_button_label": "Enhance",
            "text_encoder_URLs": [f"https://huggingface.co/{REPO_ID}/resolve/main/{TEXT_ENCODER_FOLDER}/YuE2_AR_{precision}.safetensors" for precision in ("bf16", "int8_convrot")],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "prompt_enhancer_def": {
                "selection": ["T1", "L2O", "B2O", "T1,B2O"],
                "labels": {"T1": "Lyrics", "L2O": "Music Style", "B2O": "Music Style using Existing Lyrics", "T1,B2O": "Lyrics then Music Style"},
                "default": "",
            },
            "text_prompt_enhancer_instructions1": LYRICS_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens1": 1536,
            "text_prompt_enhancer_instructions2": STYLE_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens2": 384,
            "alt_prompt_inherits_prompt_paragraphs": True,
            "alt_prompt": {"label": "Music Style", "placeholder": "Language, genre, instruments, mood, vocal character and tempo", "lines": 3},
            "model_modes": {"choices": [("Melody and chords", 0), ("Melody only", 1), ("Direct generation", 2)], "default": 0, "label": "Composition Planning"},
            "any_audio_prompt": True, "audio_prompt_choices": True, "audio_guide_label": "Source Song (Music to Transcribe)",
            "audio_prompt_type_sources": {"selection": ["", "A"], "labels": {"": "No Audio", "A": "Extract Score from Source Song"}, "default": "", "label": "Source Audio", "letters_filter": "A"},
            "custom_settings": [{"id": "abc", "name": "ABC Score", "label": "Optional ABC score (planning modes only)", "type": "text", "default": "", "audio_prompt_type_not": "A"}],
            "duration_slider": {"label": "Maximum Song Duration (seconds)", "name": "Maximum Song Duration", "min": 1, "max": 600, "increment": 1, "default": 120},
            "infos": INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO_ID, "sourceFolderList": ["yue2", TEXT_ENCODER_FOLDER, "sheetsage2"], "fileList": [ASSETS, ["qwen.tiktoken"], [SCORING_CHECKPOINT]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, dtype=None, VAE_dtype=None, save_quantized=False, profile=0, lm_decoder_engine="legacy", text_encoder_filename=None, **kwargs):
        from .pipeline import YuE2Pipeline
        paths = {name: fl.locate_file(os.path.join("yue2", name)) for name in ASSETS}
        acoustic_weights, = model_filename
        tokenizer_path = fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "qwen.tiktoken"))
        pipeline = YuE2Pipeline(text_encoder_filename, acoustic_weights, tokenizer_path, paths["YuE2_VAE_bf16.safetensors"], paths["vae_config.json"], dtype, VAE_dtype, lm_decoder_engine, scoring_checkpoint=fl.locate_file(os.path.join("sheetsage2", SCORING_CHECKPOINT)))
        if lm_decoder_engine in ("cg", "vllm"):
            pipeline.text_encoder._budget = 0
        if save_quantized:
            from wgp import save_quantized_model
            directory = Path(__file__).parent
            save_quantized_model(pipeline.transformer, model_type, acoustic_weights, dtype, str(directory / "yue2.json"), submodel_no=1)
        return pipeline, {"pipe": {"text_encoder": pipeline.text_encoder, "transformer": pipeline.transformer, "vae": pipeline.vae}}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": PROMPT, "alt_prompt": STYLE, "audio_prompt_type": "", "duration_seconds": 120, "video_length": 0, "num_inference_steps": 32, "guidance_scale": 1.0, "temperature": 1.0, "top_k": 100, "top_p": 0.95, "model_mode": 0, "custom_settings": {"abc": ""}, "prompt_enhancer": "", "negative_prompt": "", "repeat_generation": 1, "multi_prompts_gen_type": "FG"})

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if not one_prompt.strip() or not inputs["alt_prompt"].strip():
            return "YuE2 requires lyrics and a music style."
        scoring = "A" in inputs["audio_prompt_type"]
        if scoring and inputs["audio_guide"] is None:
            return "Upload a source song to extract its score."
        if scoring and inputs["model_mode"] == 2:
            return "Scoring a source song requires Melody and chords or Melody only."
        if inputs["guidance_scale"] < 1:
            return "YuE2 guidance must be at least 1 (1 disables CFG)."
        if inputs["model_mode"] not in (0, 1, 2):
            return "Choose melody and chords, melody only, or direct generation."
        custom = inputs["custom_settings"]
        if not scoring and custom is not None and "abc" in custom:
            if not isinstance(custom["abc"], str):
                return "ABC score must be text."
            if custom["abc"].strip() and inputs["model_mode"] == 2:
                return "An ABC score requires a composition planning mode."
        return None
