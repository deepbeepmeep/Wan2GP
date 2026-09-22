import os
from pathlib import Path

from shared.utils import files_locator as fl
from .prompt_enhancers import LYRICS_SYSTEM_PROMPT, STYLE_SYSTEM_PROMPT


ARCHITECTURE = "yue2"
HUM_ARCHITECTURE = "yue2_hum"
REPO_ID = "DeepBeepMeep/TTS"
TEXT_ENCODER_FOLDER = "YuE2_AR"
ASSETS = ["vae_config.json", "YuE2_VAE_bf16.safetensors"]
SCORING_CHECKPOINT = "SheetSage2_MERT2_bf16.safetensors"
HUM_ENCODER = "YuE2_VAE_Encoder.safetensors"
HUM_INFOS = """**Turn a hummed melody into a new song.** Upload a clear human hum (10–30 seconds is a useful starting point), enter the words to sing in **Lyrics**, and describe the genre, instruments and voice in **Music Style**.

**Continue Hum** uses your melody as the opening of a longer composition. **Hum Only** keeps the transcribed score without extending it; this does not guarantee an exact waveform duration. Both modes also use the hum's pitch contour and timing during audio synthesis. Transcription errors can change notes. This creates a new performance; it does not clone your voice or preserve the input recording.

Start with 32 synthesis steps and guidance 1. Guidance controls the lyrics/style conditioning, not a separate hum strength. Maximum Song Duration caps the result; it does not force that length. The song can end sooner, and a short cap can cut it off. Change the seed for another interpretation. Use an isolated hum rather than a full mixed song. No manual ABC score or direct-generation mode is offered for this finetune.

**Save ABC and MIDI Score** exports the composition used for generation. For Continue Hum, this includes the generated continuation. Abort cancels; Early Stop renders the audio tokens already composed. The included acoustic LoRA is required and already incorporates Mothersuperior's real-audio v4 LoRA; do not add that LoRA again. Prompt enhancement is optional and off by default; it edits lyrics/style only. **CC BY-NC 4.0: non-commercial use.**
"""
HUM_PROMPT_INFOS = """Write the actual words to sing, with short lines and section labels such as `[Verse]` and `[Chorus]`. Repeat chorus words explicitly. For example:
```text
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
```
Music Style example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`.
Your recording supplies the melody and phrasing, not a transcript of lyrics. Match the opening lyric phrases to the hum; Continue Hum can compose additional sections. Hum Only works best with lyrics sized for the hummed phrase. Avoid competing style instructions. The optional Lyrics and Music Style enhancers do not process or alter the hum.
"""
HUM_DEEPY_INFOS = """YuE2 Hum-to-Song: required `audio_guide` = clear human hum; `audio_prompt_type=\"A\"`. `prompt` = actual lyrics; `alt_prompt` = music style. `model_mode=0` continues the hummed melody into a longer score; `1` uses only its transcribed score. No direct generation or manual ABC. Outputs a new 48 kHz stereo song, not voice cloning or waveform-preserving editing. Start with 10–30 s of isolated humming, 32 steps and guidance 1 (lyrics/style CFG). Duration is an upper limit, not a target. `custom_settings.save_score=1` exports the final ABC/MIDI composition. Built-in hum LoRA already includes real-audio v4; do not stack it again. Abort cancels; Early Stop renders composed tokens. Enhancer off by default, edits lyrics/style only. CC BY-NC 4.0.
"""
PROMPT = """[Verse]
I left my keys beside your coffee
Caught the first bus out of town
Watched the harbor through the window
Till the morning mist came down

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Verse]
City rain against my collar
Your old song inside my head
I could hear you in the silence
Of the words we never said

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Bridge]
Now the last train crosses over
And the rooftops come in view
I have found the words I needed
I am bringing them to you

[Chorus]
Leave a light on by the water
Let it shine across the blue
Every road can take me farther
Every road leads back to you

[Outro]
There's a light on by the water
And I'm coming home to you"""
STYLE = "English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM"
INFOS = """**Turn your lyrics into a complete song** with a singing voice and accompaniment, in stereo.

- **Lyrics** supplies the words to sing. **Music Style** describes how the song should sound. Use both for your first song.
- **Melody and chords** is the recommended starting point: YuE2 plans the tune and its harmony before making the recording.
- **Melody only** plans the tune while giving the accompaniment more freedom. Use this when a supplied melody should keep its shape but the arrangement can change.
- **Direct generation** skips the written composition plan. Try it for an alternative interpretation of the same lyrics and style.

**Start with the defaults.** Write one or two short verses and a repeated chorus, choose a clear style, and allow enough time for the words. **Maximum Song Duration is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing the limit allows more time but does not force a longer song; a limit that is too short can cut it off. A very large limit is capped by the model's capacity, which also depends on the length of your lyrics and score. Change the seed to try another performance. More synthesis steps cost more time; 32 is the recommended starting point.

**Abort** cancels the generation without saving a partial song. **Early Stop** stops audio-token generation and renders what has already been composed.

**LoRAs:** select compatible YuE2 acoustic-model LoRAs in the LoRAs tab. Start with multiplier 1; lower it for a weaker effect. They affect the audio rendering after composition and audio-token generation. AR planner LoRAs are not supported by this loader.

**Optional prompt enhancer:** disabled by default. Choose Lyrics to turn an idea into singable words or tidy existing lyrics; choose Music Style to clarify the sound, or Lyrics then Music Style to prepare both. Review the result before generating. The enhancer does not edit your ABC score.

**Optional ABC score:** upload a UTF-8 `.abc` file if you already have a compatible written melody or composition. Otherwise leave the file input empty for automatic planning. ABC is a text format for musical notation, not a place for instructions such as “make it happier.” Supplying a score replaces the automatic plan. It requires Melody and chords or Melody only; melody-only scores must omit chord symbols. The supported score format uses Vocal and Ins voices. A score and lyrics that belong together give the model clearer guidance.

**Save ABC and MIDI Score:** turn this on to save both `.abc` and `.mid` beside the song, using the same filename. This exports the planned, supplied or source-transcribed composition, not a transcription of the finished performance. The score can be longer than an early-stopped song. Direct generation has no score to export. Off by default.

Changing the lyrics, style or score creates a **new recording**; it does not preserve the original voice or keep parts of an existing recording untouched. Choose **Source Audio > Extract Score from Source Song** to transcribe a recording with SheetSage2/MERT2. Melody and chords retains harmony; Melody only leaves the new accompaniment more freedom. Direct generation cannot use source audio. The ABC file input is hidden and ignored while source audio is selected. You must still supply lyrics: transcription extracts musical notes, not sung words. Match the lyrics and section order to the source; transcription mistakes can affect the result. The model weights are **CC BY-NC 4.0 (non-commercial use)**.
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
Enable **Save ABC and MIDI Score** to keep the composition for later editing or reuse; the saved score does not add lyrics automatically.
To reuse an ABC score, upload its `.abc` file under **Optional ABC Score**, select No Audio, and use Melody and chords or Melody only. Keep notation out of the lyrics and style fields.

Start with a few compatible ideas. “Gentle acoustic ballad” and “aggressive fast metal” pull in different directions. For a different arrangement, keep the lyrics and change the style; for a different performance, change the seed. If the ending is cut off, allow more time or shorten the lyrics. A style prompt describes the character of a voice; it does not guarantee a particular singer's identity.
"""
DEEPY_INFOS = """**YuE2 song generation:** `prompt` = lyrics; `alt_prompt` = music style. Outputs 48 kHz stereo vocals and accompaniment.
- `activated_loras` / `loras_multipliers`: compatible YuE2 acoustic-model LoRAs, applied during synthesis; start at 1. AR planner LoRAs are not supported.
- `model_mode`: **0** melody+chords (recommended), **1** melody only/free accompaniment, **2** direct generation.
- Optional `custom_guide`: path to a UTF-8 `.abc` score file, replacing planning in modes 0/1. Use Vocal/Ins voices and no chords for mode 1. Align the lyrics with the score. Omit for automatic planning; source audio hides and overrides this file. Old `custom_settings.abc` text is ignored.
- `custom_settings.save_score`: **0** off (default), **1** export both `.abc` and `.mid` with the song's filename. Exports the conditioning composition, not the finished performance; it may outlast a truncated song. Mode 2 exports neither. API artifacts return these side files in memory.
- **`duration_seconds` is an upper limit, not a requested song length. YuE2 can stop earlier when it considers the song finished.** Increasing it does not force a longer song; too short can cut it off. Very large limits are capped by the remaining model context after lyrics/score. Start with 32 steps and guidance 1; change seed for another take.
- Abort cancels without audio. Early Stop renders existing audio tokens; during score planning it finishes the score then makes an approximately eight-second preview, capped by duration. Acoustic synthesis/decoding still finish; previews may end mid-phrase.
- Prompt enhancement is off by default: `T1` prepares lyrics, `L2O` prepares style, `T1,B2O` prepares lyrics then style informed by them. It does not modify ABC.
- `audio_prompt_type="A"` + `audio_guide` transcribes a source song into ABC with SheetSage2/MERT2 in modes 0/1, replacing any manual ABC. Empty audio mode uses automatic planning or manual ABC. Supply lyrics separately; notes are transcribed, not words. Regeneration creates a new recording; no waveform-preserving edits. **CC BY-NC 4.0 weights: non-commercial use.**
"""
DEEPY_PROMPT_INFOS = """**Lyrics:** actual short, singable lines with `[Verse]`, `[Chorus]`, `[Bridge]` on separate lines; blank lines between sections. Repeat chorus words explicitly.
**Music style:** language + genre + instruments + mood + vocal character, optionally tempo. Example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`.
Enable `custom_settings.save_score=1` to retain the ABC/MIDI composition for editing or reuse; sung words still belong in the lyrics input.
Keep style instructions out of the lyrics; avoid conflicting styles. For a cover, supply source lyrics with section order and phrasing matching the recording. Upload a compatible `.abc` file through `custom_guide` to reuse a score; source audio overrides this file. Increase duration or shorten lyrics if truncated; change seed for a new performance. A voice description does not guarantee a singer's identity.
"""


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE, HUM_ARCHITECTURE]

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
    def get_lora_dir(base_model_type):
        return ARCHITECTURE

    @staticmethod
    def query_model_def(base_model_type, model_def):
        definition = {
            "group": "music", "audio_only": True, "image_outputs": False, "sliding_window": False, "supports_early_stop": True,
            "enabled_audio_lora": True,
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
            "custom_guide": {"id": "custom_guide", "name": "ABC Score", "label": "Optional ABC Score (.abc)", "type": "file", "default": None, "required": False, "file_types": [".abc"], "audio_prompt_type_not": "A"},
            "custom_settings": [
                {"id": "save_score", "name": "Save Score", "label": "Save ABC and MIDI Score", "type": "dropdown", "choices": [("Off", 0), ("On", 1)], "default": 0},
            ],
            "duration_slider": {"label": "Maximum Song Duration (seconds)", "name": "Maximum Song Duration", "min": 1, "max": 600, "increment": 1, "default": 120},
            "infos": INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }
        if base_model_type == HUM_ARCHITECTURE:
            definition.update({
                "parent_model_type": ARCHITECTURE,
                "model_modes": {"choices": [("Continue Hum", 0), ("Hum Only", 1)], "default": 0, "label": "Hum Melody"},
                "audio_guide_label": "Hummed Melody",
                "audio_prompt_type_sources": {"selection": ["A"], "labels": {"A": "Hummed Melody to Song"}, "default": "A", "label": "Source Audio", "letters_filter": "A"},
                "infos": HUM_INFOS, "prompt_infos": HUM_PROMPT_INFOS,
                "deepy_infos": HUM_DEEPY_INFOS, "deepy_prompt_infos": HUM_PROMPT_INFOS,
                "specialities": [{"name": "hum to song", "aliases": ["humming to song"], "description": "Create a song from a hummed melody, lyrics and music style."}],
            })
            definition.pop("custom_guide")
        return definition

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = [{"repoId": REPO_ID, "sourceFolderList": ["yue2", TEXT_ENCODER_FOLDER, "sheetsage2"], "fileList": [ASSETS, ["qwen.tiktoken"], [SCORING_CHECKPOINT]]}]
        if base_model_type == HUM_ARCHITECTURE:
            files.append({"repoId": REPO_ID, "sourceFolderList": [""], "fileList": [[HUM_ENCODER]]})
        return files

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, dtype=None, VAE_dtype=None, save_quantized=False, profile=0, lm_decoder_engine="legacy", text_encoder_filename=None, **kwargs):
        from .pipeline import YuE2Pipeline
        paths = {name: fl.locate_file(os.path.join("yue2", name)) for name in ASSETS}
        if base_model_type == HUM_ARCHITECTURE:
            acoustic_weights, hum_weights = model_filename
            hum_encoder_weights = fl.locate_file(HUM_ENCODER)
        else:
            acoustic_weights, = model_filename
            hum_weights = hum_encoder_weights = None
        tokenizer_path = fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "qwen.tiktoken"))
        pipeline = YuE2Pipeline(text_encoder_filename, acoustic_weights, tokenizer_path, paths["YuE2_VAE_bf16.safetensors"], paths["vae_config.json"], dtype, VAE_dtype, lm_decoder_engine, scoring_checkpoint=fl.locate_file(os.path.join("sheetsage2", SCORING_CHECKPOINT)), hum_weights=hum_weights, hum_encoder_weights=hum_encoder_weights)
        if lm_decoder_engine in ("cg", "vllm"):
            pipeline.text_encoder._budget = 0
        if save_quantized:
            from wgp import save_quantized_model
            directory = Path(__file__).parent
            save_quantized_model(pipeline.transformer, model_type, acoustic_weights, dtype, str(directory / "yue2.json"), submodel_no=1)
        pipe = {"text_encoder": pipeline.text_encoder, "transformer": pipeline.transformer, "vae": pipeline.vae}
        if pipeline.hum is not None:
            pipe.update(hum_projections=pipeline.hum, hum_encoder=pipeline.hum_encoder)
        return pipeline, {"pipe": pipe}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": PROMPT, "alt_prompt": STYLE, "audio_prompt_type": "", "duration_seconds": 120, "video_length": 0, "num_inference_steps": 32, "guidance_scale": 1.0, "temperature": 1.0, "top_k": 100, "top_p": 0.95, "model_mode": 0, "custom_guide": None, "custom_settings": {"save_score": 0}, "prompt_enhancer": "", "negative_prompt": "", "repeat_generation": 1, "multi_prompts_gen_type": "FG"})
        if base_model_type == HUM_ARCHITECTURE:
            ui_defaults["audio_prompt_type"] = "A"

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if not one_prompt.strip() or not inputs["alt_prompt"].strip():
            return "YuE2 requires lyrics and a music style."
        if base_model_type == HUM_ARCHITECTURE:
            if inputs["audio_prompt_type"] != "A" or inputs["audio_guide"] is None:
                return "Upload a hummed melody for Hum-to-Song."
            if inputs["model_mode"] not in (0, 1):
                return "Choose Continue Hum or Hum Only."
        scoring = "A" in inputs["audio_prompt_type"]
        if scoring and inputs["audio_guide"] is None:
            return "Upload a source song to extract its score."
        if scoring and inputs["model_mode"] == 2:
            return "Scoring a source song requires Melody and chords or Melody only."
        if inputs["guidance_scale"] < 1:
            return "YuE2 guidance must be at least 1 (1 disables CFG)."
        if inputs["model_mode"] not in (0, 1, 2):
            return "Choose melody and chords, melody only, or direct generation."
        if not scoring and inputs["custom_guide"] is not None:
            if Path(inputs["custom_guide"]).suffix.lower() != ".abc":
                return "Upload an ABC score file with the .abc extension."
            if inputs["model_mode"] == 2:
                return "An ABC score requires a composition planning mode."
        return None
