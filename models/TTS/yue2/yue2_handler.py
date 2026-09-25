import os
from pathlib import Path

from shared.utils import files_locator as fl
from .prompt_enhancers import LYRICS_SYSTEM_PROMPT, STYLE_SYSTEM_PROMPT, INSTRUMENTAL_PLAN_SYSTEM_PROMPT, INSTRUMENTAL_STYLE_SYSTEM_PROMPT


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

INFOS = """YuE2 creates new 48 kHz stereo songs from lyrics and a music style. It also supports instrumental composition and using a score from an existing recording.

### Make Your First Song

1. Enter the actual words to sing in **Lyrics**, with labels such as `[Verse]` and `[Chorus]`. See Prompt Help for an example.
2. Describe the language, genre, instruments, voice and mood in **Music Style**. For example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`.
3. Choose **Melody and chords** under **Composition Planning**. Leave **Source Audio** at **No Audio** and **Optional ABC Score** empty.
4. Start with **32 steps**, **guidance 1**, and the default **120-second maximum** for a short song. Prompt enhancement is optional; you can generate directly from finished lyrics and style.

Change the seed for another performance, or keep the lyrics and change Music Style for another arrangement.

### Composition and Duration

- **Melody and chords:** plans the tune and harmony before synthesis; recommended for your first song.
- **Melody only:** plans the tune and gives the accompaniment more freedom.
- **Direct generation:** skips the written score; try it for another interpretation. It cannot use a supplied score or source audio, or export a score.
- **Instrumental - Melody and Chords:** uses a section plan instead of sung words, with the built-in instrumental AR LoRA.

**Maximum Song Duration is a cap, not a target.** The song can end sooner; a short cap can cut it off. Increase it or shorten the lyrics if the ending is truncated. Available model context also limits length. More synthesis steps take longer; guidance 1 disables CFG, while higher values strengthen lyrics/style conditioning.

**Abort** cancels without saving a partial song. **Early Stop** renders what has already been composed; synthesis and decoding still need to finish.

### Instrumental Music

Select **Instrumental - Melody and Chords**, put `[Instrumental]` in **Lyrics**, and describe the instruments and mood in **Music Style**, for example `instrumental, ambient, piano, soft strings, reflective`. For a chosen structure, use section tags such as `[Intro]`, `[Verse 1]` and `[Chorus]` as shown in Prompt Help. Leave source audio and the ABC file empty for automatic composition.

The instrumental adapter downloads on first use and loads automatically at strength 1; no manual LoRA selection is needed. It plans melody and chords before composing audio. Section order and optional times guide the result but do not guarantee exact transitions. Use a short duration cap for previews or a larger cap to allow a longer piece and ending.

### Optional Prompt Enhancement

Enhancement is **off by default**. Choose what you want to edit:

| Songs with vocals | Instrumentals | Updates |
| --- | --- | --- |
| Lyrics | Instrumental Section Plan | Lyrics field only |
| Music Style | Instrumental Music Style | Music Style only, using the other field as context |
| Lyrics then Music Style | Instrumental Plan then Music Style | Both fields, in that order |

For finished lyrics or a finished section plan, choose style only. To start from an idea, choose lyrics/plan or both, click **Enhance**, and review the result before generation. Select an instrumental enhancer explicitly: Composition Planning does not switch it automatically. Enhancers prepare text inputs; they do not write or edit ABC scores.

### Source Audio and ABC Scores

To reinterpret an existing song, select **Source Audio > Extract Score from Source Song** and upload the recording. YuE2 transcribes musical notes with SheetSage2/MERT2; enter the lyrics separately and align their sections with the source. Choose **Melody and chords** to retain harmony or **Melody only** for freer accompaniment. Instrumental mode accepts a section plan instead of lyrics. Transcription errors can affect the result. This creates a new recording; it does not preserve the original waveform or clone its singer.

To reuse a written composition, select **No Audio** and upload a compatible UTF-8 `.abc` file under **Optional ABC Score**. It replaces automatic planning. Use the supported Vocal/Ins score format; melody-only scores must omit chord symbols. Source audio hides and overrides the manual ABC file.

Enable **Save ABC and MIDI Score** to export `.abc` and `.mid` beside the song. These contain the composition used for generation, not a transcription of the final performance, and may extend beyond an early-stopped song. Export is off by default.

### LoRAs and License

Compatible YuE2 **AR** and **acoustic/diffusion** LoRAs can be selected together, each with its own multiplier. AR LoRAs affect composition and conditioning and require a constant multiplier. Acoustic LoRAs affect rendering and can use step schedules. Start at 1 and lower it for a weaker effect. To change the built-in instrumental adapter's strength, select the same adapter manually; your multiplier replaces the automatic copy.

The model weights and built-in instrumental adapter use **CC BY-NC 4.0 (non-commercial use)**.
"""
PROMPT_INFOS = """### Lyrics for Classic YuE2

Classic modes (**Melody and chords**, **Melody only**, **Direct generation**) accept section labels as ordinary text, not a fixed list of control tokens. Use these standard song sections:

| Label | Purpose |
| --- | --- |
| `[Intro]` | Opening section |
| `[Verse]` | Main lyric verses |
| `[Pre-Chorus]` | Build-up before a chorus |
| `[Chorus]` | Repeated hook or refrain |
| `[Post-Chorus]` | Follow-up after a chorus |
| `[Bridge]` | Contrasting section |
| `[Interlude]` | Instrumental passage between sung sections |
| `[Solo]` | Featured instrumental passage |
| `[Outro]` | Closing section |

Numbered labels such as `[Verse 1]` and `[Verse 2]`, and descriptive variants such as `[Final Chorus]` or `[Saxophone Solo]`, can also express structure. These are musical cues, not guaranteed controls; there is no exhaustive supported-label whitelist in classic mode. Leave instrumental passages without lyric lines and keep detailed sound instructions in Music Style. Instrumental mode uses the same capitalization and numbering conventions, with the restricted set of section names listed below. YuE2 handles its internal formatting automatically.

Write the actual words to sing, not a request to write a song. Put labels on their own lines, use short singable lines, and leave a blank line between sections:

```text
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
```

Repeat the actual chorus words when you want it again. Keep instruments, production notes and explanations out of the lyrics. With **Lyrics** or **Lyrics then Music Style** enhancement, you can instead enter a songwriting idea, click **Enhance**, then review the words before generating.

### Music Style for Songs

Describe the language, genre, main instruments, vocal character and mood. Add tempo when it matters:

```text
English acoustic pop, warm female vocal, fingerpicked guitar,
gentle drums, hopeful, 90 BPM
```

Use a few compatible ideas rather than conflicting styles. **Music Style** enhancement uses the lyrics as context while leaving them unchanged. A voice description guides vocal character; it does not guarantee a specific singer's identity.

### Instrumental Prompts

Only for **Instrumental - Melody and Chords**, replace sung lyrics with `[Instrumental]` to let the model choose the structure, or use one section tag per line (the same title-case format as vocal songs):

```text
[Intro]
[Verse]
[Chorus]
[Bridge]
[Chorus]
[Outro]
```

Supported tags: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`. Numbered sections such as `[Verse 1]` and `[Chorus 2]` are accepted. Capitalization is flexible; YuE2 handles it and removes section numbers internally without changing section order or repetition. Optional times go inside the tags, for example `[Intro 0:00-0:15]` followed by `[Verse 1 0:15-0:45]` on the next line. They guide structure without guaranteeing exact timing. Use real line breaks, no sung words or bracketed production notes, and do not combine `[Instrumental]` with section tags. Invalid plans report the faulty line number and its original content.

Put the sound description in **Music Style**:

```text
instrumental, ambient, piano, soft strings, reflective, 80 BPM
```

Use **Instrumental Music Style** to enhance only that description. **Instrumental Section Plan** turns a structural brief such as “intro, verse, chorus twice, then outro” into tags; **Instrumental Plan then Music Style** prepares both fields. With no requested structure, plan enhancement returns `[Instrumental]`. Select the instrumental option explicitly, click **Enhance**, and review the plan before generating.

### When Using Source Audio or a Score

Source audio supplies notes, not a lyric transcript: enter the words separately and match their section order and phrasing to the recording. In instrumental mode, enter a section plan instead. Upload ABC notation through **Optional ABC Score**, not in either text field; select **No Audio** to use that file. See Model Help for score import and export details.
"""

DEEPY_INFOS = """### Classic YuE2 Songs
`prompt` = actual lyrics; `alt_prompt` = language, genre, instruments, voice, mood and optional tempo. Outputs new 48 kHz stereo audio. Start with `model_mode=0`, no source audio or ABC, 32 steps, guidance 1 and a 120-second maximum for a short song. Change seed for a new performance or style for another arrangement.

### Composition and Duration
`model_mode`: 0 = melody+chords (recommended); 1 = melody only/free accompaniment; 2 = direct generation without a score; 3 = instrumental planning. Mode 2 cannot use source audio/manual ABC or export a score. `duration_seconds` is an upper limit, not a target: songs may end earlier or be cut off; increase the cap or shorten lyrics if truncated. Remaining model context also limits length. More steps cost time; guidance 1 disables CFG, higher values strengthen text conditioning. Abort cancels without audio. Early Stop renders composed tokens after synthesis/decoding; during score planning it finishes the score then makes an approximately eight-second preview capped by duration.

### Instrumentals
Mode 3 uses `[Instrumental]` or a section-only plan such as `[Intro]`, `[Verse 1]`, `[Chorus]` in `prompt`, with instrumental `alt_prompt`. Use title-case names as for classic songs; YuE2 normalizes case and removes section numbers internally while preserving times and repetition. Only the six section names listed in prompt help are allowed; instruments belong in `alt_prompt`. Its built-in AR LoRA downloads just in time at strength 1; no manual selection needed. Selecting the same adapter manually uses your multiplier instead. Times/order guide rather than guarantee transitions. Use short caps for previews or larger caps to allow longer pieces/endings.

### Source Audio and Scores
`audio_prompt_type="A"` + `audio_guide` transcribes musical notes with SheetSage2/MERT2 in modes 0/1/3. Supply lyrics separately, aligned to the source, or a plan in mode 3. Mode 0 retains harmony; mode 1 allows freer accompaniment. Transcription can make mistakes; regeneration does not preserve waveforms or clone a singer. With no source audio, optional `custom_guide` = compatible UTF-8 `.abc` file replaces automatic planning; use Vocal/Ins voices, omit chord symbols for mode 1. Source audio hides/overrides manual ABC. Old `custom_settings.abc` text is ignored. `custom_settings.save_score=1` exports ABC/MIDI (default 0): the conditioning composition, not a transcription of final audio, possibly longer than an early-stopped result. API artifacts expose side files in memory.

### LoRAs and License
`activated_loras`/`loras_multipliers` accept compatible YuE2 AR and acoustic/diffusion adapters together with independent strengths. Keys route each component; AR affects composition/conditioning and requires constant strength, acoustic affects rendering and supports step schedules. Start at 1 and lower for a weaker effect. Native/ComfyUI fused adapters are accepted; full replacement weights require base-relative diff tensors. Model and instrumental adapter: CC BY-NC 4.0, non-commercial use.
"""
DEEPY_PROMPT_INFOS = """Prepare generation-ready `prompt` and `alt_prompt` directly from the user's request using the rules below.

### Classic Lyrics (Modes 0/1/2)
Use actual short, singable lines, one bracketed section label per line, and blank lines between sections. Standard sections: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Interlude]`, `[Solo]`, `[Outro]`. Numbered `[Verse 1]`/`[Verse 2]` and descriptive `[Final Chorus]`/`[Saxophone Solo]` are also text cues. Classic mode has no exhaustive label whitelist; labels do not guarantee behavior. Leave instrumental passages without sung lines. Repeat chorus words explicitly; keep production notes out of lyrics. When the user supplies a songwriting brief, write the finished lyrics yourself before submitting `prompt`; never submit the brief as sung text. Preserve supplied lyrics, language, meaning and section order unless the user asks for revisions. Return only lyrics and section labels in this field, without a title, explanations, Markdown fences or JSON.

### Vocal Music Style
`alt_prompt`: language + genre + instruments + vocal character + mood, optionally tempo. Example: `English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM`. Write a concise style description yourself, using the lyrics as context. Preserve the user's instruments, language, mood, exclusions and supplied tempo; add only compatible details and do not invent an exact BPM, key or meter. Avoid conflicting styles and keep lyrics, section tags and ABC notation out of this field. Vocal character does not guarantee singer identity. Change style for a new arrangement, seed for a new take; increase duration or shorten lyrics if cut off.

### Instrumental Mode 3
`prompt`: `[Instrumental]` alone or `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, one per real newline. Use the same title-case format as classic songs. Numbers and times are accepted, for example `[Verse 1 0:15-0:45]`. Do not mix `[Instrumental]` with tags, add sung words, or put production notes in brackets. Unlike classic mode, only these six section names are allowed: do not emit instrument tags such as `[Guitar]` or `[Piano]`, or classic-only variants such as `[Solo]` or `[Final Chorus]`. Put instruments in `alt_prompt`. YuE2 handles capitalization and removes section numbers internally, preserving order, repeated sections and times. Times guide rather than guarantee transitions or duration. Example style: `instrumental, ambient, piano, soft strings, reflective, 80 BPM`. Convert a structural brief into valid tags yourself, preserving requested order, repetitions and explicit times. If no structure is requested, use `[Instrumental]`; do not invent timestamps or an arrangement. Preserve an existing valid plan. Write `alt_prompt` beginning with instrumental, describing genre, instruments, mood and arrangement; preserve explicit constraints and never introduce singing, speech, humming, choir or backing vocals. The AR LoRA loads automatically. If validation fails, use the reported line number and original content to correct that line; do not remove valid times or rewrite unrelated sections.

### Source Audio and Scores
Source audio supplies notes, not words: provide lyrics with matching section order/phrasing, or a plan for mode 3. Upload notation via `custom_guide`, never either text field; use No Audio for manual ABC (source audio overrides it). `custom_settings.save_score=1` retains the ABC/MIDI composition for editing/reuse, not a transcription of the final performance. See model help for modes and score constraints.
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
            "yue2_lora_instrumental": f"https://huggingface.co/{REPO_ID}/resolve/main/yue2/ar_lora_inst_v3abc.bf16.safetensors",
            "prompt_enhancer_def": {
                "selection": ["T1", "B2O", "T1,B2O", "B3", "B4O", "B3,B4O"],
                "labels": {"T1": "Lyrics", "B2O": "Music Style", "T1,B2O": "Lyrics then Music Style", "B3": "Instrumental Section Plan", "B4O": "Instrumental Music Style", "B3,B4O": "Instrumental Plan then Music Style"},
                "default": "",
            },
            "text_prompt_enhancer_instructions1": LYRICS_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens1": 1536,
            "text_prompt_enhancer_instructions2": STYLE_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens2": 384,
            "text_prompt_enhancer_instructions3": INSTRUMENTAL_PLAN_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens3": 512,
            "text_prompt_enhancer_instructions4": INSTRUMENTAL_STYLE_SYSTEM_PROMPT, "text_prompt_enhancer_max_tokens4": 384,
            "alt_prompt_inherits_prompt_paragraphs": True,
            "alt_prompt": {"label": "Music Style", "placeholder": "Language, genre, instruments, mood, vocal character and tempo", "lines": 3},
            "model_modes": {"choices": [("Melody and chords", 0), ("Melody only", 1), ("Direct generation", 2), ("Instrumental - Melody and Chords", 3)], "default": 0, "label": "Composition Planning"},
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
            definition["prompt_enhancer_def"]["selection"] = ["T1", "B2O", "T1,B2O"]
            definition["prompt_enhancer_def"]["labels"] = {key: value for key, value in definition["prompt_enhancer_def"]["labels"].items() if key in definition["prompt_enhancer_def"]["selection"]}
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
        return pipeline, {"pipe": pipe, "loras": ["text_encoder"]}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": PROMPT, "alt_prompt": STYLE, "audio_prompt_type": "", "duration_seconds": 120, "video_length": 0, "num_inference_steps": 32, "guidance_scale": 1.0, "temperature": 1.0, "top_k": 100, "top_p": 0.95, "model_mode": 0, "custom_guide": None, "custom_settings": {"save_score": 0}, "prompt_enhancer": "", "negative_prompt": "", "repeat_generation": 1, "multi_prompts_gen_type": "FG"})
        if base_model_type == HUM_ARCHITECTURE:
            ui_defaults["audio_prompt_type"] = "A"

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        # Retired style-only variants now share the contextual style choice.
        mode = ui_defaults.get("prompt_enhancer", "")
        if isinstance(mode, str):
            aliases = {"L2O": "B2O", "L4O": "B4O"}
            ui_defaults["prompt_enhancer"] = ",".join(
                aliases.get(step.replace("K", ""), step.replace("K", "")) + ("K" if "K" in step else "")
                for step in mode.split(",")
            )

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
        if inputs["model_mode"] not in (0, 1, 2, 3):
            return "Choose melody and chords, melody only, direct generation, or instrumental."
        if inputs["model_mode"] == 3:
            from .instrumental import validate_instrumental_prompt
            error = validate_instrumental_prompt(one_prompt)
            if error:
                return error
        if not scoring and inputs["custom_guide"] is not None:
            if Path(inputs["custom_guide"]).suffix.lower() != ".abc":
                return "Upload an ABC score file with the .abc extension."
            if inputs["model_mode"] == 2:
                return "An ABC score requires a composition planning mode."
        return None
