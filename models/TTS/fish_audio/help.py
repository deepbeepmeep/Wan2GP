# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
# A two-speaker dialogue with a change of tone on every line. Directions open the lines: cues inside the lines make takes stop speaking more
# often (3 of 8 takes for a script with mid-line cues and sounds, none for plain sentences).
DEFAULT_PROMPT = "Speaker 1: [confident, almost playful] Look, it’s simple. You just have to stop trusting the cameras.\nSpeaker 2: [skeptical, raising an eyebrow] Cameras? I’ve seen satellites, Mark.\nSpeaker 1: [waving a hand dismissively, then leaning in] Manufactured images. Have you ever felt the curve?\nSpeaker 2: [frustrated, sighing] I’ve flown in planes. I’ve crossed oceans.\nSpeaker 1: [insistent, then dropping to a whisper] That’s exactly what they want you to think. The horizon is perfectly straight, isn’t it?\nSpeaker 2: [exhausted, rubbing her temples] It looks straight because we’re on top of a sphere, not because it’s flat.\nSpeaker 1: [smiling mysteriously, slowly] And who told you that... The same people selling you the round rock."

INFOS = """Fish Audio Drama 3 generates 44.1 kHz mono speech from a script, performed as directed: short directions in square brackets set tones and emotions, sounds and pauses land where they are written, and paired tags such as whisper apply to chosen words (see the prompt help). Start with temperature 0.8 and a maximum duration of 60 seconds, which fits a short dialogue. Speech may finish sooner; increase the maximum if a sentence is cut off. Occasionally a take stops speaking before the end of its script, more often with many cues inside the lines: after 8 seconds of silence WanGP stops it and cuts the silence (a console warning says so). Generate it again with another seed if words are missing.

**Write** drafts a speech or a dialogue from a brief in the prompt, with its directions, using the prompt enhancer.

Choose **Voice Reference** to imitate a recording, or **Voice References for Speakers 1 and 2** (or **1, 2 and 3**) to give each character of a dialogue its own cloned voice. Supply a clean recording of one person per reference, ending between two sentences. **Reference Transcripts** holds the exact words spoken in each recording, one `Speaker N: ...` line per reference (with a single reference, the words alone are enough); a transcript describes its recording, not the new speech. Leave a transcript empty and WanGP transcribes that recording with Whisper (the model chosen in Configuration > Extensions, downloaded on first use), shown in the console: type it yourself when the recording is noisy or the words matter, since a recognition error weakens the imitation. Speakers without a reference get a voice chosen by the model, and without any reference the model chooses every voice.

The whole script is generated in one pass by default, so the model sees the entire dialogue. For long scripts, **Cut Script Every N Lines** generates whole lines in parts of N lines; each part is prompted with the voice references and, as history, the most recent parts (their text and the audio already generated) that fit in **History Kept at Each Cut**, so voices and delivery carry over. Each cut processes this prompt again, which takes a few seconds, more with long references or history. **Maximum Speech Duration** applies to the whole result.

**Early Stop** decodes the frames already generated; **Abort** discards the current generation. Prompt enhancement is optional and off by default. Drama 3 does not use diffusion steps or classifier-free guidance.

Profile 3 is the default for audio. With every profile, the ordinary layers and the codebook decoder stay in VRAM (about 3.7 GB with INT8, 6 GB with BF16): they run for every audio frame, and sending them to the GPU part by part would make generation several times slower. The routed experts (32 GB in FP8, 16.6 GB with INT4 Experts) stay in RAM, unpinned, and run on the CPU at their released precision: speed mostly depends on the CPU's memory bandwidth, and keeping the FP8 experts cached needs plenty of free system RAM (48 GB or more); otherwise they are read from disk again. With vLLM, profiles that preload VRAM (1 and 3) also copy the most used experts to the GPU before speech generation, as many as the free VRAM allows, which can nearly double the generation speed. Profile 1 also keeps these experts in reserved RAM (about 28 GB more on a 32 GB GPU, and a longer model load), so this copy takes about a second at each generation; with profile 3 it takes a few seconds, which pays off for speech longer than about 10 seconds. With CUDA Graphs and Legacy, experts placed in VRAM are only used while the prompt is processed.

**INT4 Experts** uses Fish Audio's 4-bit release of the routed experts (4-bit weights with a scale per group of 128, made from the FP8 experts) with the same ordinary layers and settings. It halves the RAM and disk reads the experts need, which speeds up the CPU experts, and with profile 3 a 32 GB GPU holds every expert. The 4-bit weights differ slightly from the FP8 ones, which may slightly change the delivery; keep the FP8 model for the reference quality.

Legacy runs layer by layer with PyTorch and is by far the slowest. CUDA Graphs and vLLM generate each audio frame as one recorded GPU sequence that includes the CPU experts; vLLM additionally uses accelerated attention, DeltaNet and fused normalization kernels and is the fastest. The codebook-MTP decoder predicts the ten codebooks of each audio frame; it is not speculative decoding of future speech frames.

Weights and upstream code are subject to the Fish Audio Community License.
"""

PROMPT_INFOS = """Write the exact words to speak in their original language, with natural punctuation, and direct the performance with short cues. Square-bracket cues are performed, not spoken.

### Tones and Emotions

- **A direction at the start of a line** sets how the line is performed: `[warm and relaxed, like talking to an old friend] It's been a while.`
- **A direction in the middle** changes the delivery from that point: `I'm so happy for you. [voice breaking] Really, I am.`
- **Sounds and pauses** land where they are written: `[sigh]`, `[sighs]`, `[laughs]`, `[pause]`.
- **Paired tags** change only the words inside them and must be closed: `<whisper>Go on without me.</whisper>`, `<stress>You</stress> were there that night.`, `<slow>Read it once more.</slow>`.
- Directions are free descriptions, in English or in the script's language: `[conteniendo las lágrimas, pero sonriendo] Gracias por venir.`

```text
[whispering, terrified] Did you hear that? [pause] Something is moving downstairs.
```

```text
[bored, monotone] Yes, the meeting is at nine. [suddenly excited] Wait, is that free pizza?
```

```text
[angry, through clenched teeth] I told you three times. [sigh] [calm and patient] Let's start again, slowly.
```

```text
[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.
```

Short, concrete directions work best; change them as the emotion of the text changes. Directions at the start of lines are the most reliable: every cue inside a line is a point where a take occasionally stops speaking, so keep those for the moments that need them.

### Dialogue

Start each line with its speaker, from `Speaker 1:` to `Speaker 16:`, and give each line its own directions:

```text
Speaker 1: [skeptical, questioning tone] Then why is your coat still wet?
Speaker 2: [defensive, slightly uneasy] I was home all night. [pause] Asleep.
Speaker 1: <stress>Nobody</stress> saw you. [sighs] Just tell me the truth.
Speaker 2: [barely holding back tears] <slow>I didn't do it.</slow>
```

Speakers 1, 2 and 3 use the voice references in their order; the others get voices chosen by the model. A line without a label continues the previous speaker, and an unlabelled script uses Speaker 1. Drama 3's own tags `<|speaker:0|>` to `<|speaker:15|>` also work (`<|speaker:0|>` is Speaker 1). Keep each character's number consistent. Do not insert conversation-control or semantic-code tokens.

**Reference Transcripts** must contain the exact words of the voice recordings, for example:

```text
Speaker 1: Thanks for calling, how can I help you today?
Speaker 2: I'd like to change my reservation, please.
```

Temperature changes variation; top-k and top-p restrict sampling. Start with temperature 0.8, top-k 30 and top-p 0.95.
"""

DEEPY_INFOS = """Drama 3 produces new 44.1 kHz mono speech. `prompt` is the spoken script. `audio_prompt_type`: empty = new voices; `A` = clone `audio_guide` as Speaker 1; `AB` adds `audio_guide2` as Speaker 2; `ABD` adds `audio_guide3` as Speaker 3. `alt_prompt` = exact transcript of each reference, one `Speaker N: ...` line per reference (plain words for `A`); an empty transcript is made by Whisper (less exact). Script speakers 1..16, unreferenced ones get model voices. One pass by default; `custom_settings.dialogue_cut_lines` N>0 generates parts of N whole lines, each prompted with the references plus the last parts' text+audio fitting in `dialogue_history_seconds` (default 20); each cut re-processes that prompt (seconds). No audio preservation, diffusion or CFG. `duration_seconds` is a maximum and can truncate speech; a take that falls silent mid-script is a sampling failure, stopped after 8 s of silence with the silence cut (finish reason `silence`): retry with another `seed`. Cues inside lines raise that risk; line-start directions are the most reliable. Start at 60 seconds, `temperature=0.8`, `top_k=30`, `top_p=0.95`, `prompt_enhancer=""`. Early Stop renders completed frames; Abort returns no audio. On every profile the ordinary layers and codebook decoder stay in VRAM (~3.7 GB INT8, ~6 GB BF16); the 32 GB of routed experts run on the CPU from pageable RAM (48 GB+ system RAM recommended), so CPU memory bandwidth bounds speed. With `vllm`, profiles 1 and 3 copy the most used experts to free VRAM before generation (up to ~2x faster frames; profile 1 keeps them in reserved RAM, ~28 GB more, so the copy takes ~1 s; profile 3 copies them in a few seconds per generation, worth it beyond ~10 s of speech); with `cg`/`legacy` VRAM experts serve prompt processing only. Model `fish_drama3_int4`: same settings, Fish Audio's 4-bit experts (16.6 GB instead of 32 GB: faster CPU experts, all experts on a 32 GB GPU with profile 3), possibly slightly different delivery; `fish_drama3` (FP8) is the reference quality. Engines: `vllm` fastest, `cg`, `legacy` slowest (layer by layer). Prompt enhancer (`prompt_enhancer`, off by default): `T` writes a speech, `T1` a dialogue (`Speaker N:` lines), both with directions. Codebook-MTP predicts codes within each frame, not future frames. Fish Audio Community License applies."""

DEEPY_PROMPT_INFOS = """Submit the actual words to speak, preserving the user's language and exact supplied wording. Direct the performance with cues that are performed, not spoken: a free-form direction in square brackets at a line start sets that line's tone/emotion (`[warm and cheerful] Good morning!`), mid-line it changes the delivery from there (`I'm fine. [voice breaking] Really.`); sounds/pauses where they happen (`[sigh]`, `[sighs]`, `[laughs]`, `[pause]`); paired tags on chosen words only, always closed (`<whisper>...</whisper>`, `<stress>...</stress>`, `<slow>...</slow>`); directions may use the script's language. Prefer one direction at the start of each line and avoid cues inside lines (each is a point where a take can stop speaking): fold changes into the opening direction (`[calm, then suddenly angry]`) and make pauses with punctuation. Dialogue: one line per turn starting `Speaker 1:` .. `Speaker 16:` (`Speaker 1: Are you ready?\nSpeaker 2: Almost.`); unlabelled lines continue the previous speaker, an unlabelled script is Speaker 1; Speakers 1-3 take the references in order. Native `<|speaker:0|>`..`<|speaker:15|>` also work (0 = Speaker 1). Do not insert `<|im_start|>`, `<|im_end|>`, `<|voice|>`, `<|text|>`, `<|interleave|>` or semantic tokens. `alt_prompt` transcribes the references exactly (`Speaker 1: ...` / `Speaker 2: ...` lines; plain words for `A`), or stays empty for Whisper; it is not a voice-style description or the new script. Without reference audio, use empty `alt_prompt`."""

_DIRECTIONS = """Direct the performance with cues that the model performs instead of speaking:
- one short direction in square brackets at the very start of every line, setting that line's tone or emotion ([warm and cheerful], [suddenly worried], [barely holding back tears]);
- when the delivery changes within a turn, say it in that opening direction ([calm, then suddenly angry], [laughing, then whispering]) or start a new line;
- sounds, whispering, stress or slowness inside the opening direction too ([sighing], [laughing softly], [whispering], [stressing each word], [slowly]). Do not write any angle-bracket tag yourself; keep the paired tags the user wrote (<whisper>...</whisper>, <stress>...</stress>, <slow>...</slow>) exactly as they are.
Never put a square-bracket cue after the start of a line, not even [pause] or a sound: a cue inside a line is where a take most often stops speaking. Make pauses and hesitations with punctuation instead (commas, "...", dashes). Write directions in English or in the script's language, keep them short and concrete, and let the emotions change from line to line. Keep the cues, tags and speaker labels the user already wrote as they are. Use natural punctuation. Do not add structural conversation tokens (such as <|im_start|>), semantic tokens, reference transcripts, Markdown fences or commentary."""

ENHANCER = """Write only the speech script of one speaker for Fish Audio Drama 3. Preserve the user's language, exact supplied wording, meaning and explicit exclusions; if the user supplied the exact words to speak, keep them and only add directions. If asked to author speech from a brief, write the finished words to speak, 3 to 8 sentences unless another length is asked, on one line per change of tone or emotion. """ + _DIRECTIONS + """ Do not add speaker labels unless the user wrote some. Return only the script. The example shows the format only: write new content for the user's request.

Example:
[hushed, nervous] Is anyone there? The lights went out an hour ago...
[relieved, laughing softly, then whispering] Oh, it's only you. You scared me half to death.
[warm] Come in, the tea is still hot."""

DIALOGUE_ENHANCER = """Write a dialogue script for Fish Audio Drama 3 from the user prompt, with two speakers unless the user asks for more (up to 16). Preserve the user's language, exact supplied wording, characters, meaning and explicit exclusions; if the user supplied exact lines, keep them and only add directions. Write 6 to 14 lines unless another length is asked. Every line starts with its speaker label, Speaker 1: to Speaker 16:, one turn per line, each character keeping the same number. """ + _DIRECTIONS + """ Return only the dialogue lines. The example shows the format only: write new content for the user's request.

Example:
Speaker 1: [skeptical, questioning tone] Then why is your coat still wet?
Speaker 2: [defensive, slightly uneasy, hesitating] I was home all night... Asleep.
Speaker 1: [sighing, then stressing each word] Nobody saw you. Just tell me the truth.
Speaker 2: [barely holding back tears, slowly] I didn't do it."""
