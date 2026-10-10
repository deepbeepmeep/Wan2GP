"""User guidance shared by the AuK model definitions."""

INFOS = """
### Choose how to use AuK
- **Instruction TTS:** enter what should be said and describe the voice. No audio upload is needed.
- **Voice cloning / edit source audio:** upload one clip. Ask AuK to speak new words in that voice, change the recording, clean up speech, or keep a particular speaker.

### Set the target duration
Without source audio, the duration sets the length of the **new result**, up to **300 seconds**. With source/reference audio, AuK uses only the opening section up to the requested duration and produces a result of that length. If the clip is shorter, the result stops at the clip's duration: a 30-second clip with a 5-second target uses its first 5 seconds; a 3-second clip with a 5-second target produces 3 seconds. A notice appears when source audio will be cropped. A clip longer than 300 seconds can only contribute its first 300 seconds, or less if you choose a shorter target.

Choose a cutoff at a natural pause to avoid splitting a word. Later parts of the recording are excluded. Give spoken text enough time to fit naturally; inserting many words into a short section can make speech rushed or incomplete. Increasing the target duration includes more source audio, up to the clip's length. For cloning, use a reference at least as long as the result you want.

### Get better results
- Start with a short clip and one clear request. For cloning, use a clean recording of one speaker with little background noise.
- Prefer sections of **30 seconds or less**, matching the upstream interface's recommended range. Longer runs need more memory and may lose consistency or instruction accuracy.
- For editing, say what to change **and what to preserve**. Begin with one change before combining several.
- Use English or Chinese speech and instructions. Other languages are not validated here.
- Keep the same seed when comparing prompt changes. Try another seed if a result is poor.
- **Flash** is the fast four-step option. **Base** offers adjustable steps and guidance; start with its defaults. Flash's sampling recipe is fixed.

### Know the limits
AuK creates a new recording. Even words you did not ask to change may sound slightly different; this is not an exact cut-and-paste editor. Check the words, speaker identity and background after edits or cleanup. Output is mono speech at 24 kHz. Use voices you have permission to use.

AuK uses a Qwen component licensed for research/evaluation. Commercial use requires a separate Qwen license.
""".strip()

PROMPT_INFOS = """
### Prompt wording matters
**AuK is sensitive to the exact instruction. Small wording changes can produce a different task or leave the original speech largely unchanged.** Uploading a reference does not by itself tell AuK whether to clone its voice or edit its contents. Start with the templates below, change only the quoted words or requested emotion, and keep the seed fixed when comparing results. These examples are starting points, not guarantees.

### Write an instruction
Tell AuK the task in a complete sentence. For new speech, include the **exact words in quotation marks**. For edits, name the change and any qualities you want to keep.

| Goal | Example prompt | Audio needed? |
| --- | --- | --- |
| Create speech | Say \"Welcome back. It is wonderful to hear from you.\" in a warm, calm female voice. | No |
| Reuse a voice | Say the following with the same voice: \"Welcome back. It is wonderful to hear from you.\" | Yes, a voice reference |
| Change words | Replace \"see you tomorrow\" with \"see you next week\", keeping the same speaker and tone. | Yes, the original recording |
| Change emotion | Change the emotion to happy. | Yes |
| More energetic emotion | Change the emotion to excited. | Yes |
| Convert a whisper | Convert this whispered speech into normal speech while preserving the speaker and content. | Yes |
| Clean up speech | Remove background noise and reverberation while preserving all spoken words and speakers. | Yes |
| Keep one speaker | Keep only the second speaker in order of appearance and remove the other speakers. | Yes |

### Best practices
- For cloning, keep the full wording `Say the following with the same voice: "..."`. In our tests, a bare `Say "Welcome back. It is wonderful to hear from you."` reproduced the reference's original words; the full cloning template spoke the requested sentence. Start with refinement off when testing this template.
- For emotion editing, use a named emotion such as `happy`, `excited`, `sad` or `angry`. Vague requests such as "make it hilarious" may be reinterpreted: in our test, Refine turned that request into `happy`, not laughter or comic delivery. The strength of the emotion change varies; compare the result with the source.
- For new speech or cloning, match duration to the words you want spoken. The welcome sentence above worked with a 5-second target. **There is no automatic stop after the requested words**: excess duration can produce repetitions or invented speech, while too little can cut words off. The 30-second default is not a recommended length for every sentence.
- Quote the original and replacement words exactly for content edits.
- Describe voice, emotion and pace with a few concrete words. Avoid long lists of competing directions.
- For cleanup or emotion edits, choose the length of the opening section to process. The result has that same length, capped at the source duration. Shorten the requested speech if it cannot fit naturally.
- The prompt is an instruction, not a negative-prompt list. No timestamps, edit mask or special task tags are required.
- **Refine** optionally reformats your request and improves voice descriptions while preserving quoted words. Choose **AuK instruction from text** for ordinary rewriting. For content edits, **AuK instruction from text + source transcription** first transcribes the same opening section selected by the target duration to help locate words. Transcription takes extra time and may mishear words; your explicitly quoted words take priority. Check the rewritten instruction and set the target duration yourself. Automatic enhancement is off by default.
- Transcription runs locally. If no source is selected, no speech is detected or transcription fails, a notice appears and refinement continues using text only. Cancelling still stops the operation. If you select an external prompt-enhancer engine, that engine receives any transcript along with your request.
""".strip()

DEEPY_INFOS = """
### Inputs and duration
- **Instruction TTS:** prompt only. **Cloning/editing/cleanup/separation:** one source/reference audio clip (`audio_prompt_type: A`).
- `duration_seconds` sets the output length, at most 300 s; prefer ≤30 s for consistency and memory use. With source audio, both conditioning and output use the opening `min(duration_seconds, source duration)` seconds; longer requested durations stop at the clip's end. Cropping produces an informational notice. Sources longer than 300 s contribute only their first 300 s or the shorter requested duration. Later speech is excluded. Choose a natural pause as the cutoff; inserted speech still needs to fit this duration.
- Clean single-speaker references work best for cloning. English and Chinese are supported; other languages are unvalidated.
- Flash uses four fixed steps with guidance off. Start Base at 32 steps and guidance 2. Keep the seed fixed when comparing changes.

### Limits
Output is mono 24 kHz. Edits regenerate the audio, so check words and speaker identity; untouched regions are not sample-identical. Use authorized voices. The required Qwen component is research/evaluation licensed; commercial use needs a separate Qwen license.
""".strip()

DEEPY_PROMPT_INFOS = """
### Prompt essentials
**AuK is sensitive to exact wording: small changes can select a different task or reproduce the original speech.** An audio upload alone does not distinguish cloning from editing. Start with the templates, refinement off, and a fixed seed; results are not guaranteed.
Use one clear instruction. Quote exact target speech or original/replacement words; say what must stay unchanged.
- TTS: `Say "Welcome back" in a warm, calm female voice.`
- Clone: `Say the following with the same voice: "Welcome back. It is wonderful to hear from you."` (5-second target worked in testing).
- Edit: `Replace "tomorrow" with "next week", keeping the same speaker and tone.`
- Emotion: `Change the emotion to happy.` (or `excited` for a more energetic target; strength varies).
- Cleanup: `Remove background noise and reverberation while preserving all speech.`
- Separation: `Keep only the second speaker in order of appearance.`

Only TTS works without audio. Match target duration to the intended speech, especially for changed words or pace. No negative prompt, timestamps, mask or special task tags are required.
Keep the full cloning template: a bare `Say "..."` with source audio can reproduce the original words. Start with refinement off when testing it.
Use named emotions; vague "hilarious" was rewritten to `happy` by Refine in testing, not laughter or comic delivery. Inspect the refined instruction. There is no automatic stop after the requested words: too much duration can produce repetition or invented speech; too little can cut words off. The 30-second default may be too long for short TTS/cloning requests.
Optional **Refine** modes: `T` = **AuK instruction from text**; `TW` = **AuK instruction from text + source transcription**, using a local transcript of the same opening section selected by duration. `TW` uses `audio_prompt_type: A` and `audio_guide`; missing source, no detected speech or transcription failure produces a notice and text-only refinement. Cancellation still stops. Useful for word edits; adds transcription time, and explicit quoted words override recognition errors. Neither enhancer mode changes the duration setting. Disabled by default. An external enhancer receives the transcript.
""".strip()
