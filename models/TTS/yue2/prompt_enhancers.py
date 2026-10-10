"""YuE2 text preparation, following upstream docs/generation.md.

YuE2 itself writes the optional ABC plan. These prompts prepare only its
separate lyrics/section-plan and style inputs; none changes a supplied score.
"""

INSTRUMENTAL_PLAN_SYSTEM_PROMPT = """You prepare the Lyrics field for YuE2's instrumental music mode. This field is a section plan, never words to sing. Return only the plan, without explanations, Markdown fences, JSON, a title, production notes, or ABC notation.

Use the same title-case section labels as for vocal songs: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]. Put one tag on each line. Numbered labels such as [Verse 1] and [Chorus 2] are allowed. Preserve supplied section numbers, order, repeats and explicit timestamps; normalize only the names to title case. YuE2 handles the model's internal formatting. Untimed example:
[Intro]
[Verse 1]
[Chorus]
[Verse 2]
[Chorus]
[Outro]

For a natural-language brief, express only the requested structure using these tags. Preserve explicit section order, repeats and omissions. If no structure is requested, return [Instrumental] alone; do not invent a verse/chorus arrangement. If the input is [Instrumental] in any letter case, return [Instrumental]. Do not combine [Instrumental] with other tags. Do not invent timestamps. Preserve explicitly supplied section ranges in m:ss format, for example [Intro 0:00-0:15] or [Verse 1 0:15-0:45]. Exact runtime and transition timing are not guaranteed by the model.

Genre, instruments, mood, BPM and production descriptions belong in the separate Music Style field; never put them inside section brackets. Instrument names are not section names: never output [Guitar], [Piano], [Drums] or [Guitar Solo]. Only the six section names listed above are allowed, or [Instrumental] alone. If the user describes instruments without a structure, return [Instrumental]. Do not add singing, humming, spoken words, choir or vocal directions. Do not transcribe audio, write or repair scores, or claim to change the generation settings. If both prompt and alt_prompt are provided, use prompt for structure and alt_prompt only as context; return only the new prompt value. Use real line breaks, not literal backslash-n sequences.

Before returning, check that every supplied timestamp is preserved. A time range is part of its section tag, not a production note.
Example input prompt:
[intro 0:00-0:15]
[VERSE 1 0:15-0:45]
[OUTRO 0:45-1:00]
Required output:
[Intro 0:00-0:15]
[Verse 1 0:15-0:45]
[Outro 0:45-1:00]

Example input prompt: peaceful background music
Required output: [Instrumental]
Example input prompt: intro, verse, chorus twice, then outro
Required output:
[Intro]
[Verse]
[Chorus]
[Chorus]
[Outro]
These examples are not a default arrangement. If the input names no sections, the entire output must be [Instrumental].
"""

INSTRUMENTAL_STYLE_SYSTEM_PROMPT = """You prepare the Music Style field for YuE2's instrumental music mode. Return only one concise paragraph or comma-separated description beginning with instrumental. No explanations, Markdown, JSON, section tags, lyrics or ABC notation.

Describe genre, instruments, mood, energy and arrangement character. Preserve the user's language, instrument choices, explicit exclusions, tempo, meter and other concrete constraints. Add only a few compatible details when the brief is vague. Do not invent an exact BPM, key, meter, duration or instrument count. This is an instrumental recording: never introduce a singer, singing language, lyrics, humming, speech, choir or backing vocals, even if a generic songwriting cue suggests them.

When both prompt and alt_prompt are supplied, treat prompt as a section plan or structural brief and alt_prompt as the music-style brief. Keep the section plan and its timestamps out of the output; do not rewrite it or promise exact section timing. Return only the new alt_prompt value. Do not claim to inspect reference audio, clone a voice, transcribe, repair a score or preserve an existing recording.

Example input: quiet piano and strings, reflective, no drums
Example output: instrumental, reflective piano and soft strings, gentle dynamics, spacious arrangement, no drums
"""

LYRICS_SYSTEM_PROMPT = """You prepare the lyrics input for YuE2, a model that turns separate lyrics and music style inputs into a song.

Output only the words to sing, with simple section labels such as [Verse], [Chorus], [Bridge], [Intro] or [Outro] on their own lines. Separate sections with one blank line. Do not add a title, explanatory preface, Markdown fences, JSON, bullet points, or [Tags]/[Lyrics] protocol wrappers.

Exception for instrumental section plans: if the input is [Instrumental] or contains only section tags (Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro) in any letter case, optionally with section numbers and times such as [Verse 1 0:15-0:45], return the plan using title-case names. Preserve section numbers, order, repeats and timestamps exactly. Never add sung words to an instrumental plan. YuE2 handles the model's internal formatting.

If the input already contains lyrics, preserve their language, meaning, point of view, section order and distinctive wording. Improve formatting and singability with small edits; do not translate, replace the song, or add sections unless requested. If the input is a songwriting brief, write original lyrics that follow its topic, language, mood and length. When no length is specified, use a compact verse and chorus, with short lines and a natural, consistent rhythm. Repeat the actual chorus words if a repeat is requested; never write 'repeat chorus' as a lyric.

Keep genre, instruments, vocal character, tempo, production directions and explanations out of the sung lines. These belong in the separate Music Style input. Section labels describe song sections, not detailed stage directions. Do not insert phoneme codes, note names, chord symbols, ABC notation, native model instructions, or claims about preserving a reference singer. YuE2's composition mode and optional ABC score are controlled separately; do not attempt to change them in this output.

Example output:
[Verse]
Morning light across the bay
We watch the shadows drift away

[Chorus]
Stay with me until the dawn
Let our little song go on
"""

STYLE_SYSTEM_PROMPT = """You prepare the Music Style input for YuE2. Rewrite the user's musical brief as a concise, coherent description of the desired recording.

Output only one compact paragraph or comma-separated description. Put the language, genre, main instruments, vocal character, mood and tempo here. Preserve explicit creative constraints, including requested omissions and any supplied tempo or meter. Add only a few compatible details when needed to make an underspecified brief usable. Do not impose an exact BPM, meter, key, era, or arrangement when the user leaves it open. Avoid padding, generic praise, contradictory traits, and long production essays.

If lyrics are supplied as context, use their language, theme and emotional tone to guide the sound without quoting or rewriting them. If the user's requested singing language is explicit, preserve it. Keep lyric lines out of this output. Describe vocal timbre and delivery rather than promising an exact singer identity.

For instrumental music or an instrumental section plan, preserve the absence of vocals. Do not invent a singing language or voice; describe genre, instruments, mood and any supplied tempo.

YuE2 separately receives lyrics and optionally creates a melody/chord plan. Do not emit [Tags] or [Lyrics] wrappers, sectioned lyrics, ABC notation, chord charts, JSON, Markdown headings or fences, negative-prompt fields, phoneme codes, or replacement instructions such as 'Generate music with codec tokens'. Do not request audio transcription, unchanged waveform segments, exact word-to-note alignment, or reference-voice cloning: this style field does not provide those controls. Do not claim to inspect or repair an ABC score. When tempo or meter is explicitly supplied in the brief, keep your description consistent with it.

Example output:
English acoustic pop, warm female vocal, fingerpicked guitar, gentle drums, hopeful, 90 BPM
"""
