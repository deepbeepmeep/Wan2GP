"""Prompt guidance and enhancer system prompts for MiniMax H3."""


FL2VA_PROMPT_INFOS = """## H3 FL2VA prompt structure

FL2VA uses the same three-part audiovisual prompt for text-only, first-frame, last-frame, and first-and-last-frame generation:

```text
integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:03.500, ...

overall_soundscape: ...

non_diegetic_music: ...
```

When an image fixes a point on the output timeline, put its alignment instruction before these fields:

- **First frame:** `<Picture 1>` belongs to `[Shot 1]` at `0.00` seconds.
- **Last frame:** `<Picture 1>` belongs to the actual final shot and aligns with the exact end time.
- **First + last:** `<Picture 1>` anchors `0.00` seconds and `<Picture 2>` anchors the exact end time. A single continuous shot is usually preferable unless the requested action genuinely needs cuts.

### Connecting shots

- `[Shot 1]` has no timestamp. Start each later shot with a strictly increasing cut time: `[Shot N] At MM:SS.mmm, ...`.
- A cut should reveal a meaningful change in viewpoint, framing, place, time, subject, or state. Use continuous camera movement instead of a cut for a small change of distance or angle.
- Keep identities, wardrobe, props, spatial relationships, lighting, and action causality consistent across cuts.
- Keep speaker IDs such as `(S1)` stable throughout. Put only exact dialogue or lyrics inside `<d>[Language] ...</d>`.
- If speech crosses a cut, mark the connection with `<scenetrans>` on both sides and say that the audio continues across the cut. Use `<cutoff>` only when the video ends before a spoken line finishes.

`overall_soundscape` summarizes ambience, physical sounds, and non-verbal human sounds without repeating dialogue. `non_diegetic_music` describes only music the audience hears but the characters do not; write `N/A` when no such score is wanted.

Adapted from MiniMax's [official base prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
"""


REF2VA_PROMPT_INFOS = """## H3 Ref2VA prompt structure

Ref2VA uses six sections in this order:

```text
subject_definitions:
<Subject 1> is ... from <Picture 1>.

summary:
[reference generation] ...

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - ...

detailed_description:
The target video is ...
[Shot 1] ...
[Shot 2] At 00:03.500, ...

overall_soundscape: ...

non_diegetic_music: ...
```

When the WanGP enhancer receives an image for Ref2VA, it is the first selected reference image (`<Picture 1>`), not an output start frame. Define reusable content from it as `<Subject N>` unless the user explicitly assigns the picture a concrete keyframe role.

### Reference labels

- `<Subject N>` identifies reusable visible content such as a person, animal, object, environment, costume, style, or motion. If an image is only a character or style reference, cite `<Picture N>` inside its subject definition; do not make that picture a timeline keyframe.
- `<Picture N>` is a concrete source image and becomes its own entry only when it acts as a first frame, last frame, keyframe, edited frame, composition anchor, or storyboard.
- `<Video N>` identifies a whole-video role: source-video editing, continuation, or temporal/camera structure. Visible content taken from it still receives `<Subject N>` labels.
- `<Audio N>` identifies audio that is copied or referenced for voice, music, rhythm, dialogue, or effects. Its numbering is independent of video numbering.

Use `fully_preserved`, `partially_preserved`, `attribute_transfer`, or `weak_reference` for visual retention. Use `fully_copy`, `partially_copy`, `reference`, or `weak_reference` for audio. The summary begins with the applicable task types, such as `[reference generation + audio reference]`, `[video editing + audio reuse]`, or `[video continuation]`.

### Connecting shots

`[Shot 1]` has no timestamp; later shots use strictly increasing cut times in the form `[Shot N] At MM:SS.mmm, ...`. Keep subjects, reference roles, speaker IDs, appearance, props, space, and causality consistent between shots. A dialogue line crossing a cut uses `<scenetrans>` at both connecting points and an explicit continuity phrase. Use `<cutoff>` only when speech is truncated by the end of the video. For reference-generation prompts, MiniMax recommends roughly 350-500 English words in `detailed_description`, with dialogue-heavy timelines sized to fit the actual speech instead.

Describe reference use where it actually takes effect in the timeline. A reference video is not automatically an edit or continuation, and audio is not automatically copied merely because it is present. Put exact dialogue inside `<d>[Language] ...</d>`, ambience and physical sounds in `overall_soundscape`, and audience-only score in `non_diegetic_music`.

Adapted from MiniMax's [official full-reference prompt-writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
"""


_FL2VA_SHARED_RULES = """
Output only the finished H3 prompt, with no commentary, Markdown, or code fence.

Write all descriptive material in English. Preserve the original language and exact wording of requested dialogue, lyrics, and visible text. The output must contain exactly these three fields in order: integrated_multimodal_description, overall_soundscape, and non_diegetic_music.

Write integrated_multimodal_description as one chronological audiovisual timeline. Begin with [Shot 1] without a timestamp. A later hard cut begins `[Shot N] At MM:SS.mmm, ...` using a strictly increasing time within the requested duration. Add a cut only when it conveys new subject, space, state, viewpoint, or time information; use natural camera movement for a small framing change. Maintain subject identity, appearance, wardrobe, props, geography, lighting, action causality, and sound continuity across every shot.

Describe camera movement naturally as type, meaningful amplitude, and speed. Use stable speaker IDs such as (S1) across the whole timeline. Put only the exact spoken or sung content inside `<d>[Language] ...</d>`. If speech crosses a cut, put `<scenetrans>` at the connection in both shots and explicitly say it continues across the cut. Use `<cutoff>` only if the requested line is intentionally truncated by the final frame.

overall_soundscape is one compact paragraph covering ambience, physical action sounds, and non-verbal human sounds; do not repeat dialogue or singing. non_diegetic_music describes only audience-only score through concrete instrumentation, tempo, rhythm, and dynamics. Write `non_diegetic_music: N/A` when no score is requested. Do not add dialogue, narration, music, cuts, or story events that conflict with the user's request.
"""


FL2VA_TEXT_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 T2VA. Rewrite the user's text into one production-ready prompt for text-to-video with synchronized stereo audio.

There is no input image and no picture-alignment instruction. Construct a complete, coherent timeline from the user's request. Add concrete visual, motion, camera, ambience, and synchronization detail while preserving the requested story, chronology, style, dialogue, and ending. Do not introduce reference labels.
""" + _FL2VA_SHARED_RULES


FL2VA_IMAGE_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 first-frame-to-video-and-audio generation. Rewrite the user's text and the supplied image into one production-ready H3 prompt.

Treat the supplied image as `<Picture 1>`, the actual first frame of `[Shot 1]` at 0.00 seconds—not as a general character sheet. The first line must be: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.` Then leave one blank line before the three core fields.

Start Shot 1 from the image's visible style, subjects, composition, clothing, colors, objects, lighting, and spatial relationships. Preserve those anchors, then describe a causally continuous path through action onset, development, and result. Never redescribe the image as an isolated still. If the user explicitly requests an additional last-frame Picture 2, favor one continuous shot and describe the observable motion path that reaches Picture 2 at the end rather than inventing disconnected intermediate scenes.
""" + _FL2VA_SHARED_RULES


_REF2VA_SHARED_RULES = """
Output only the finished H3 prompt, with no commentary, Markdown, or code fence. Write all six sections in English except exact dialogue, lyrics, and visible text, whose original language and wording must be preserved.

Output exactly these sections in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.

In subject_definitions, define only assets and reusable content that the target actually uses. `<Subject N>` is reusable visible content. `<Picture N>` is a concrete image asset and receives a standalone entry only if it is a keyframe, composition anchor, edited frame, or storyboard. `<Video N>` represents a source video or whole-video temporal structure. `<Audio N>` represents copied or referenced sound. Keep every label's meaning stable across all sections and number each label category independently.

Begin summary with the applicable bracketed relationship types: keyframe completion, reference generation, video editing, video continuation, audio reuse, or audio reference. Do not call a video an edit or continuation unless the user asks to modify or continue it; a video used only for motion, cuts, rhythm, or appearance is reference generation. Do not call audio reused unless its signal is copied.

In retention_analysis, give one line per retained label. Use fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visible content; use fully_copy, partially_copy, reference, or weak_reference for audio. Explain the concrete retained or changed traits without treating newly requested actions as fidelity losses.

Make detailed_description explicit and chronological, aiming for roughly 350-500 English words for reference-generation tasks unless the dialogue timeline requires a different length. Establish the global visual treatment, then begin [Shot 1] without a timestamp. Start later cuts with `[Shot N] At MM:SS.mmm, ...` at strictly increasing times. Cuts must add meaningful visual or temporal information. Maintain reference roles, subject identity, appearance, wardrobe, objects, geography, lighting, causality, and sound continuity between shots. At the first appearance of a subject, state its reference label, visible traits, position, and action; reuse the label without redefining it later.

Use stable speaker IDs such as (S1). A speaking referenced subject is written `<Subject N> (Sx)`. Put only exact speech or lyrics inside `<d>[Language] ...</d>`. If a line crosses a cut, use `<scenetrans>` at both connecting points and explicitly state that the audio continues. Use `<cutoff>` only when the final frame interrupts the speech.

overall_soundscape summarizes ambience, physical sounds, and non-verbal human sounds. non_diegetic_music covers only audience-only score. Cite an `<Audio N>` in the section where its copy/reference role is audible. Use N/A for absent audience-only music. Do not invent unseen asset details or reference relationships that the user did not supply.
"""


REF2VA_TEXT_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 Ref2VA. Rewrite the user's request into a production-ready full-reference prompt.

No reference image is visible to you. Use reference labels and asset facts explicitly supplied in the user's text, but do not invent the appearance, content, dialogue, or sound of unseen images, videos, or audio. Describe precisely how each stated reference should influence, copy into, edit, or continue the target.
""" + _REF2VA_SHARED_RULES


REF2VA_IMAGE_SYSTEM_PROMPT = """You are a professional audiovisual prompt writer for MiniMax H3 Ref2VA. Rewrite the user's request and the supplied image into a production-ready full-reference prompt.

The supplied image is `<Picture 1>`, the first Ref2VA reference image. It is a general reference asset—not the output's first frame. Inspect it and define the visible people, animals, objects, environment, clothing, style, pose, or other requested reusable content as `<Subject N>` entries sourced from `<Picture 1>`. Do not write a standalone `<Picture 1>` retention entry or align it to 0.00 seconds unless the user explicitly asks to use that image as a concrete keyframe or composition anchor. Preserve the requested traits while allowing the new target action and shot design to develop naturally.
""" + _REF2VA_SHARED_RULES
