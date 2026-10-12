K6_INFOS = """
# Kandinsky 6 Video

Kandinsky 6 generates a video together with its synchronized soundtrack (44.1 kHz mono): ambient sound, music, effects and speech with
lip-sync, all from one prompt.

## Inputs

- **Text Prompt** only: the whole clip is generated from the description.
- **Start Image**: the image becomes the first frame and the prompt describes what happens next. Kandinsky 6 has no end frame, video
  continuation or audio input.

## Distilled Models

Pro (30B parameters) and Lite (3B) are distilled for 10 steps: no guidance and no negative prompt. Pro has the best quality and follows
long, detailed prompts better; Lite is about ten times smaller and fits small GPUs.

## Duration and Resolution

The models were trained on 5-second clips (121 frames at 24 fps): keep this duration for the best results. Clips up to 10 seconds
(241 frames) work but follow the prompt less and tend to drift back toward the opening shot. They generate around 864x480 natively:
832x480 is the recommended resolution, larger sizes are slower and less tested. Width and height must be multiples of 16.
"""

K6_PROMPT_INFOS = """
# Kandinsky 6 Prompts

Kandinsky 6 was trained on long, detailed captions. Write one paragraph of about 150 to 300 words in English or Russian: the shot and
setting, a camera sentence ("The camera is static.", "The camera slowly dollies forward."), the characters' appearance, the actions over
five seconds in their order, then light and visual style.

## Speech

Write each spoken line inside `<S>` and `<E>` at the moment it is said, introduced by its speaker:

`She turns to the camera and says: <S>Let's go!<E>`

Each line is its own `<S>...<E>` pair. Keep lines short enough to be spoken within the clip. When nobody speaks, use no speech tags and
say that visible people keep their lips closed.

## Sound

End the prompt with the soundtrack inside `<AUDCAP>` and `<ENDAUDCAP>`: sound sources, music with its instruments, voices' character.
Do not describe sounds in the scene part, and do not repeat the spoken words there.

## Example

`A medium shot of a bakery counter at dawn. The camera is static. A young baker with flour on her apron slides a tray of croissants onto
the counter, looks up at the camera and says: <S>Fresh out of the oven!<E> She smiles and steps back as steam rises from the pastries.
Warm golden light from the window, realistic documentary style. <AUDCAP>Loudly, a cheerful female voice; quietly, a tray scraping on
metal and the hum of an oven.<ENDAUDCAP>`

The Prompt Enhancer writes prompts in this format from a short request.
"""

K6_DEEPY_INFOS = """Generate a video with its synchronized soundtrack (44.1 kHz mono, speech with lip-sync, music, effects) from `prompt`. `image_prompt_type`: empty = text only; `S` = `image_start` becomes the first frame. No end frame, continuation or audio input. Distilled: 10 steps, no guidance or negative prompt. Trained on 121 frames (5 s at 24 fps) around 864x480; use 832x480 and 121 frames (up to 241 frames possible, less consistent)."""

K6_DEEPY_PROMPT_INFOS = """One detailed paragraph (150-300 words, English or Russian): shot and setting, one camera sentence ("The camera is static."), appearance, actions in order over 5 s, light and style. Spoken lines: `<S>exact words<E>` at the moment they are said, after the speaker and a speech verb; no speech tags when nobody speaks. End with the soundtrack inside `<AUDCAP>...<ENDAUDCAP>` (sources, music, voices), with sounds only there. Example: `... looks up and says: <S>Fresh out of the oven!<E> ... <AUDCAP>Loudly, a cheerful female voice; quietly, a tray scraping on metal.<ENDAUDCAP>`"""

_ENHANCER_RULES = """Keep everything the request says: every named object, person, clothing item, number, location and stated visual style. Do not
invent a new plot. Drop technical tags (masterpiece, 4k, best quality, --ar, negative prompts).

Write one continuous paragraph of 150 to 300 words: what is in the frame and the shot size, exactly one sentence that starts with the
camera (the requested movement, otherwise "The camera is static."), the appearance of the characters, the action over the clip duration
in order, then light and visual style. Describe only what is visible, with concrete materials, colors and motion; give every moving
object a direction.

Speech: copy any words the request makes someone say exactly, in their language, and wrap each line as <S>words<E> at the moment it is
spoken, introduced by the speaker and a speech verb right before the tag, for example: The barista looks up and says: <S>Your latte is
ready.<E>. Remove the quotes, change nothing else. If the request names a speech act without its words, write one or two short plausible
lines in the language of the scene, each in its own <S>...<E> pair. Lettering on signs is quoted, not spoken. Animal sounds are not
speech. Without speech, use no <S> or <E> tags, and a visible silent person keeps the lips closed.

Sounds are never described in the scene paragraph. End the answer with one sentence naming the audible sound sources, wrapped in
<AUDCAP> and <ENDAUDCAP>: the loudest source with the word "loudly", music as "music" with its instruments and character, common sound
sources only, no echo, no spoken words. Without speech it names no human voices. The answer always ends with this sound sentence, even for
room tone.

Write in the language of the request (English or Russian) and answer with the prompt only: no title, list or commentary. The whole answer is
one single paragraph without line breaks: the sound sentence follows the description on the same line."""

K6_TEXT_ENHANCER_INSTRUCTIONS = "You are a prompt engineer for a text-to-video model that generates the video and its soundtrack together. You turn a short scene request into one detailed prompt.\n\n" + _ENHANCER_RULES

K6_IMAGE_ENHANCER_INSTRUCTIONS = """You are a prompt engineer for an image-to-video model that generates the video and its soundtrack together. You get a reference image, which is the first frame of the clip, and a short request, and you write one detailed prompt.

The image is the ground truth: describe what it shows with its real colors, materials, style, clothing and setting; if the request names
something the image does not show, it enters the frame during the clip. Quote visible lettering exactly. Focus on the motion, actions,
camera and sound that follow from the image.

""" + _ENHANCER_RULES
