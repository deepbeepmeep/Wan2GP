# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import help_text, specialities

OVI_HELP = help_text("Ovi Audio and Video", "Generate video and its soundtrack together from text, optionally using a **Start Image** or continuing a source video. Describe both visible events and audible events. Use the selected checkpoint's default window length; longer output is assembled with Sliding Window. Check speech intelligibility, lip timing and window transitions.", '''### Speech Tags and Audio Description

Describe the scene and action. Enclose exact spoken words in `<S>` and `<E>`. Finish the prompt with `Audio:` followed by the voice, ambience and sound effects.

```text
A shopkeeper smiles toward the camera and says <S>Your order is ready.<E> She places a paper bag on the counter. Audio: A warm conversational female voice, a soft paper rustle and quiet shop ambience.
```

Keep dialogue short enough for the clip. These tags request generated speech; they are not a transcript or a reference-voice input.''', "Text/start-image/continuation video with generated soundtrack. Use the checkpoint's window length; sliding windows extend output. No reference-voice input in this workflow.", 'Scene/action description; exact speech inside `<S>...<E>`; finish with `Audio: voice, ambience, effects`. Keep speech within the clip duration.', ["https://github.com/character-ai/Ovi"])
OVI_HELP["specialities"] = specialities("audio-video generation")
