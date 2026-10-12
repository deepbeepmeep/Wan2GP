# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import help_text, specialities

MAGI_HELP = help_text("MagiHuman Talking Characters", "Animate a **Start Image** into a human-centered video with speech and sound. Choose text-driven soundtrack generation, or supply **Driving Audio** so that its recording sets the words and timing. A stable portrait or upper-body composition is a good starting point; large body actions and moving cameras are less suited to this workflow. Sliding windows can extend a performance; inspect identity and voice continuity. Distilled and full models use their own sampling presets.", '''### Performance, Dialogue and Sound

Write the visual description in **English**, using chronological facial, head and shoulder movements. For a detailed performance, aim for about **150–200 words** covering appearance, expression, vocal delivery and a stable camera. Keep the spoken words in their original language. Add **Dialogue:** with a speaker description and language, then **Background Sound:**.

**Short example:**

```text
A woman in a green sweater sits at a desk, facing a fixed camera. She smiles and calmly says "The parcel is ready." Soft daylight, subtle head movement.

Dialogue:
Woman wearing a green sweater, English: "The parcel is ready."

Background Sound:
Quiet room tone.
```

For supplied Driving Audio, match the description to that performance instead of asking for conflicting dialogue.''', "Start image + prompt generate synchronized video/audio, or audio_prompt_type A uses audio_guide as driving soundtrack. Best with a stable portrait and small expressive movements. Sliding windows extend performance; check continuity.", 'Use an English visual performance paragraph (about 150–200 words for detail), then `Dialogue:` with `speaker description, language: "exact words"` in the spoken language, then `Background Sound:`. Keep camera/body stable; match supplied driving audio.', ["https://github.com/SandAI-org/daVinci-MagiHuman"])
MAGI_HELP["specialities"] = specialities("audio-video generation", "lip sync")
