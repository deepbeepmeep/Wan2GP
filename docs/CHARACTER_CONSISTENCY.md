# Character & Environment Consistency in WanGP

This project now includes the **Character & Environment Consistency Studio** plugin. It is designed to make WanGP behave more like commercial "character" and "ingredients" workflows: create a reusable pack, keep locked identity and environment prompts, attach reference images, export WanGP settings, and generate repeated shots with the same character and world anchors.

## What the Commercial Tools Are Doing

Google Flow / Veo 3.1 uses an "Ingredients to Video" style workflow. The public Gemini API docs say Veo 3.1 accepts up to three reference images of a person, character, or product to preserve appearance in the output video, and the Google Cloud prompt guide recommends generating character and setting "ingredients" before composing each shot.

Higgsfield exposes an explicit character workflow. Their character page asks users to upload multiple-angle photos to train a character, while their Soul ID guidance recommends 10-20 clear photos and reusing the stable avatar across image and video work.

OpenArt exposes a reusable AI Character feature. Its product page describes creating, customizing, and reusing consistent characters from prompts, references, or presets, then using them inside image and video workflows.

The common pattern is:

1. Make reference assets first: face, body, outfit, side/back angles, props, and optionally setting/style.
2. Keep short locked identity and environment descriptions and repeat them every shot.
3. Use reference-aware models or trained identity assets rather than prompt-only generation.
4. For long stories, split into short shots and re-anchor the character and world each shot.
5. Keep lighting, setting, prop, and style language stable unless the shot intentionally changes it.

## Best WanGP Model Choices

Use **Bernini-R 14B** when you want the closest "ingredients" behavior with multiple reference images. It is the default in the plugin because it supports reference images directly and is derived from Wan 2.2.

Use **VACE Stand-In 14B** when the character is a human face and you also want VACE-style control. Put body/outfit references first and a close-up face reference last.

Use **Stand-In 14B** for simpler face identity transfer. Use one clean close-up face image on a white or simple background.

Use **SCAIL-2 14B** when motion is the main problem: a character image plus a driving/control video. This is strongest for dance, body motion, and repeatable movement.

Use **JoyAI-Echo** when the goal is a connected multi-shot story with memory commands. It is not reference-image-first, but it can carry compact memories across windows.

## How to Use the Plugin

1. Start WanGP and open the **Characters** tab.
2. Add a character name, locked identity prompt, and locked environment/world prompt.
3. Upload 2-4 clean references for Bernini, or follow the model-specific guidance above.
4. Add one or more shot prompts. Separate shots with a blank line.
5. Click **Preview Settings** to inspect the generated WanGP JSON.
6. Click **Generate First Shot** for a smoke test.
7. Click **Export WanGP Settings** to save a single settings JSON or a multi-shot manifest under `character_packs/`.

The exported settings can be processed from the UI, CLI, or Python API.

## RunPod / Jupyter API Pattern

```python
from pathlib import Path
from shared.api import init

session = init(
    root=Path("/workspace/Wan2GP-main"),
    cli_args=["--attention", "sage2", "--profile", "4"],
)

settings_path = Path("/workspace/Wan2GP-main/character_packs/my-character-wan2gp-settings.json")
job = session.submit(settings_path)
result = job.result()

print(result.success)
print(result.generated_files)
print([str(error) for error in result.errors])
```

## Reference Quality Checklist

- Use clear, high-resolution images.
- Include a front face, one three-quarter/body image, and one outfit/prop image.
- Add environment references when the location, room, vehicle, product set, or world design must stay consistent.
- Avoid heavy filters, extreme expressions, occluding sunglasses, and mixed ages/outfits unless they are part of the character.
- For VACE/Stand-In, use background removal or simple white backgrounds for human/object references.
- Test short clips first, then batch shots after the character is stable.
