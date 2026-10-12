# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import IMAGE_PROMPT, IMAGE_PROMPT_SHORT, VIDEO_PROMPT, VIDEO_PROMPT_SHORT, help_text


def kandinsky_help(architecture, definition):
    image = definition.get("image_outputs", False)
    i2v = architecture.endswith("i2v")
    info = "Generate an image from a natural-language description." if image else "Generate a short silent video from a natural-language description."
    if i2v:
        info = "Animate the required **Start Image**. Describe movement compatible with the subject and composition in that image. This model does not generate a soundtrack."
    info += " Use this entry's default duration, resolution and sampling settings: Lite, Pro and distilled variants have different intended settings. Choose an entry trained for the desired duration instead of assuming any length is equally reliable."
    return help_text("Kandinsky 5", info, IMAGE_PROMPT if image else VIDEO_PROMPT, ("Image generation." if image else "Start image required; prompt describes motion." if i2v else "Text-to-video, no audio.") + " Use the entry's duration/resolution and full/distilled preset.", IMAGE_PROMPT_SHORT if image else VIDEO_PROMPT_SHORT, ["https://github.com/kandinskylab/kandinsky-5"])
