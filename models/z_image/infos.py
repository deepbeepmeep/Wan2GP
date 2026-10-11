# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import IMAGE_PROMPT, IMAGE_PROMPT_SHORT, help_text, specialities


def z_image_help(architecture):
    info = "Generate an image from English or Chinese text. The Base and Turbo entries use different sampling defaults; keep the selected checkpoint's preset."
    short = "English/Chinese text-to-image; keep Base/Turbo sampling defaults matched to the checkpoint."
    strengths = []
    if "control" in architecture:
        info += "\n\n### Structure Guidance\n\nSupply a **Control Image** and choose pose, depth, edges or the model's raw control format. The image supplies structure; the text describes the desired appearance. **Control Strength** adjusts adherence. Do not select raw format for an ordinary photo when you intend to extract a pose or depth map."
        short += " Control image + pose/depth/edge/raw process. Prompt describes finished appearance; Control Strength sets adherence."
        strengths += ["control image"]
    if architecture in ("z_image_control2", "z_image_control2_1"):
        info += "\n\n**Image Inpainting** uses a source, mask and replacement description for local edits. These Control v2 selections support masks; the original Control entry does not expose this workflow."
        short += " Control v2: source + mask for inpainting."
        strengths += ["image inpainting"]
    result = help_text("Z-Image", info, IMAGE_PROMPT, short, IMAGE_PROMPT_SHORT, ["https://github.com/Tongyi-MAI/Z-Image"])
    result["specialities"] = specialities(*strengths)
    return result
