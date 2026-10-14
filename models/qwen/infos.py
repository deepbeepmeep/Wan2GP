# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import EDIT_PROMPT, EDIT_PROMPT_SHORT, IMAGE_PROMPT, IMAGE_PROMPT_SHORT, REFERENCE_PROMPT, help_text, specialities


def qwen_help(architecture):
    if architecture == "qwen_image_layered_20B":
        return help_text("Qwen Image Layer Decomposition", "Supply a **Control Image** to separate into transparent layers. **Number of Layers** chooses the requested decomposition. Layers retain their place on the original canvas so they can be composited and edited separately. Inspect edges and occluded content: the model may infer hidden portions and cannot recover the original design document exactly.", "### Describe the Source Image\n\nUse a factual caption of the supplied image, including foreground objects and background, rather than a request to create a different scene.\n\n**Example:** A blue bicycle in front of a yellow wall, with a potted plant on the left.", "Control image required; batch_size = Number of Layers. Produces aligned RGBA layers. Decomposition/hidden content is inferred, not exact source-document recovery.", "Caption the source scene and its objects. Do not describe a new image to generate.", ["https://github.com/QwenLM/Qwen-Image-Layered"])
    editing = architecture != "qwen_image_20B"
    info = "Generate images from text, including layouts with Chinese or English writing. Quote exact copy and proofread spelling and placement."
    compact = "Text-to-image with Chinese/English text rendering; proofread output."
    prompt, short_prompt = IMAGE_PROMPT, IMAGE_PROMPT_SHORT
    strengths = ["text rendering", "image inpainting", "image outpainting"]
    if editing:
        info = "Use **Reference Images** and an instruction to change appearance, objects, written text or style. Main-scene mode (`KI`) treats the first image as the scene to edit; additional images supply subjects. Describe what each reference contributes and what must remain unchanged."
        compact = "Image editing from ordered references. KI uses scene first, then subjects; I uses subject references. State changes and preservation constraints."
        prompt, short_prompt = EDIT_PROMPT, EDIT_PROMPT_SHORT
        prompt += "\n\n" + REFERENCE_PROMPT
        short_prompt += " " + REFERENCE_PROMPT
        prompt += '\n\nFor writing edits, quote both strings: `Replace "CLOSED" with "OPEN"; keep the sign material, position and lettering style.`'
        strengths += ["image editing", "identity preservation", "style transfer"]
        if architecture in ("qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"):
            info += "\n\nThe Edit Plus variants combine multiple references and offer **Control Image Process** for pose, depth, sketch or edge guidance. Use the appropriate process for your source; describe the finished image, not the map colors."
            compact += " Edit Plus: multiple references and pose/depth/sketch/edge controls."
            strengths += ["control image"]
    info += "\n\n### Masked Edits and Canvas Extension\n\n**Image Inpainting** uses a source and mask to target a region. **Spatial Outpainting** extends the canvas. Use the chosen model's default inpainting method; LanPaint spends additional time refining masked regions. Keep descriptions of untouched content consistent."
    compact += " Inpainting uses source + mask; outpainting extends canvas. LanPaint adds refinement time."
    result = help_text("Qwen Image Editing" if editing else "Qwen Image Generation", info, prompt, compact, short_prompt, ["https://github.com/QwenLM/Qwen-Image"])
    result["specialities"] = specialities(*strengths)
    return result
