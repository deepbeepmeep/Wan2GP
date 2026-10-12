# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import VIDEO_PROMPT, VIDEO_PROMPT_SHORT, EDIT_PROMPT, EDIT_PROMPT_SHORT, help_text, specialities


def hunyuan_help(architecture):
    info = "Generate a video from a detailed text description. Begin with this checkpoint's default resolution, duration and sampling settings."
    compact = "Text-to-video; start with the selected checkpoint's defaults."
    prompt, short_prompt = VIDEO_PROMPT, VIDEO_PROMPT_SHORT
    strengths = []
    source = "https://github.com/Tencent-Hunyuan/HunyuanVideo"
    if "1_5" in architecture:
        source += "-1.5"
    if architecture in ("hunyuan_i2v", "hunyuan_1_5_i2v"):
        info = "Supply a **Start Image** and describe the desired movement. The image establishes the scene and character appearance. Continuing a source video extends its ending; keep the new action compatible with that ending."
        compact = "Start image + motion description. Continuation extends the source ending."
    if architecture.startswith("hunyuan_custom"):
        info = "Supply a **Reference Image** of the person or object and describe the video to create. The reference supplies identity and appearance; the prompt supplies the scene and action. Use a clear reference and avoid conflicting descriptions."
        compact = "Required subject reference + scene/action prompt; preserve subject appearance."
        strengths += ["identity preservation"]
        source = "https://github.com/Tencent-Hunyuan/HunyuanCustom"
        if architecture == "hunyuan_custom_edit":
            info += "\n\nAdd the **Control Video** and a target-region mask for a replacement in an existing scene. Select the mask mode carefully: it determines the region to change or preserve."
            compact += " Custom Edit also needs control video and target mask; check mask direction."
            prompt, short_prompt = EDIT_PROMPT, EDIT_PROMPT_SHORT
            strengths += ["character replacement", "video inpainting"]
    if architecture in ("hunyuan_avatar", "hunyuan_custom_audio"):
        info += "\n\nSupply **driving audio** and the character image to animate speech or singing. The recording determines the words and timing; describe expression and gesture in the prompt."
        compact += " Required character image + driving audio. Audio determines words/timing; prompt sets visual behavior."
        strengths += ["lip sync"]
        if architecture == "hunyuan_avatar":
            info = "Animate a character from its **Reference Image** and **driving audio**. The recording supplies speech/singing and timing; the prompt controls expression, gesture and setting. Use a clearly visible face and clean audio."
            compact = "Character reference + driving audio required; prompt controls expression, gesture and setting."
            source = "https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar"
    if architecture == "hunyuan_1_5_upsampler":
        info = "Supply the **Control Video** to increase its resolution. Describe the source scene consistently so newly generated detail fits it. This is generative enhancement: fine details can change. Use the matching Hunyuan 1.5 upsampler settings."
        compact = "Video upscaler: control video + source-consistent caption. Fine details may change."
        strengths += ["video upscaling"]
    result = help_text("Hunyuan Video", info, prompt, compact, short_prompt, [source])
    result["specialities"] = specialities(*strengths)
    return result
