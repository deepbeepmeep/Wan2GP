# ui.py
#
# UI layer for Frames2Video model in Wan2GP.
# This file defines the Gradio components and binds JSON defaults.

import gradio as gr

def build_ui(json_def):
    """
    Build the UI for Frames2Video.
    json_def: the model's JSON definition (frames2video.json)
    """

    # ------------------------------------------------------------
    # 1. Load defaults from JSON
    # ------------------------------------------------------------
    default_prompt = json_def.get("prompt", "")
    default_image_start = json_def.get("image_start", None)
    default_image_end = json_def.get("image_end", None)

    # ------------------------------------------------------------
    # 2. Build UI components
    # ------------------------------------------------------------
    with gr.Column():

        # Prompt
        prompt = gr.Textbox(
            label="Prompt",
            value=default_prompt,
            lines=2,
            placeholder="Describe the morph or leave blank."
        )

        # Start Image
        image_start = gr.Image(
            label="Start Image",
            value=default_image_start,
            type="pil"
        )

        # End Image
        image_end = gr.Image(
            label="End Image",
            value=default_image_end,
            type="pil"
        )

        # Additional parameters
        frame_num = gr.Slider(1, 300, value=json_def.get("frame_num", 81), label="Frame Count")
        shift = gr.Slider(0, 20, value=json_def.get("shift", 5.0), label="Shift")
        guide_scale = gr.Slider(0, 20, value=json_def.get("guidance_scale", 5.0), label="Guidance Scale")
        steps = gr.Slider(1, 100, value=json_def.get("sampling_steps", 40), label="Sampling Steps")

    # ------------------------------------------------------------
    # 3. Return components to Wan2GP
    # ------------------------------------------------------------
    return {
        "prompt": prompt,
        "image_start": image_start,
        "image_end": image_end,
        "frame_num": frame_num,
        "shift": shift,
        "guide_scale": guide_scale,
        "sampling_steps": steps,
    }
