BERNINI_PROMPT_INFOS = """## Bernini Prompt Tips
Use direct editing language. Say what should change, where it should appear, and what must remain unchanged.

### Referring to Inputs
- Use `source video` for the control video or original clip.
- Use `reference image` when there is one image.
- With several images, refer to them by order: `the first reference image`, `the second reference image`, `the third reference image`.
- Assign each reference a role in the prompt. For example: `Use the first reference image as the jacket design and the second reference image as the logo on the back.`

### Good Prompt Shapes
- **Video edit:** `Replace the person's blue shirt in the source video with the red floral shirt from the reference image. Preserve the person's face, body shape, pose, motion, camera movement, lighting, shadows, and background. Keep the shirt stable with no flicker.`
- **Multiple references:** `Dress the person in the source video with the coat from the first reference image and the scarf from the second reference image. Match the original perspective, body motion, fabric folds, and scene lighting. Do not change the face, hair, hands, background, or camera framing.`
- **Object insertion:** `Add the handbag from the reference image onto the table at the right side of the source video. Keep it the same scale across frames, aligned with the table perspective, with natural contact shadows and no sliding.`
- **Style transfer:** `Apply the watercolor painting style from the reference image to the source video while preserving the original composition, object shapes, motion, and camera path.`
- **Reference-to-video:** `Generate a video of the subject from the first reference image walking through a rainy street. Use the outfit from the second reference image. Keep the subject identity consistent, with realistic steps, wet reflections, and stable clothing details.`
- **Text-only generation:** `A close-up video of a ceramic teapot on a wooden table, morning sunlight through a window, slow camera push-in, steam rising gently, realistic reflections and shadows.`

### Practical Tips
- Put the requested edit first, then the preservation constraints.
- Mention materials, colors, placement, scale, lighting, and motion when they matter.
- For local edits, explicitly name untouched regions: `Do not change the face, hands, background, camera framing, or original motion.`
- Avoid vague prompts like `make it better`; use concrete nouns and visible constraints."""

BERNINI_1_3B_INFOS = """## Bernini 1.3B Notes
This smaller variant is best for simple or localized tasks such as style transfer, subtitle or watermark removal, garment or object swaps, and local edits.

It is less suitable than the 14B Bernini model for complex human generation, large motion changes, dense multi-reference composition, or edits that require many interacting objects."""


def get_bernini_prompt_infos(base_model_type):
    return BERNINI_PROMPT_INFOS


def get_bernini_infos(base_model_type):
    return BERNINI_1_3B_INFOS if base_model_type == "bernini_1.3B" else None
