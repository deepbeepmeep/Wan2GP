"""Image-aware enhancer labels derived from the selected model's inputs."""


INPUT_KEYS = ("image_mode", "video_prompt_type", "image_prompt_type", "image_start", "image_end", "image_refs", "image_guide", "image_mask_guide", "video_source")


def has_media(value):
    return bool(value) if isinstance(value, (str, list, tuple, dict)) else value is not None


def image_input_names(model_def, image_mode=None, video_prompt_type=None, inputs=None):
    from models.model_metadata import infer_media_inputs

    if image_mode is None:
        image_mode = int(bool(model_def.get("image_outputs", False)))
    if image_mode > 0 and model_def.get("guide_custom_choices_image") is not None:
        model_def = {**model_def, "guide_custom_choices": model_def["guide_custom_choices_image"]}
    media_inputs = infer_media_inputs(model_def)
    capabilities = media_inputs["image"]
    flags = video_prompt_type or ""
    active = inputs is not None
    image_flags = (inputs.get("image_prompt_type") or "") if active else ""
    start = not active or "S" in image_flags and has_media(inputs.get("image_start")) and not model_def.get("fake_start_image", False)
    continuation = not active or "L" in image_flags or "V" in image_flags and has_media(inputs.get("video_source"))
    names = []
    if image_mode == 0 and not model_def.get("audio_only", False):
        if capabilities["start"] and start or media_inputs["video"]["continue"] and continuation:
            names.append("Start Image")
        if capabilities["end"] and (not active or "E" in image_flags and has_media(inputs.get("image_end"))):
            names.append("End Images")
    references = capabilities["reference"] or capabilities["injected_frames"]
    if image_mode == 2:
        references = bool(model_def.get("inpaint_with_image_ref", False))
    if references and (not active or "I" in flags and has_media(inputs.get("image_refs"))):
        if "F" in flags:
            names.append("Injected Frames")
        else:
            if "K" in flags:
                names.append("Main Ref. Image")
            if "K" not in flags or not capabilities["single_reference"] and (not active or len(inputs["image_refs"]) > 1):
                names.append("Ref. Images")
    control = inputs.get("image_guide") if active else None
    editor = inputs.get("image_mask_guide") if active else None
    if isinstance(editor, dict) and "A" in flags and "U" not in flags:
        control = editor.get("background", control)
    if image_mode > 0 and (capabilities["control"] or media_inputs["video"]["control"] or image_mode == 2):
        if not active or "V" in flags and has_media(control):
            names.append("Control Image")
    return " / ".join(names)


def default_labels(model_def, image_mode=None, video_prompt_type=None, inputs=None):
    names = image_input_names(model_def, image_mode, video_prompt_type, inputs)
    image_label = "Based on Text Prompt and Images" + (f" ({names})" if names else "")
    if inputs is not None and not names:
        image_label = "Based on Text Prompt (No Images Selected)"
    return {"T": "Based on Text Prompt Content", "TI": image_label}


def resolve_definition(model_def, image_mode=None, video_prompt_type=None, inputs=None):
    definition = model_def.get("prompt_enhancer_def")
    if definition is None:
        return None
    names = image_input_names(model_def, image_mode, video_prompt_type, inputs)
    def format_label(label):
        if inputs is not None and not names and "{image_inputs}" in label:
            label = label.replace(" and {image_inputs}", "").replace(" + {image_inputs}", "")
            return label.replace("{image_inputs}", "Images") + " (No Images Selected)"
        return label.replace("{image_inputs}", names or "Images")
    return {**definition, "labels": {mode: format_label(label) for mode, label in definition.get("labels", {}).items()}}
