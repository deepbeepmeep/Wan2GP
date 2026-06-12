import copy
import os
import re


METADATA_KEY = "metadata"


_IMAGE_PROMPT_LABELS = {
    "": "Text/new generation",
    "T": "Text/new generation",
    "S": "Start image",
    "E": "End image",
    "V": "Continue from source video",
    "L": "Continue from last generated video",
}


def get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=False):
    if for_ui:
        model_family = model_def.get("group", None)
        if model_family is not None and model_family in families_infos:
            return model_family
    handler = model_types_handlers.get(base_model_type, None)
    if handler is None:
        return "unknown"
    return handler.query_model_family()


def _model_def_finetune(model_def):
    return os.path.basename(os.path.dirname(str(model_def.get("path", "") or ""))).casefold() == "finetunes"


def _choice_values(choice_def):
    if not isinstance(choice_def, dict):
        return []
    values = []
    for choice in choice_def.get("choices", []) or []:
        if isinstance(choice, (list, tuple)):
            values.append(str(choice[1] if len(choice) > 1 else choice[0] if len(choice) > 0 else "") or "")
        else:
            values.append(str(choice or ""))
    values.extend(str(value or "") for value in choice_def.get("selection", []) or [])
    default = choice_def.get("default", None)
    if default is not None:
        values.append(str(default or ""))
    return values


def _choice_values_contain(choice_def, letters):
    return any(any(letter in value for letter in letters) for value in _choice_values(choice_def))


def _normalized_choices(choice_def):
    if not isinstance(choice_def, dict):
        return []
    labels = choice_def.get("labels", {}) or {}
    choices = []
    seen = set()

    def add_choice(label, value):
        value = str(value or "")
        if value in seen:
            return
        seen.add(value)
        choices.append({"label": str(label or value or "None"), "value": value})

    for choice in choice_def.get("choices", []) or []:
        if isinstance(choice, (list, tuple)):
            label = choice[0] if len(choice) > 0 else ""
            value = choice[1] if len(choice) > 1 else label
            add_choice(label, value)
        else:
            add_choice(labels.get(choice, choice), choice)
    for key, label in labels.items():
        add_choice(label, key.replace("V", "").replace("P", ""))
    for value in choice_def.get("selection", []) or []:
        add_choice(labels.get(value, value), value)
    return choices


def normalize_choice_def(choice_def):
    if not isinstance(choice_def, dict):
        return None
    keys = ("default", "label", "letters_filter", "visible", "show_label", "scale", "trigger", "type")
    normalized = {key: choice_def[key] for key in keys if key in choice_def}
    normalized["choices"] = _normalized_choices(choice_def)
    return normalized


def infer_main_outputs(model_def):
    outputs = []
    if model_def.get("audio_only", False):
        return ["audio"]
    if model_def.get("image_outputs", False):
        outputs.append("image")
    if model_def.get("v2i_switch_supported", False) or model_def.get("inpaint_support", False):
        outputs.extend(output for output in ("image", "video") if output not in outputs)
    return outputs or ["video"]


def infer_outputs(model_def):
    outputs = infer_main_outputs(model_def)
    if model_def.get("returns_audio", False) and "audio" not in outputs:
        outputs.append("audio")
    return outputs


def infer_inputs(model_def):
    inputs = ["text"]
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_refs = model_def.get("image_ref_choices", None)
    alt_guide_refs = model_def.get("guide_custom_choices", None)
    guide_preprocessing = model_def.get("guide_preprocessing", None)
    custom_video_selection = model_def.get("custom_video_selection", None)
    image_outputs = bool(model_def.get("image_outputs", False))

    if model_def.get("any_audio_prompt", False):
        inputs.append("audio")
    if "S" in image_prompt_types_allowed or "E" in image_prompt_types_allowed or model_def.get("end_frames_always_enabled", False) or model_def.get("inpaint_support", False) or _choice_values_contain(image_refs, "IKF") or _choice_values_contain(alt_guide_refs, "IKF"):
        inputs.append("image")
    if image_outputs and (_choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(alt_guide_refs, "V")) and "image" not in inputs:
        inputs.append("image")
    if "V" in image_prompt_types_allowed or "L" in image_prompt_types_allowed or (not image_outputs and (_choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(alt_guide_refs, "V") or _choice_values_contain(custom_video_selection, "V"))):
        inputs.append("video")
    return list(dict.fromkeys(inputs))


def infer_media_inputs(model_def):
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_refs = model_def.get("image_ref_choices", None)
    guide_preprocessing = model_def.get("guide_preprocessing", None)
    guide_custom_choices = model_def.get("guide_custom_choices", None)
    custom_video_selection = model_def.get("custom_video_selection", None)
    has_reference = _choice_values_contain(image_refs, "I") or _choice_values_contain(guide_custom_choices, "I")
    single_reference = bool(model_def.get("one_image_ref_needed", False) or model_def.get("one_image_ref_only", False))
    has_control = _choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(guide_custom_choices, "V") or _choice_values_contain(custom_video_selection, "V")
    image_outputs = bool(model_def.get("image_outputs", False))
    return {
        "image": {
            "start": "S" in image_prompt_types_allowed,
            "end": "E" in image_prompt_types_allowed or bool(model_def.get("end_frames_always_enabled", False)),
            "reference": has_reference,
            "single_reference": has_reference and single_reference,
            "multiple_references": has_reference and not single_reference,
            "background": _choice_values_contain(image_refs, "K") or _choice_values_contain(guide_custom_choices, "K"),
            "injected_frames": _choice_values_contain(image_refs, "F") or _choice_values_contain(guide_custom_choices, "F"),
            "control": image_outputs and has_control,
            "mask": image_outputs and (_choice_values_contain(model_def.get("mask_preprocessing", None), "A") or bool(model_def.get("inpaint_support", False))),
        },
        "video": {
            "continue": "V" in image_prompt_types_allowed,
            "last": "L" in image_prompt_types_allowed,
            "control": (not image_outputs) and has_control,
            "mask": (not image_outputs) and _choice_values_contain(model_def.get("mask_preprocessing", None), "A"),
        },
        "audio": {
            "prompt": bool(model_def.get("any_audio_prompt", False)),
            "output": bool(model_def.get("audio_only", False) or model_def.get("returns_audio", False)),
        },
    }


def infer_capabilities(model_def, main_outputs, outputs, inputs, media_inputs):
    image_inputs = media_inputs["image"]
    video_inputs = media_inputs["video"]
    audio_inputs = media_inputs["audio"]
    return {
        "text_to_video": "video" in main_outputs and "text" in inputs,
        "image_to_video": "video" in main_outputs and image_inputs["start"],
        "video_to_video": "video" in main_outputs and (video_inputs["continue"] or video_inputs["control"]),
        "text_to_image": "image" in main_outputs and "text" in inputs,
        "image_to_image": "image" in main_outputs and (image_inputs["start"] or image_inputs["reference"] or image_inputs["control"]),
        "text_to_audio": "audio" in main_outputs and "text" in inputs,
        "audio_to_audio": "audio" in main_outputs and audio_inputs["prompt"],
        "audio_to_video": "video" in main_outputs and audio_inputs["prompt"],
        "audio_output": "audio" in outputs,
        "inpainting": bool(model_def.get("inpaint_support", False) or image_inputs["mask"] or video_inputs["mask"]),
        "outpainting": bool(model_def.get("video_guide_outpainting", False)),
        "reference_images": image_inputs["reference"],
        "background_image": image_inputs["background"],
        "injected_frames": image_inputs["injected_frames"],
        "control_image": image_inputs["control"],
        "control_video": video_inputs["control"],
        "video_continuation": video_inputs["continue"],
        "sliding_window": bool(model_def.get("sliding_window", False)),
        "lora": not bool(model_def.get("no_lora", False)),
    }


def infer_setting_values(model_def):
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_prompt_choices = [{"label": _IMAGE_PROMPT_LABELS.get("", ""), "value": ""}]
    for letter in image_prompt_types_allowed:
        if letter == "T":
            continue
        image_prompt_choices.append({"label": _IMAGE_PROMPT_LABELS.get(letter, letter), "value": letter})
    return {
        "image_prompt_type": {
            "allowed": image_prompt_types_allowed,
            "choices": image_prompt_choices,
        },
        "video_prompt_type": {
            "guide_preprocessing": normalize_choice_def(model_def.get("guide_preprocessing", None)),
            "mask_preprocessing": normalize_choice_def(model_def.get("mask_preprocessing", None)),
            "guide_custom_choices": normalize_choice_def(model_def.get("guide_custom_choices", None)),
            "image_ref_choices": normalize_choice_def(model_def.get("image_ref_choices", None)),
            "custom_video_selection": normalize_choice_def(model_def.get("custom_video_selection", None)),
            "forced": str(model_def.get("set_video_prompt_type", "") or ""),
        },
        "audio_prompt_type": {
            "sources": normalize_choice_def(model_def.get("audio_prompt_type_sources", None)),
            "custom_option": model_def.get("audio_prompt_type_custom_option", None),
        },
        "model_mode": normalize_choice_def(model_def.get("model_modes", None)),
        "sample_solver": normalize_choice_def({"choices": model_def.get("sample_solvers", [])}) if model_def.get("sample_solvers", None) is not None else None,
        "prompt_enhancer": normalize_choice_def(model_def.get("prompt_enhancer_def", None)),
    }


def store_metadata(model_type, model_def, model_types_handlers, families_infos):
    base_model_type = model_def.get("architecture", None) or model_type
    family = get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=True)
    family_label = families_infos.get(family, families_infos.get("unknown", (100, "Unknown")))[1]
    main_outputs = infer_main_outputs(model_def)
    outputs = infer_outputs(model_def)
    inputs = infer_inputs(model_def)
    media_inputs = infer_media_inputs(model_def)
    model_def[METADATA_KEY] = {
        "model_type": model_type,
        "family": family,
        "family_label": family_label,
        "base_model_type": base_model_type,
        "finetune": _model_def_finetune(model_def),
        "main_output": main_outputs,
        "outputs": outputs,
        "inputs": inputs,
        "media_inputs": media_inputs,
        "capabilities": infer_capabilities(model_def, main_outputs, outputs, inputs, media_inputs),
        "setting_values": infer_setting_values(model_def),
    }
    return model_def


def _normalize_filter_values(value, *, split_string=True):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        values = [str(one).strip() for one in value if str(one).strip()]
    elif split_string and isinstance(value, str):
        values = [one.strip() for one in re.split(r"[,|/]", value) if one.strip()]
    else:
        values = [str(value).strip()] if str(value).strip() else []
    return values or None


def _normalize_bool_filter(value):
    if value is None or isinstance(value, str) and len(value.strip()) == 0:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid finetune filter value: {value}")


def _metadata_matches_filter(value, expected):
    expected = _normalize_filter_values(expected)
    if expected is None:
        return True
    if isinstance(value, (list, tuple, set)):
        actual = {str(one).casefold() for one in value}
        return any(one.casefold() in actual for one in expected)
    actual = str(value or "").casefold()
    return any(actual == one.casefold() for one in expected)


def list_model_defs(models_def, *, family=None, base_model_type=None, finetune=None, model_type=None, main_output=None, inputs=None):
    finetune_filter = _normalize_bool_filter(finetune)
    requested_outputs = _normalize_filter_values(main_output)
    requested_inputs = _normalize_filter_values(inputs)
    requested_model_types = _normalize_filter_values(model_type, split_string=False)
    records = []
    for one_model_type, model_def in models_def.items():
        metadata = model_def.get(METADATA_KEY, {})
        if requested_model_types is not None and one_model_type not in requested_model_types:
            continue
        if not _metadata_matches_filter(metadata.get("family"), family) and not _metadata_matches_filter(metadata.get("family_label"), family):
            continue
        if not _metadata_matches_filter(metadata.get("base_model_type"), base_model_type):
            continue
        if finetune_filter is not None and bool(metadata.get("finetune", False)) != finetune_filter:
            continue
        if requested_outputs is not None and not _metadata_matches_filter(metadata.get("main_output", []), requested_outputs):
            continue
        if requested_inputs is not None and not _metadata_matches_filter(metadata.get("inputs", []), requested_inputs):
            continue
        record = copy.deepcopy(model_def)
        record["model_type"] = one_model_type
        records.append(record)
    return records
