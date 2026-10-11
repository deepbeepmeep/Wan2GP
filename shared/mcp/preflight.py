# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Read-only preflight using the same request preparation as submission."""

import copy
import json
import math
import re
import subprocess
import time
from pathlib import Path

from shared import extra_settings, resolutions
from shared.api import apply_media_flag_defaults, apply_video_length_duration
from shared.utils.frame_scheduler import has_slash_commands


def task_settings(source):
    tasks = source.get("tasks", [source]) if isinstance(source, dict) else source
    return [task.get("params", task.get("settings", task)) for task in tasks]


def effective_settings(session, supplied):
    model_type = supplied.get("model_type", supplied.get("base_model_type"))
    model_def = session.get_model_def(model_type)
    normalized = copy.deepcopy(supplied)
    apply_media_flag_defaults(normalized, model_def)
    apply_video_length_duration(normalized, model_def)
    effective = session.get_default_settings(model_type)
    effective.update(normalized)
    if "guidance_phases" in effective:
        effective["guidance_phases"], effective["video_prompt_type"] = resolutions.normalize_phase_2_tiling_selection(model_def, effective["guidance_phases"], effective.get("video_prompt_type", ""))
    return model_def, effective


def input_metadata(session, settings):
    from .server import _MCP_MEDIA_SETTING_KEYS, _mcp_media_type
    from PIL import Image
    from shared.utils.video_decode import resolve_media_binary

    result = []
    probe = resolve_media_binary("ffprobe")
    for role in sorted(_MCP_MEDIA_SETTING_KEYS & settings.keys()):
        value = settings[role]
        for item in value if isinstance(value, list) else [value]:
            if not item:
                continue
            path = Path(item)
            record = {"role": role, "path": str(path), "media_type": _mcp_media_type(str(path)), "size_bytes": path.stat().st_size}
            if record["media_type"] == "image":
                with Image.open(path) as image:
                    record.update(width=image.width, height=image.height)
            elif record["media_type"] in {"audio", "video"}:
                if probe:
                    process = subprocess.run([str(probe), "-v", "error", "-show_entries", "stream=codec_type,width,height,sample_rate,channels,avg_frame_rate:format=duration", "-of", "json", str(path)], capture_output=True, text=True, timeout=15)
                    if process.returncode:
                        record["metadata_unavailable"] = process.stderr[-500:]
                    else:
                        record["streams"] = json.loads(process.stdout)
                else:
                    record["metadata_unavailable"] = "ffprobe is not installed in this runtime; preflight will not download it."
            result.append(record)
    return result


def validate_declared_settings(session, supplied):
    model_def, effective = effective_settings(session, supplied)
    resolution = effective.get("resolution")
    if resolution and (not isinstance(resolution, str) or not re.fullmatch(r"[1-9]\d*x[1-9]\d*", resolution)):
        raise ValueError('resolution must be a positive "WIDTHxHEIGHT" string.')
    if resolution and not (model_def.get("audio_only") or model_def.get("mesh_outputs")):
        resolutions.validate_agent_resolution(model_def, resolution, effective.get("guidance_phases", 1), effective.get("video_prompt_type", ""))
    custom = effective.get("custom_settings") or {}
    definitions = []
    if model_def.get("custom_settings") or model_def.get("custom_guide"):
        module = session._ensure_runtime().module
        definitions = module.get_model_custom_settings(model_def)
        if model_def.get("custom_guide"):
            definitions = [*definitions, {**model_def["custom_guide"], "id": "custom_guide", "type": "file"}]
        definitions = [definition for definition in definitions if module.custom_setting_visible(definition, effective.get("video_prompt_type", ""), effective.get("audio_prompt_type", ""))]
    error = extra_settings.validate_inputs(effective, {**model_def, "custom_settings": definitions})
    if error:
        raise ValueError(error)
    for definition in definitions:
        key = definition["id"]
        value = effective.get(key) if definition["type"] == "file" else custom.get(key, definition.get("default"))
        if definition.get("required") and value in (None, "", []):
            raise ValueError(f"{key} is required for the selected input mode.")
        if definition["type"] in {"int", "integer", "float", "number", "slider"}:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"custom_settings.{key} must be finite.")
            error = extra_settings.validate_setting_value(key, value, definition["type"], definition.get("min"), definition.get("max"))
            if error:
                raise ValueError(error)
        choices = definition.get("choices")
        if choices and value is not None and value not in [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in choices]:
            raise ValueError(f"custom_settings.{key} must be one of the model's declared choices.")
    refs = effective.get("image_refs") or []
    if model_def.get("one_image_ref_needed") and len(refs) != 1:
        raise ValueError("This model requires exactly one image_refs input.")
    if (model_def.get("at_least_one_image_ref_needed") or "I" in effective.get("video_prompt_type", "")) and not refs:
        raise ValueError("The selected model/input mode requires image_refs.")


def preflight(session, prepare_generation, prepare_postprocess, settings=None, media=None, process=None, parameters=None, detail="summary"):
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be summary or full.")
    report = {"valid": False, "submitted": False, "checked_at": time.time(), "diagnostics": [], "tasks": [],
              "limitations": ["Snapshot only; submission validates again. No downloads, model loading, inference, GPU reservation or VRAM/runtime prediction.", "Queue state covers this WanGP instance, not GPU use by other processes.", "Requested dimensions/video describe normalized settings, not predicted output. Reference sizing, window scheduling, continuation and trimming can change the saved dimensions or duration; inspect completed media for actual values."]}
    config = session._ensure_runtime().module.server_config
    try:
        if settings is not None:
            if media is not None or process is not None or parameters is not None:
                raise ValueError("Supply settings for generation, or media + process + parameters for post-processing, not both.")
            prepared = prepare_generation(copy.deepcopy(settings))
            for index, (explicit, supplied) in enumerate(zip(task_settings(settings), task_settings(prepared))):
                if str(supplied.get("mode", "")).startswith("edit_"):
                    report["tasks"].append({"index": index, "mode": supplied["mode"], **({"settings": supplied} if detail == "full" else {}), "checks_unavailable": ["Use media + process preflight for processor-specific contracts."]})
                    continue
                model_def, effective = effective_settings(session, supplied)
                model_type = effective.get("model_type", effective.get("base_model_type"))
                declared = extra_settings.iter_defs(model_def, settings=effective)
                inactive = [key for key in explicit if key in declared and not declared[key].visible]
                if model_def.get("custom_settings"):
                    module = session._ensure_runtime().module
                    inactive += [f"custom_settings.{definition['id']}" for definition in module.get_model_custom_settings(model_def) if definition["id"] in (explicit.get("custom_settings") or {}) and not module.custom_setting_visible(definition, effective.get("video_prompt_type", ""), effective.get("audio_prompt_type", ""))]
                schema = session.get_model_schema(model_type)["metadata"]
                task = {"index": index, "model_type": model_type, "settings": effective,
                        "sources": {key: "explicit" if explicit.get(key) is not None else "standing_default_or_inferred" if key in supplied else "model_default" for key in effective},
                        "inactive_settings": inactive, "media_inputs": schema.get("media_inputs", {}),
                        "availability": session.get_model_availability(model_type),
                        "checks_unavailable": ["Ancillary/optional processor downloads and model-specific inference checks are not covered by checkpoint availability.", "Settings copied from profiles/templates are explicit inputs; their original source cannot be inferred."]}
                task["inputs"] = input_metadata(session, supplied)
                task["active_modes"] = {key: effective.get(key, "") for key in ("image_prompt_type", "video_prompt_type", "audio_prompt_type")}
                task["declared_limits"] = {key: schema[key] for key in ("frames_minimum", "frames_maximum", "frames_steps", "sliding_window") if schema.get(key) is not None}
                resolution = effective.get("resolution")
                if resolution and not (model_def.get("audio_only") or model_def.get("mesh_outputs")):
                    task["requested_dimensions"] = dict(zip(("width", "height"), map(int, resolution.split("x"))))
                    block = model_def.get("vae_block_size", 16)
                    assessment = resolutions.assess_resolution(model_def, resolution, effective.get("guidance_phases", 1), effective.get("video_prompt_type", ""))
                    task["resolution_assessment"] = assessment
                    canvas = dict(zip(("width", "height"), resolutions.parse_resolution(assessment["aligned"])))
                    fit = config.get("fit_canvas", 0)
                    task["sizing"] = {"multiple": block, "mode": {0: "pixel_budget", 1: "outer_box", 2: "crop"}[fit], "canvas_dimensions": canvas}
                    visuals = [item for item in task["inputs"] if item["media_type"] in {"image", "video"}]
                    if has_slash_commands([effective["prompt"]]) or effective.get("video_guide_outpainting", "#") not in ("", "#"):
                        task["checks_unavailable"].append("Initial dimensions with prompt scheduling or outpainting are resolved during generation.")
                    elif not visuals:
                        task["sizing"]["initial_dimensions"] = canvas.copy()
                    elif len(visuals) == 1 and visuals[0]["role"] == "image_start" and "S" in effective.get("image_prompt_type", ""):
                        from shared.utils.utils import calculate_new_dimensions
                        source = visuals[0]
                        height, width = calculate_new_dimensions(canvas["height"], canvas["width"], source["height"], source["width"], fit, block_size=block)
                        task["sizing"]["initial_dimensions"] = {"width": width, "height": height}
                    else:
                        task["checks_unavailable"].append("Initial dimensions depend on reference/control inputs and are resolved during generation.")
                    task["sizing"]["scope"] = "Initial canvas/input sizing, before model-specific processing or optional upscaling; later windows may use other inputs."
                fps = float(effective.get("force_fps") or 0) or schema.get("fps")
                if effective.get("video_length") and fps and not (model_def.get("audio_only") or model_def.get("image_outputs") or model_def.get("mesh_outputs")):
                    task["requested_video"] = {"frames": int(effective["video_length"]), "fps": float(fps), "duration_seconds": int(effective["video_length"]) / float(fps)}
                if model_def.get("audio_only") and effective.get("duration_seconds"):
                    task["target_audio_duration_seconds"] = effective["duration_seconds"]
                selected = effective.get("activated_loras") or []
                if selected:
                    available = session.list_loras(model_type)
                    module = session._ensure_runtime().module
                    local_loras = {Path(name) for name in available["loras"]}
                    missing, downloadable = [], []
                    for name in selected:
                        path = Path(module.get_lora_local_path(None, name))
                        if path in local_loras or (path.is_absolute() and path.is_file()):
                            continue
                        if available["supported"] and name.startswith(("http:", "https:")):
                            downloadable.append(name)
                        else:
                            missing.append(name)
                    task["missing_loras"] = missing
                    if task["missing_loras"]:
                        report["diagnostics"].append({"severity": "error", "task": index, "message": f"LoRAs not found in this model's catalog: {task['missing_loras']}"})
                    if downloadable:
                        task["downloadable_loras"] = downloadable
                        report["diagnostics"].append({"severity": "warning", "task": index, "message": f"LoRAs are not installed locally; execution will attempt to download them. Remote availability was not checked: {downloadable}"})
                if detail == "summary":
                    task.pop("settings")
                    task.pop("sources")
                    # Show normalization and injected defaults, not a second copy of the request.
                    task["adjustments"] = {key: value for key, value in effective.items() if (key in supplied or key in task["active_modes"] and value) and explicit.get(key) != value and key not in inactive}
                    resolved_text = {key: len(task["adjustments"].pop(key)) for key in ("prompt", "alt_prompt", "negative_prompt") if isinstance(task["adjustments"].get(key), str)}
                    if resolved_text:
                        task["resolved_text_lengths"] = resolved_text
                    if "custom_settings" in task["adjustments"]:
                        task["adjustments"]["custom_settings"] = {key: value for key, value in task["adjustments"]["custom_settings"].items() if f"custom_settings.{key}" not in inactive}
                    task["media_inputs"] = {kind: inputs if kind == "custom_file" else {key: value for key, value in inputs.items() if value} for kind, inputs in task["media_inputs"].items() if any(inputs.values())}
                    task["active_modes"] = {key: value for key, value in task["active_modes"].items() if value}
                report["tasks"].append(task)
                if not task["availability"]["available"]:
                    report["diagnostics"].append({"severity": "warning", "task": index, "message": "Model checkpoint availability is partial or missing; execution may require downloads."})
        else:
            if not media or not process:
                raise ValueError("Provide settings, or both media and process with optional parameters.")
            from .server import _is_media_id
            task, details = prepare_postprocess(media_id=media if _is_media_id(media) else None, path=None if _is_media_id(media) else media, process=process, parameters=parameters)
            report["tasks"].append({**details, **({"settings": task} if detail == "full" else {}), "checks_unavailable": ["Processor weight/download availability and runtime memory requirements."]})
        report["valid"] = not any(item["severity"] == "error" for item in report["diagnostics"])
    except (ValueError, TypeError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        report["diagnostics"].append({"severity": "error", "message": str(exc)})
    report["queue"] = session.list_queue()
    if detail == "summary":
        report["queue"] = {key: value for key, value in report["queue"].items() if key.endswith("_count")}
    report["runtime_config"] = {key: config[key] for key in ("transformer_quantization", "text_encoder_quantization", "mixed_precision", "vae_precision", "profile", "attention_mode") if key in config}
    return report
