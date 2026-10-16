# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Explicit generation-context placeholders for enhancer system instructions."""

import math
import re

from shared.resolutions import parse_resolution


_PLACEHOLDER = re.compile(r"(\\)?\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")


def uses_placeholder(template, name):
    return any(not match[1] and match[2] == name for match in _PLACEHOLDER.finditer(template))


def build_context(enhancer_kwargs, is_image, audio_only):
    inputs = enhancer_kwargs or {}
    context = {}
    if not audio_only and inputs.get("resolution") is not None:
        width, height = parse_resolution(inputs["resolution"])
        divisor = math.gcd(width, height)
        context.update(width=width, height=height, resolution=f"{width}x{height}", aspect_ratio=f"{width // divisor}:{height // divisor}", half_width=width / 2, half_height=height / 2)
    if audio_only:
        duration = inputs.get("duration_seconds")
        if duration is not None and duration > 0:
            context["duration_seconds"] = duration
    elif not is_image:
        for name in ("fps", "frame_count"):
            if inputs.get(name) is not None:
                context[name] = inputs[name]
        if inputs.get("video_duration_seconds") is not None:
            context["duration_seconds"] = inputs["video_duration_seconds"]
    return context


def render_system_prompt(template, context=None, duration_seconds=None, *, inject_duration=False, duration_instructions=""):
    values = dict(context or {})
    if duration_seconds is not None:
        values["duration_seconds"] = duration_seconds
        if "fps" in values:
            values["frame_count"] = round(duration_seconds * values["fps"])

    duration_requested = False

    def replace(match):
        nonlocal duration_requested
        if match[1]:
            return match[0][1:]
        name = match[2]
        if name not in values:
            raise ValueError(f"Prompt enhancer template placeholder '{{{{{name}}}}}' is unknown or unavailable for this generation.")
        duration_requested |= name == "duration_seconds"
        value = values[name]
        return f"{value:.6f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)

    rendered = _PLACEHOLDER.sub(replace, template)
    if inject_duration and not duration_requested:
        rendered += f"\nThis window lasts {duration_seconds:g} seconds. Keep the described action within this duration." if duration_seconds is not None else duration_instructions
    return rendered
