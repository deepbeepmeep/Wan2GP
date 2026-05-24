from __future__ import annotations

import math

import gradio as gr

from shared.utils.loras_mutipliers import parse_loras_multipliers, preparse_loras_multipliers


def coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def coerce_float(value, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not math.isfinite(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def coerce_int(value, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        float_value = float(value)
        result = int(round(float_value)) if math.isfinite(float_value) else int(default)
    except (TypeError, ValueError, OverflowError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def require_float(value, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{label} must be a number.") from exc
    if not math.isfinite(result):
        raise gr.Error(f"{label} must be a number.")
    if minimum is not None and result < minimum:
        raise gr.Error(f"{label} must be at least {minimum:g}.")
    return result


def require_int(value, label: str, *, minimum: int | None = None) -> int:
    try:
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError
        result = int(round(float_value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error(f"{label} must be a number.") from exc
    if minimum is not None and result < minimum:
        raise gr.Error(f"{label} must be at least {minimum}.")
    return result


def get_error_message(exc: BaseException) -> str:
    message = getattr(exc, "message", exc)
    return str(message or "").strip()


def plugin_info(message: str) -> None:
    text = str(message or "").strip()
    if len(text) == 0:
        return
    print(f"[Process Full Video] {text}")
    gr.Info(text)


def get_single_lora_simple_multiplier(settings: dict) -> float | None:
    if not isinstance(settings, dict):
        return None
    activated_loras = settings.get("activated_loras") or []
    if not isinstance(activated_loras, list) or len([lora for lora in activated_loras if len(str(lora).strip()) > 0]) != 1:
        return None
    raw_multiplier = settings.get("loras_multipliers", "")
    if isinstance(raw_multiplier, bool) or raw_multiplier is None or not isinstance(raw_multiplier, (int, float, str)):
        return None
    multiplier_text = str(raw_multiplier).strip()
    if len(multiplier_text) == 0:
        return 1.0
    tokens = preparse_loras_multipliers(multiplier_text)
    if len(tokens) != 1 or str(tokens[0]).strip() != multiplier_text:
        return None
    values, slists, error = parse_loras_multipliers(multiplier_text, 1, 1, nb_phases=coerce_int(settings.get("guidance_phases"), 1, minimum=1))
    if len(error) > 0 or len(values) != 1 or not slists.get("shared", [False])[0] or not isinstance(slists.get("phase1", [None])[0], float):
        return None
    multiplier = float(values[0])
    return multiplier if math.isfinite(multiplier) else None


def get_default_process_strength(process_settings: dict) -> float:
    simple_lora_multiplier = get_single_lora_simple_multiplier(process_settings)
    if simple_lora_multiplier is not None:
        return simple_lora_multiplier
    process_strength = process_settings.get("process_strength")
    if process_strength is None:
        process_strength = process_settings.get("loras_multipliers", 1.0)
    return coerce_float(process_strength, 1.0)
