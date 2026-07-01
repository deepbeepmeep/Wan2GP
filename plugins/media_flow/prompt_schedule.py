from __future__ import annotations

import math
import re

import gradio as gr

TIMED_PROMPT_EXAMPLE = "00:00\nA calm cinematic opening shot.\n\n00:30\nThe mood becomes tense and dramatic."
TIMED_PROMPT_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$")


def parse_time_input(value, *, label: str, allow_empty: bool) -> float | None:
    if value is None:
        return None if allow_empty else 0.0
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise gr.Error(f"{label} must be a finite time value.")
        return max(0.0, float(value))
    text = str(value).strip()
    if len(text) == 0:
        return None if allow_empty else 0.0
    if ":" not in text:
        try:
            seconds = float(text)
            if not math.isfinite(seconds):
                raise ValueError
            return max(0.0, seconds)
        except ValueError as exc:
            raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if not math.isfinite(seconds):
                raise ValueError
            return max(0.0, minutes * 60.0 + seconds)
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        if not math.isfinite(seconds):
            raise ValueError
        return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)
    except ValueError as exc:
        raise gr.Error(f"{label} must be a number of seconds, MM:SS(.xx), or HH:MM:SS(.xx).") from exc


def parse_prompt_schedule(prompt_text: str) -> list[tuple[float, str]]:
    text = str(prompt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) == 0:
        return [(0.0, "")]
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if len(block.strip()) > 0]
    first_line = text.split("\n", 1)[0].strip()
    if len(blocks) <= 1 and not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(first_line):
        return [(0.0, text)]
    schedule: list[tuple[float, str]] = []
    for block in blocks:
        lines = block.split("\n")
        timestamp_line = lines[0].strip()
        if not TIMED_PROMPT_TIMESTAMP_RE.fullmatch(timestamp_line):
            raise gr.Error(
                "Timed prompts must be separated by blank lines, and each block must start with a timestamp like MM:SS(.xx) or HH:MM:SS(.xx).\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        prompt_body = "\n".join(lines[1:]).strip()
        if len(prompt_body) == 0:
            raise gr.Error(
                "Each timed prompt block must contain prompt text after its timestamp.\n\n"
                f"Example:\n{TIMED_PROMPT_EXAMPLE}"
            )
        schedule.append((float(parse_time_input(timestamp_line, label="Timed prompt timestamp", allow_empty=False) or 0.0), prompt_body))
    return sorted(schedule, key=lambda item: item[0])


def resolve_prompt_for_chunk(prompt_schedule: list[tuple[float, str]], chunk_start_seconds: float, default_prompt: str) -> str:
    prompt_text = str(default_prompt or "")
    for start_seconds, scheduled_prompt in prompt_schedule:
        if float(start_seconds) <= float(chunk_start_seconds) + 1e-9:
            prompt_text = scheduled_prompt
        else:
            break
    return prompt_text
