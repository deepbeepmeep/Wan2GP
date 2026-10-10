"""Instrumental section plans supported by Mothersuperior's YuE2 AR adapter."""

import re


_SECTION_TAG = re.compile(
    r"\[(intro|verse|pre-chorus|chorus|bridge|outro)(?:[ \t]+[0-9]+)?"
    r"(?:[ \t]+([0-9]+:[0-5][0-9]-[0-9]+:[0-5][0-9]))?[ \t]*\]",
    re.IGNORECASE,
)


def normalize_instrumental_prompt(prompt):
    """Lowercase section names and remove indices, preserving times and other text."""
    prompt = re.sub(r"\[instrumental\]", "[instrumental]", prompt, flags=re.IGNORECASE)
    return _SECTION_TAG.sub(
        lambda match: "[" + match[1].lower() + (" " + match[2] if match[2] else "") + "]",
        prompt,
    )


def validate_instrumental_prompt(prompt):
    lines = [(number, line) for number, line in enumerate(prompt.splitlines(), 1) if line.strip()]
    if not lines:
        return "Instrumental plan is empty. Enter [Instrumental] or one section tag per line."
    if len(lines) == 1 and normalize_instrumental_prompt(lines[0][1].strip()) == "[instrumental]":
        return None
    for number, line in lines:
        normalized = normalize_instrumental_prompt(line.strip())
        if normalized == "[instrumental]":
            return f"Invalid instrumental plan at line {number}: {line!r}. [Instrumental] must be used alone, without other section tags or text."
        if not _SECTION_TAG.fullmatch(normalized):
            return f"Invalid instrumental plan at line {number}: {line!r}. Expected one tag: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge] or [Outro]. Numbers such as [Verse 1] are allowed; optional times use [Verse 1 0:15-0:45]. Use section tags only; put instruments and production notes in Music Style."
    return None
