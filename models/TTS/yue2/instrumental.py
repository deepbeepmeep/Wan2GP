"""Instrumental section plans supported by Mothersuperior's YuE2 AR adapter."""

import re


def validate_instrumental_prompt(prompt):
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if lines == ["[instrumental]"]:
        return None
    section = r"\[(?:intro|verse|pre-chorus|chorus|bridge|outro)(?: \d+:[0-5]\d-\d+:[0-5]\d)?\]"
    if lines and all(re.fullmatch(section, line) for line in lines):
        return None
    return "Instrumental mode requires [instrumental] or lowercase section tags such as [intro], [verse], [chorus], [outro], one per line. Optional times use [intro 0:00-0:15]. Put instruments and production notes in Music Style, not Lyrics."
