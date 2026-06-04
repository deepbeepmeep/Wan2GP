from functools import lru_cache
from pathlib import Path


_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "magic_prompt_system_prompts" / "v1.txt"


@lru_cache(maxsize=None)
def _load_sections() -> dict[str, str]:
    sections = {}
    current = None
    lines = []
    for line in _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    if "system" not in sections:
        raise ValueError(f"{_SYSTEM_PROMPT_FILE} has no [SYSTEM] section")
    return sections


IDEOGRAM4_PROMPT_ENHANCER = _load_sections()["system"]


IDEOGRAM4_PROMPT_INFOS = """
# Ideogram 4 Prompt Notes

Ideogram 4 accepts plain text, but it was trained on structured JSON captions. For best quality and layout control, use
JSON or use **Magic Prompt** to turn a short idea into JSON.

## Recommended JSON Shape

Use these top-level keys in this order:

```json
{
  "high_level_description": "One or two sentences describing the whole image.",
  "style_description": {
    "aesthetics": "mood and visual keywords",
    "lighting": "lighting setup",
    "photo": "camera and lens details",
    "medium": "photograph",
    "color_palette": ["#FFFFFF", "#111111"]
  },
  "compositional_deconstruction": {
    "background": "Environment and background description.",
    "elements": [
      {"type": "obj", "bbox": [120, 260, 760, 760], "desc": "A detailed subject description."},
      {"type": "text", "bbox": [80, 100, 180, 900], "text": "EXACT WORDS", "desc": "Typography and placement description."}
    ]
  }
}
```

## Layout And Bboxes

- `compositional_deconstruction` is the most important section for spatial control.
- `bbox` uses normalized coordinates from `0` to `1000`, ordered as `[y_min, x_min, y_max, x_max]`.
- Keep each bbox aligned with its description: if an object is on the left, its `x` values should be low; if it is near the bottom, its `y` values should be high.
- Text elements should use `type: "text"` and include both the exact `text` to render and a `desc` describing font, color, size, and placement.

## Style

- In `style_description`, use either `photo` for photographic prompts or `art_style` for non-photo prompts, not both.
- Include `aesthetics`, `lighting`, and `medium` when you include `style_description`.
- Use uppercase hex colors like `#1B1B2F`. Up to 16 colors can guide the whole image; up to 5 can guide one element.

## Practical Tips

- JSON is optional, but it usually improves typography, composition, color control, and repeatability.
- Keep object descriptions positive and concrete.
- Serialize compactly when possible, with no comments or markdown fences in the prompt itself.
- If writing JSON by hand is tedious, enable **Magic Prompt** and start from a short natural-language idea.
"""
