"""Structured aligned-column debug cards for debug-mode logging."""

from __future__ import annotations

import reprlib
import textwrap
from typing import Any

TITLE_RULE = "\u2500\u2500\u2500\u2500"
HEAVY_TITLE_RULE = "\u2501\u2501\u2501\u2501"
INDENT = "  "
LABEL_CAP = 24
MAX_LINE_WIDTH = 120
SEPARATOR = " : "

_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = 240
_VALUE_REPR.maxother = 240
_VALUE_REPR.maxlist = 12
_VALUE_REPR.maxdict = 12
_VALUE_REPR.maxset = 12
_VALUE_REPR.maxtuple = 12


class DebugCard:
    """Collect labeled rows and render them as an aligned debug block."""

    def __init__(self, title: str, heavy: bool = False):
        self.title = title
        self.heavy = heavy
        self._rows: list[tuple[str, Any]] = []

    def row(self, label: str, value: Any) -> "DebugCard":
        self._rows.append((str(label), value))
        return self

    # Invariant: keep 0/1-row rendering framed here. RunSummary/essential() relies on
    # DebugCard.render() staying byte-for-byte stable; debug_block fallback logic lives in core.py.
    def render(self) -> str:
        rule = HEAVY_TITLE_RULE if self.heavy else TITLE_RULE
        header = f"{rule} {self.title} {rule}"
        if not self._rows:
            return header

        label_width = min(max(len(label) for label, _ in self._rows), LABEL_CAP)
        prefix = f"{INDENT}{' ' * label_width}{SEPARATOR}"
        value_width = max(20, MAX_LINE_WIDTH - len(prefix))
        lines = [header]

        for label, value in self._rows:
            rendered_label = _truncate_label(label).ljust(label_width)
            value_lines = _wrap_value(value, width=value_width)
            lines.append(f"{INDENT}{rendered_label}{SEPARATOR}{value_lines[0]}")
            continuation_prefix = f"{INDENT}{' ' * label_width}{SEPARATOR}"
            for wrapped in value_lines[1:]:
                lines.append(f"{continuation_prefix}{wrapped}")

        return "\n".join(lines)


def render_card(logger: Any, card: DebugCard, task_id: str | None) -> None:
    """Render a card through logger.debug so interceptor tagging still applies."""
    logger.debug(card.render(), task_id=task_id)


def _truncate_label(label: str) -> str:
    return label[:LABEL_CAP]


def format_value(value: Any) -> str:
    return _VALUE_REPR.repr(value)


def _wrap_value(value: Any, *, width: int) -> list[str]:
    value_text = format_value(value)
    wrapped = textwrap.wrap(
        value_text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=False,
        replace_whitespace=False,
    )
    return wrapped or [""]
