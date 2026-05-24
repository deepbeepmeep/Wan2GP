from __future__ import annotations

from typing import Any


def get_system_handler(name: str | None) -> Any:
    name = str(name or "").strip().lower()
    if name == "flashvsr":
        from postprocessing.flashvsr.process_handler import HANDLER
        return HANDLER
    return None
