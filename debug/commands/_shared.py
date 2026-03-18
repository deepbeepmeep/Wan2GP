"""Shared CLI command helpers."""

from __future__ import annotations

import json


def run_with_error_boundary(action, *, error_message: str, json_output: bool = False, debug: bool = False):
    try:
        action()
        return True
    except Exception as exc:
        if json_output:
            print(json.dumps({"error": error_message, "detail": str(exc)}))
        else:
            print(f"{error_message}: {exc}")
            if debug:
                import traceback

                print(traceback.format_exc())
        return False


__all__ = ["run_with_error_boundary"]
