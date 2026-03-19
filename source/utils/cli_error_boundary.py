"""Small CLI error-boundary helper used by smoke tests and scripts."""

from __future__ import annotations

import json
import traceback


def run_with_cli_error_boundary(
    fn,
    *,
    error_message: str = "Command failed",
    debug: bool = False,
    json_output: bool = False,
) -> bool:
    """Run ``fn`` and convert handled exceptions into a boolean success flag."""
    try:
        fn()
        return True
    except Exception as exc:
        if json_output:
            payload = {
                "error": error_message,
                "detail": str(exc),
            }
            if debug:
                payload["traceback"] = traceback.format_exc()
            print(json.dumps(payload))
        elif debug:
            print(traceback.format_exc())
        return False
