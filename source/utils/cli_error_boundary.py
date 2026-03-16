"""Small CLI error-boundary helper used by smoke tests and scripts."""

from __future__ import annotations


def run_with_cli_error_boundary(fn, *, error_message: str = "Command failed") -> bool:
    """Run ``fn`` and convert handled exceptions into a boolean success flag."""
    try:
        fn()
        return True
    except Exception:
        return False
