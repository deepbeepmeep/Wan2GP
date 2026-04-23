"""Termination safety wrapper for live RunPod tests."""

from __future__ import annotations

import os

import scripts.live_test as live_test_pkg


def guarded_terminate(pod_id: str | None, api_key: str | None, *, no_terminate: bool) -> bool:
    """Terminate the pod only when neither the CLI flag nor env opt-out is set."""
    if not pod_id or not api_key:
        return False
    if no_terminate or os.getenv("REIGH_LIVE_TEST_NO_TERMINATE") == "1":
        return False
    live_test_pkg.terminate_pod(pod_id, api_key)
    return True


__all__ = ["guarded_terminate"]
