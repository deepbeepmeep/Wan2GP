"""Contracts for join transition payload exchange."""

from __future__ import annotations

import json


_INT_FIELDS = (
    "transition_index",
    "frames",
    "gap_frames",
    "gap_from_clip1",
    "gap_from_clip2",
    "context_from_clip1",
    "context_from_clip2",
    "blend_frames",
)


def build_transition_output_payload(**kwargs) -> dict:
    payload = dict(kwargs)
    for field in _INT_FIELDS:
        if field in payload and payload[field] is not None:
            payload[field] = int(payload[field])
    return payload


def parse_transition_output_payload(
    *,
    raw_output: str,
    transition_index: int,
    default_blend_frames: int,
    default_gap_from_clip1: int,
    default_gap_from_clip2: int,
) -> dict:
    if not str(raw_output).lstrip().startswith("{"):
        raise ValueError("transition output must be JSON; legacy raw URL output is no longer supported")
    payload = json.loads(raw_output)
    payload.setdefault("transition_index", transition_index)
    payload.setdefault("blend_frames", default_blend_frames)
    payload.setdefault("gap_from_clip1", default_gap_from_clip1)
    payload.setdefault("gap_from_clip2", default_gap_from_clip2)
    return build_transition_output_payload(**payload)


__all__ = ["build_transition_output_payload", "parse_transition_output_payload"]
