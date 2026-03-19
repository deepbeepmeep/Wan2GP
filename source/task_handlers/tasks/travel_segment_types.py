"""TypedDict definitions for travel segment task payloads.

These provide IDE autocomplete and type checking for the dict-heavy
task_registry.py code. All fields are optional (total=False) since payloads
arrive from the database and may have missing keys.
"""
from __future__ import annotations

from typing import Any, List, Mapping, TypedDict


SEGMENT_PARAM_PRECEDENCE = (
    "individual_segment_params",
    "segment_params",
    "orchestrator_details",
)


class IndividualSegmentParams(TypedDict, total=False):
    """Per-segment override parameters.

    Stored under ``task_params_dict["individual_segment_params"]`` and given
    highest priority when resolving prompt, seed, frame count, images, LoRAs,
    and phase config for a single segment.
    """

    # --- Prompts ---
    enhanced_prompt: str
    base_prompt: str
    negative_prompt: str

    # --- Frame count ---
    num_frames: int
    frame_overlap_from_previous: int
    continuation_config: dict[str, Any]

    # --- Images ---
    start_image_url: str
    end_image_url: str
    input_image_paths_resolved: List[str]

    # --- Generation tuning ---
    seed_to_use: int
    num_inference_steps: int
    guidance_scale: float

    # --- Phase config (per-segment override) ---
    phase_config: dict

    # --- LoRA (per-segment override) ---
    segment_loras: List[dict]


_LIST_FIELDS = {"input_image_paths_resolved", "segment_loras"}
_KNOWN_FIELDS = set(IndividualSegmentParams.__annotations__)


def coerce_individual_segment_params(
    payload: Mapping[str, Any] | None,
    *,
    context: str,
    task_id: str,
    allow_unknown: bool = True,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} (task {task_id}): individual_segment_params must be a mapping")

    normalized = dict(payload)
    for key in _LIST_FIELDS:
        if key in normalized and not isinstance(normalized[key], list):
            raise ValueError(f"{context} (task {task_id}): {key} must be a list")

    if not allow_unknown:
        unknown = sorted(set(normalized) - _KNOWN_FIELDS)
        if unknown:
            raise ValueError(f"{context} (task {task_id}): unknown keys {unknown}")

    return normalized
