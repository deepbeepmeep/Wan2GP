# source/task_types.py
"""
Centralized task type definitions for the Headless-Wan2GP system.

This module provides the single source of truth for all task type classifications.
Import from here instead of duplicating these definitions across modules.
"""
from dataclasses import dataclass
from typing import FrozenSet, Dict, Optional


@dataclass(frozen=True)
class TaskTypeMeta:
    default_model: str
    is_direct_queue: bool
    is_wgp_output: bool
    allows_empty_prompt: bool = False
    forced_video_length: Optional[int] = None

# =============================================================================
# WGP Generation Task Types
# =============================================================================
# Task types that represent actual WGP video/image generation tasks.
# Used for output file organization and routing.

WGP_TASK_TYPES: FrozenSet[str] = frozenset({
    # VACE models
    "vace", "vace_21", "vace_22",
    # Flux model
    "flux",
    # Text-to-video models
    "t2v", "t2v_22", "wan_2_2_t2i",
    # Image-to-video models
    "i2v", "i2v_22",
    # Other video models
    "hunyuan", "ltxv", "ltx2",
    # Qwen image tasks
    "qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit",
    # Specialized handlers that enqueue WGP tasks
    "inpaint_frames",
    # Generic generation
    "generate_video",
    # Z image models
    "z_image_turbo", "z_image_turbo_i2i",
})

# =============================================================================
# Direct Queue Task Types
# =============================================================================
# Task types that can be directly submitted to HeadlessTaskQueue without
# orchestration/special handling. These bypass the task_registry handlers.

DIRECT_QUEUE_TASK_TYPES: FrozenSet[str] = frozenset({
    # VACE models
    "vace", "vace_21", "vace_22",
    # Flux model
    "flux",
    # Text-to-video models
    "t2v", "t2v_22", "wan_2_2_t2i",
    # Image-to-video models
    "i2v", "i2v_22",
    # Other video models
    "hunyuan", "ltxv", "ltx2",
    # Generic generation
    "generate_video",
    # Qwen image tasks
    "qwen_image_edit", "qwen_image_hires", "qwen_image_style",
    "image_inpaint", "annotated_image_edit",
    # Text-to-image tasks (no input image required)
    "qwen_image", "qwen_image_2512", "z_image_turbo",
    # Image-to-image tasks
    "z_image_turbo_i2i",
})

# =============================================================================
# Task Type to Model Mapping
# =============================================================================
# Default model to use for each task type when not explicitly specified.
# This maps task types to their canonical WGP model identifiers.

TASK_TYPE_TO_MODEL: Dict[str, str] = {
    # Generic/default generation
    "generate_video": "t2v",
    # VACE models
    "vace": "vace_14B_cocktail_2_2",
    "vace_21": "vace_14B",
    "vace_22": "vace_14B_cocktail_2_2",
    # Text-to-video / image models
    "wan_2_2_t2i": "t2v_2_2",
    "t2v": "t2v",
    "t2v_22": "t2v_2_2",
    # Flux model
    "flux": "flux",
    # Image-to-video models
    "i2v": "i2v_14B",
    "i2v_22": "i2v_2_2",
    # Other video models
    "hunyuan": "hunyuan",
    "ltxv": "ltxv_13B",
    "ltx2": "ltx2_19B",
    # Segment/inpaint handlers (use lightning baseline)
    "join_clips_segment": "wan_2_2_vace_lightning_baseline_2_2_2",
    "inpaint_frames": "wan_2_2_vace_lightning_baseline_2_2_2",
    # Qwen image tasks
    "qwen_image_edit": "qwen_image_edit_20B",
    "qwen_image_hires": "qwen_image_edit_20B",
    "qwen_image_style": "qwen_image_edit_20B",
    "image_inpaint": "qwen_image_edit_20B",
    "annotated_image_edit": "qwen_image_edit_20B",
    # Text-to-image tasks
    "qwen_image": "qwen_image_edit_20B",
    "qwen_image_2512": "qwen_image_2512_20B",
    "z_image_turbo": "z_image",
    # Image-to-image tasks
    "z_image_turbo_i2i": "z_image_img2img",
}


TASK_TYPE_CATALOG: Dict[str, TaskTypeMeta] = {
    task_type: TaskTypeMeta(
        default_model=default_model,
        is_direct_queue=task_type in DIRECT_QUEUE_TASK_TYPES,
        is_wgp_output=task_type in (WGP_TASK_TYPES | frozenset({"qwen_image_hires"})),
        allows_empty_prompt=task_type in {
            "qwen_image_edit",
            "qwen_image_hires",
            "qwen_image_style",
            "image_inpaint",
            "annotated_image_edit",
            "qwen_image",
            "qwen_image_2512",
            "z_image_turbo_i2i",
        },
        forced_video_length=1 if task_type == "wan_2_2_t2i" else None,
    )
    for task_type, default_model in TASK_TYPE_TO_MODEL.items()
}

# Keep the legacy constant projections derived from the catalog.
WGP_TASK_TYPES = frozenset(task_type for task_type, meta in TASK_TYPE_CATALOG.items() if meta.is_wgp_output)
DIRECT_QUEUE_TASK_TYPES = frozenset(task_type for task_type, meta in TASK_TYPE_CATALOG.items() if meta.is_direct_queue)


def get_default_model(task_type: str) -> str:
    """
    Get the default model for a given task type.

    Args:
        task_type: The task type identifier

    Returns:
        The default model identifier, or "t2v" as fallback
    """
    return TASK_TYPE_TO_MODEL.get(task_type, "t2v")


def is_wgp_task(task_type: str) -> bool:
    """Check if a task type is a WGP generation task."""
    return task_type in WGP_TASK_TYPES


def is_direct_queue_task(task_type: str) -> bool:
    """Check if a task type can be directly queued."""
    return task_type in DIRECT_QUEUE_TASK_TYPES


def allows_empty_prompt(task_type: str) -> bool:
    meta = TASK_TYPE_CATALOG.get(task_type)
    return bool(meta and meta.allows_empty_prompt)


def forced_video_length_for_task(task_type: str) -> Optional[int]:
    meta = TASK_TYPE_CATALOG.get(task_type)
    return None if meta is None else meta.forced_video_length
