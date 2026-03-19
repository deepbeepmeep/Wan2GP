"""Canonical travel-segment queue entrypoint."""

from __future__ import annotations

from pathlib import Path

from source.task_handlers.tasks import task_registry as _registry

SegmentContext = _registry.SegmentContext
GenerationInputs = _registry.GenerationInputs
get_task_params = _registry.get_task_params
get_model_grid_size = _registry.get_model_grid_size


def _resolve_segment_context(task_params_dict: dict, is_standalone: bool, task_id: str) -> SegmentContext:
    original_get_task_params = _registry.get_task_params
    try:
        _registry.get_task_params = get_task_params
        return _registry._resolve_segment_context(
            task_params_dict,
            is_standalone=is_standalone,
            task_id=task_id,
        )
    finally:
        _registry.get_task_params = original_get_task_params


def _resolve_generation_inputs(
    ctx: SegmentContext,
    task_id: str,
    main_output_dir_base: Path,
) -> GenerationInputs:
    original_get_model_grid_size = _registry.get_model_grid_size
    try:
        _registry.get_model_grid_size = get_model_grid_size
        return _registry._resolve_generation_inputs(
            ctx,
            task_id=task_id,
            main_output_dir_base=main_output_dir_base,
        )
    finally:
        _registry.get_model_grid_size = original_get_model_grid_size


def handle_travel_segment_via_queue(*args, **kwargs):
    original_get_task_params = _registry.get_task_params
    original_get_model_grid_size = _registry.get_model_grid_size
    try:
        _registry.get_task_params = get_task_params
        _registry.get_model_grid_size = get_model_grid_size
        return _registry._handle_travel_segment_via_queue_impl(*args, **kwargs)
    finally:
        _registry.get_task_params = original_get_task_params
        _registry.get_model_grid_size = original_get_model_grid_size


__all__ = [
    "GenerationInputs",
    "SegmentContext",
    "_resolve_generation_inputs",
    "_resolve_segment_context",
    "get_model_grid_size",
    "get_task_params",
    "handle_travel_segment_via_queue",
]
