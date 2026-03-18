"""Worker post-generation policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class WorkerPostGenerationRequest:
    task_id: str
    task_type: str
    normalized_task_params: dict[str, Any]
    output_location_to_db: str
    image_download_dir: str | None
    main_output_dir_base: Path


def apply_worker_post_generation_policy(
    *,
    request: WorkerPostGenerationRequest,
    chain_handler: Callable[..., tuple[bool, str, str | None]],
    relocate_output: Callable[..., str],
    log_error: Callable[[str], None],
) -> str:
    output_path = request.output_location_to_db
    chain_details = request.normalized_task_params.get("travel_chain_details") or {}
    if chain_details.get("enabled"):
        ok, message, chained_output = chain_handler(
            actual_wgp_output_video_path=output_path,
            normalized_task_params=request.normalized_task_params,
            image_download_dir=request.image_download_dir,
            task_id=request.task_id,
        )
        if ok and chained_output:
            output_path = chained_output
        else:
            log_error(f"Travel chaining failed: {message}")
    return relocate_output(
        output_path=output_path,
        task_type=request.task_type,
        task_id=request.task_id,
        main_output_dir_base=request.main_output_dir_base,
    )


def resolve_orchestrator_final_output(
    *,
    existing_join_orchestrators,
    existing_stitch,
    existing_segments,
    segment_index_fn,
):
    if existing_join_orchestrators:
        return existing_join_orchestrators[0].get("output_location")
    if existing_stitch:
        return existing_stitch[0].get("output_location")
    if not existing_segments:
        return None
    return max(existing_segments, key=segment_index_fn).get("output_location")


__all__ = [
    "WorkerPostGenerationRequest",
    "apply_worker_post_generation_policy",
    "resolve_orchestrator_final_output",
]
