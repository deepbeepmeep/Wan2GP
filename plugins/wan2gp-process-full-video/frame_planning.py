from __future__ import annotations

import math
from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class ChunkPlan:
    control_start_frame: int
    requested_frames: int
    overlap_frames: int


@dataclass(frozen=True)
class FramePlanRules:
    frame_step: int
    minimum_requested_frames: int


class FramePlanningError(RuntimeError):
    pass


def require_model_def(model_type: str, get_model_def) -> dict:
    model_def = get_model_def(str(model_type))
    if not isinstance(model_def, dict):
        raise gr.Error(f"Unsupported model type: {model_type}")
    return model_def


def get_frame_plan_rules(model_type: str, get_model_def) -> FramePlanRules:
    model_def = require_model_def(model_type, get_model_def)
    return FramePlanRules(frame_step=int(model_def.get("frames_steps", 1)), minimum_requested_frames=int(model_def.get("frames_minimum", 1)))


def get_vae_temporal_latent_size(model_type: str, get_model_def) -> int:
    model_def = require_model_def(model_type, get_model_def)
    return int(model_def.get("latent_size", model_def.get("frames_steps", 1)))


def get_overlap_slider_max(model_type: str, get_model_def, *, exclusive_upper_bound: int = 100) -> int:
    step = get_vae_temporal_latent_size(model_type, get_model_def)
    last_allowed_value = int(exclusive_upper_bound) - 1
    return 1 + ((last_allowed_value - 1) // step) * step


def align_requested_frames(frame_count: int, *, frame_step: int, round_up: bool) -> int:
    if frame_count <= 1:
        return 1
    if round_up:
        return int(math.ceil((frame_count - 1) / float(frame_step)) * frame_step + 1)
    return int(math.floor((frame_count - 1) / float(frame_step)) * frame_step + 1)


def normalize_chunk_frames(chunk_seconds: float, fps_float: float, *, frame_step: int, minimum_requested_frames: int) -> int:
    if chunk_seconds < 0.1:
        raise FramePlanningError("Chunk Size must be at least 0.1 seconds.")
    if fps_float <= 0.0:
        raise FramePlanningError("Source FPS must be positive.")
    target_frames = int(round(chunk_seconds * fps_float))
    if target_frames < minimum_requested_frames:
        target_frames = minimum_requested_frames
    below = align_requested_frames(target_frames, frame_step=frame_step, round_up=False)
    if below < minimum_requested_frames:
        below = minimum_requested_frames
    above = align_requested_frames(target_frames, frame_step=frame_step, round_up=True)
    if above < minimum_requested_frames:
        above = minimum_requested_frames
    return below if abs(below - target_frames) <= abs(above - target_frames) else above


def normalize_overlap_frames(overlap_frames: float, *, frame_step: int) -> int:
    if overlap_frames < 1:
        raise FramePlanningError("Sliding Window Overlap must be at least 1 frame.")
    target_frames = int(round(float(overlap_frames)))
    below = align_requested_frames(target_frames, frame_step=frame_step, round_up=False)
    if below < 1:
        below = 1
    above = align_requested_frames(target_frames, frame_step=frame_step, round_up=True)
    if above < 1:
        above = 1
    return below if abs(below - target_frames) <= abs(above - target_frames) else above


def align_total_unique_frames(total_unique_frames: int, *, frame_step: int, minimum_requested_frames: int, initial_overlap_frames: int) -> int:
    if total_unique_frames <= 0:
        return 0
    if initial_overlap_frames < 0:
        raise FramePlanningError("Initial overlap cannot be negative.")
    if initial_overlap_frames > 0:
        minimum_unique_frames = minimum_requested_frames - initial_overlap_frames
        if minimum_unique_frames < 1:
            minimum_unique_frames = 1
        return 0 if total_unique_frames < minimum_unique_frames else total_unique_frames - (total_unique_frames % frame_step)
    return 0 if total_unique_frames < minimum_requested_frames else ((total_unique_frames - 1) // frame_step) * frame_step + 1


def count_planned_unique_frames(plans: list[ChunkPlan]) -> int:
    return sum(plan.requested_frames - plan.overlap_frames for plan in plans)


def describe_frame_range(start_frame: int, frame_count: int) -> str:
    if frame_count <= 0:
        return "0 frame(s)"
    return f"{frame_count} frame(s) [{start_frame}..{start_frame + frame_count - 1}]"


def build_chunk_plan(
    start_frame: int,
    end_frame_exclusive: int,
    total_source_frames: int,
    chunk_frames: int,
    *,
    frame_step: int,
    minimum_requested_frames: int,
    overlap_frames: int,
    initial_overlap_frames: int = 0,
) -> list[ChunkPlan]:
    if chunk_frames < minimum_requested_frames:
        raise FramePlanningError("Chunk size is below the model minimum frame count.")
    if overlap_frames < 0:
        raise FramePlanningError("Sliding Window Overlap cannot be negative.")
    if initial_overlap_frames < 0:
        raise FramePlanningError("Initial overlap cannot be negative.")
    if overlap_frames >= chunk_frames:
        raise FramePlanningError("Sliding Window Overlap must stay below the computed chunk size.")
    if initial_overlap_frames >= chunk_frames:
        raise FramePlanningError("Initial overlap must stay below the computed chunk size.")
    plans: list[ChunkPlan] = []
    cursor = start_frame
    total_unique_frames = align_total_unique_frames(
        end_frame_exclusive - start_frame,
        frame_step=frame_step,
        minimum_requested_frames=minimum_requested_frames,
        initial_overlap_frames=initial_overlap_frames,
    )
    if total_unique_frames <= 0:
        raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")
    written_unique_frames = 0
    while written_unique_frames < total_unique_frames:
        plan_overlap_frames = initial_overlap_frames if len(plans) == 0 else overlap_frames
        remaining_unique = total_unique_frames - written_unique_frames
        max_unique_frames = chunk_frames - plan_overlap_frames
        requested_frames = chunk_frames if remaining_unique > max_unique_frames else remaining_unique + plan_overlap_frames
        control_start_frame = cursor - plan_overlap_frames
        max_available_frames = total_source_frames - control_start_frame
        if max_available_frames < requested_frames:
            raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        if requested_frames < minimum_requested_frames:
            raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        plans.append(ChunkPlan(control_start_frame=control_start_frame, requested_frames=requested_frames, overlap_frames=plan_overlap_frames))
        unique_frames = requested_frames - plan_overlap_frames
        written_unique_frames += unique_frames
        cursor += unique_frames
    return plans


def count_completed_chunks(plans: list[ChunkPlan], completed_unique_frames: int) -> tuple[int, int]:
    if completed_unique_frames <= 0:
        return 0, 0
    completed_chunks = 0
    consumed_frames = 0
    for plan in plans:
        unique_frames = plan.requested_frames - plan.overlap_frames
        if consumed_frames + unique_frames <= completed_unique_frames:
            consumed_frames += unique_frames
            completed_chunks += 1
            continue
        break
    return completed_chunks, consumed_frames
