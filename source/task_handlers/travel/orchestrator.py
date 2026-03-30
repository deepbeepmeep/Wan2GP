"""Travel orchestrator task handler - creates and manages segment tasks."""

from dataclasses import replace
from pathlib import Path
import uuid
from datetime import datetime
from typing import Any, Optional

# Import structured logging
from ...core.log import travel_logger, safe_json_repr, safe_dict_repr

from ... import db_operations as db_ops
from ...core.db import config as db_config
from ...utils import (
    parse_resolution,
    snap_resolution_to_model_grid,
    upload_and_get_final_output_location,
    get_video_frame_count_and_fps)
from ...utils.resolution_utils import get_model_grid_size
from ...core.params.generation_policy import GenerationPolicy
from ...core.params.structure_guidance import StructureGuidanceConfig
from ...core.params.travel_guidance import AnchorEntry, TravelGuidanceConfig
from ...core.params.task_result import TaskResult
from ...runtime.wgp_bridge import get_model_fps, get_model_min_frames_and_step

from .svi_config import SVI_DEFAULT_PARAMS, SVI_STITCH_OVERLAP
from .debug_utils import log_ram_usage

# Default seed used when no seed_base is provided in the orchestrator payload
DEFAULT_SEED_BASE = 12345
IC_LORA_UNION_CONTROL_FILENAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
HYBRID_SEGMENT_ANCHOR_SOFT_LIMIT = 4
HYBRID_SEGMENT_ANCHOR_HARD_LIMIT = 8
get_orchestrator_child_tasks = db_ops.get_orchestrator_child_tasks
cleanup_duplicate_child_tasks = db_ops.cleanup_duplicate_child_tasks


def _get_model_fps(model_name: str | None) -> int:
    """Get the native FPS for a model (e.g. 16 for Wan, 24 for LTX-2).

    Queries WGP's model definition when available, falls back to 24.
    """
    if not model_name:
        return 24
    try:
        return get_model_fps(model_name)
    except (ImportError, TypeError, ValueError, KeyError):
        # Heuristic fallback
        if "ltx2" in model_name.lower():
            return 24
        return 24


def _get_frame_step(model_name: str | None) -> int:
    """Get the frame quantization step for a model (e.g. 4 for Wan, 8 for LTX-2).

    Queries WGP's model definition when available, falls back to 4 (Wan default).
    """
    if not model_name:
        return 4
    try:
        _min_frames, frames_step, _latent_size = get_model_min_frames_and_step(model_name)
        return frames_step
    except (ImportError, TypeError, ValueError, KeyError):
        return 4


def _quantize_frames(frames: int, step: int) -> int:
    """Quantize a frame count down to the nearest valid step*N+1 value.

    E.g. for step=4: valid values are 1, 5, 9, 13, ...
    For step=8: valid values are 1, 9, 17, 25, ...
    """
    return ((frames - 1) // step) * step + 1


def _quantize_frames_up(frames: int, step: int) -> int:
    """Quantize a frame count up to the nearest valid step*N+1 value."""
    if frames <= 1:
        return 1
    return ((frames + step - 2) // step) * step + 1


def _calculate_segment_stitched_offsets(
    segment_frames_expanded: list[int],
    frame_overlap_expanded: list[int],
) -> tuple[list[int], int]:
    """Return stitched segment start offsets and total stitched frame count."""
    total_stitched_frames = 0
    segment_stitched_offsets: list[int] = []

    for idx, segment_total_frames in enumerate(segment_frames_expanded):
        if idx == 0:
            segment_stitched_offsets.append(0)
            total_stitched_frames = segment_total_frames
        else:
            overlap = frame_overlap_expanded[idx - 1] if idx - 1 < len(frame_overlap_expanded) else 0
            segment_start = total_stitched_frames - overlap
            segment_stitched_offsets.append(segment_start)
            total_stitched_frames = segment_start + segment_total_frames

    return segment_stitched_offsets, total_stitched_frames


def _segment_has_travel_guidance_overlap(
    *,
    segment_index: int,
    segment_frames_expanded: list[int],
    segment_stitched_offsets: list[int],
    total_stitched_frames: int,
    travel_guidance_config: TravelGuidanceConfig,
) -> bool:
    """Check travel-guidance overlap using the stitched timeline semantics."""
    if travel_guidance_config.kind == "none":
        return False
    segment_start = segment_stitched_offsets[segment_index] if segment_index < len(segment_stitched_offsets) else 0
    segment_end = segment_start + (
        segment_frames_expanded[segment_index]
        if segment_index < len(segment_frames_expanded)
        else 0
    )

    if travel_guidance_config.is_ltx_hybrid and travel_guidance_config.anchors:
        for anchor in travel_guidance_config.anchors:
            if segment_start <= anchor.frame_position < segment_end:
                return True

    if not travel_guidance_config.videos:
        return bool(travel_guidance_config.guidance_video_url)

    for video in travel_guidance_config.videos:
        video_start = video.start_frame
        video_end = total_stitched_frames if video.end_frame is None else video.end_frame
        if video_start < segment_end and video_end > segment_start:
            return True

    return False


def _segment_has_travel_guidance_control_overlap(
    *,
    segment_index: int,
    segment_frames_expanded: list[int],
    segment_stitched_offsets: list[int],
    total_stitched_frames: int,
    travel_guidance_config: TravelGuidanceConfig,
) -> bool:
    if travel_guidance_config.kind == "none":
        return False
    if not travel_guidance_config.videos:
        return bool(travel_guidance_config.guidance_video_url)

    segment_start = segment_stitched_offsets[segment_index] if segment_index < len(segment_stitched_offsets) else 0
    segment_end = segment_start + (
        segment_frames_expanded[segment_index]
        if segment_index < len(segment_frames_expanded)
        else 0
    )
    for video in travel_guidance_config.videos:
        video_start = video.start_frame
        video_end = total_stitched_frames if video.end_frame is None else video.end_frame
        if video_start < segment_end and video_end > segment_start:
            return True
    return False


def _build_segment_hybrid_guidance_config(
    *,
    config: TravelGuidanceConfig,
    segment_index: int,
    segment_frames_expanded: list[int],
    segment_stitched_offsets: list[int],
    total_stitched_frames: int,
) -> TravelGuidanceConfig:
    segment_start = segment_stitched_offsets[segment_index] if segment_index < len(segment_stitched_offsets) else 0
    segment_frame_count = (
        segment_frames_expanded[segment_index]
        if segment_index < len(segment_frames_expanded)
        else 0
    )
    remapped_anchors: list[AnchorEntry] = []
    for anchor in config.anchors:
        local_pos = anchor.frame_position - segment_start
        if 0 <= local_pos < segment_frame_count:
            remapped_anchors.append(
                AnchorEntry(
                    image_url=anchor.image_url,
                    frame_position=local_pos,
                    strength=anchor.strength,
                )
            )

    local_positions = [anchor.frame_position for anchor in remapped_anchors]
    if len(local_positions) != len(set(local_positions)):
        raise ValueError(
            "ltx_hybrid segment contains duplicate local frame positions after remap"
        )

    if len(remapped_anchors) > HYBRID_SEGMENT_ANCHOR_HARD_LIMIT:
        raise ValueError(
            "ltx_hybrid segment exceeds anchor cap: "
            f"{len(remapped_anchors)} > {HYBRID_SEGMENT_ANCHOR_HARD_LIMIT}"
        )
    if len(remapped_anchors) > HYBRID_SEGMENT_ANCHOR_SOFT_LIMIT:
        travel_logger.warning(
            "[TRAVEL_GUIDANCE] Segment %s has %s hybrid anchors (soft limit=%s)",
            segment_index,
            len(remapped_anchors),
            HYBRID_SEGMENT_ANCHOR_SOFT_LIMIT,
        )

    has_control = _segment_has_travel_guidance_control_overlap(
        segment_index=segment_index,
        segment_frames_expanded=segment_frames_expanded,
        segment_stitched_offsets=segment_stitched_offsets,
        total_stitched_frames=total_stitched_frames,
        travel_guidance_config=config,
    )
    if not remapped_anchors and not has_control:
        return TravelGuidanceConfig(kind="none")

    audio_config = config.audio
    if (
        audio_config is not None
        and audio_config.source == "control_track"
        and not has_control
    ):
        audio_config = None

    return replace(
        config,
        videos=list(config.videos) if has_control else [],
        anchors=remapped_anchors,
        audio=audio_config,
        _frame_offset=segment_start,
        _guidance_video_url=config.guidance_video_url if has_control else None,
    )


def _build_segment_travel_guidance_payload(
    config: TravelGuidanceConfig,
    *,
    frame_offset: int,
    has_guidance: bool,
) -> dict[str, Any]:
    """Return the explicit segment-level travel guidance payload."""
    if not has_guidance:
        return {"kind": "none"}
    return config.to_segment_payload(frame_offset=frame_offset)


def _ensure_ic_lora_in_lora_dir() -> None:
    """Ensure the IC-LoRA union control file is in WGP's lora directory.

    WGP downloads models to ckpts/ but validates LoRAs in loras/ltx2/.
    If the file exists in ckpts/ but not in the lora dir, create a symlink.
    """
    from source.core.runtime_paths import get_repo_root

    repo_root = get_repo_root()
    ckpts_path = repo_root / "Wan2GP" / "ckpts" / IC_LORA_UNION_CONTROL_FILENAME
    lora_dir = repo_root / "Wan2GP" / "loras" / "ltx2"
    lora_path = lora_dir / IC_LORA_UNION_CONTROL_FILENAME

    if lora_path.exists():
        return

    if not ckpts_path.exists():
        travel_logger.warning(
            f"[IC_LORA] IC-LoRA not found in ckpts/ ({ckpts_path}) — WGP may need to download it"
        )
        return

    lora_dir.mkdir(parents=True, exist_ok=True)
    try:
        lora_path.symlink_to(ckpts_path)
        travel_logger.info(f"[IC_LORA] Symlinked IC-LoRA: {ckpts_path} → {lora_path}")
    except OSError as e:
        travel_logger.warning(f"[IC_LORA] Failed to symlink IC-LoRA: {e}")


def _auto_inject_travel_guidance_lora(
    segment_loras: Optional[list[dict[str, Any]]],
    travel_guidance_config: Optional[TravelGuidanceConfig],
) -> Optional[list[dict[str, Any]]]:
    """Add the union IC-LoRA for LTX control modes that require it."""
    if not travel_guidance_config or not travel_guidance_config.needs_ic_lora():
        return segment_loras

    # Ensure the IC-LoRA file is where WGP expects it
    _ensure_ic_lora_in_lora_dir()

    lora_strength = (
        travel_guidance_config.control_strength
        if travel_guidance_config.is_ltx_hybrid
        else travel_guidance_config.strength
    )

    merged_loras = [dict(entry) for entry in (segment_loras or [])]
    for existing in merged_loras:
        if existing.get("path") == IC_LORA_UNION_CONTROL_FILENAME:
            existing["strength"] = lora_strength
            existing.setdefault("name", "ic-lora-union-control (auto-injected)")
            return merged_loras

    merged_loras.append(
        {
            "path": IC_LORA_UNION_CONTROL_FILENAME,
            "strength": lora_strength,
            "name": "ic-lora-union-control (auto-injected)",
        }
    )
    return merged_loras


# Offset added to the base seed to derive a deterministic but distinct seed for upscaling
UPSCALE_SEED_OFFSET = 5000

def handle_travel_orchestrator_task(task_params_from_db: dict, main_output_dir_base: Path, orchestrator_task_id_str: str, orchestrator_project_id: str | None):
    travel_logger.essential("Starting travel orchestrator task", task_id=orchestrator_task_id_str)
    log_ram_usage("Orchestrator start", task_id=orchestrator_task_id_str)
    travel_logger.debug(f"Project ID: {orchestrator_project_id}", task_id=orchestrator_task_id_str)
    # Safe logging: Use safe_json_repr to prevent hangs on large nested structures
    travel_logger.debug(f"Task params: {safe_json_repr(task_params_from_db)}", task_id=orchestrator_task_id_str)

    try:
        if 'orchestrator_details' not in task_params_from_db:
            travel_logger.error("'orchestrator_details' not found in task_params_from_db", task_id=orchestrator_task_id_str)
            return TaskResult.failed("orchestrator_details missing")
        
        orchestrator_payload = task_params_from_db['orchestrator_details']
        # Safe logging: Use safe_dict_repr for better performance than JSON serialization
        travel_logger.debug(f"Orchestrator payload: {safe_dict_repr(orchestrator_payload)}", task_id=orchestrator_task_id_str)

        # Validate required keys are present before proceeding
        from source.core.params.contracts import validate_orchestrator_details
        validate_orchestrator_details(orchestrator_payload, context="travel_orchestrator", task_id=orchestrator_task_id_str)

        # Determine frame quantization step for this model (e.g. 4 for Wan, 8 for LTX-2)
        frame_step = _get_frame_step(orchestrator_payload.get("model_name"))
        travel_logger.debug(f"Frame quantization step: {frame_step} (model: {orchestrator_payload.get('model_name')})", task_id=orchestrator_task_id_str)

        # Set fps_helpers from model native FPS if not explicitly provided
        if "fps_helpers" not in orchestrator_payload:
            model_fps = _get_model_fps(orchestrator_payload.get("model_name"))
            orchestrator_payload["fps_helpers"] = model_fps
            travel_logger.debug(f"Set fps_helpers={model_fps} from model native FPS", task_id=orchestrator_task_id_str)

        # Normalize chain_segments: true = chain segments together (default), false = keep separate
        chain_segments_raw = orchestrator_payload.get("chain_segments", True)
        orchestrator_payload["chain_segments"] = bool(chain_segments_raw)

        # Parse phase_config if present and add parsed values to orchestrator_payload
        if "phase_config" in orchestrator_payload:
            travel_logger.info(f"phase_config detected in orchestrator - parsing comprehensive phase configuration", task_id=orchestrator_task_id_str)

            try:
                from source.core.params.phase_config_parser import parse_phase_config

                # Get total steps from phase_config
                phase_config = orchestrator_payload["phase_config"]
                steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
                total_steps = sum(steps_per_phase)

                # Parse phase_config to get all parameters
                parsed = parse_phase_config(
                    phase_config=phase_config,
                    num_inference_steps=total_steps,
                    task_id=orchestrator_task_id_str,
                    model_name=orchestrator_payload.get("model_name")
                )

                # Add parsed values to orchestrator_payload so segments can use them
                # NOTE: We use lora_names + lora_multipliers directly, NOT additional_loras
                for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                           "guidance_scale", "guidance2_scale", "guidance3_scale",
                           "flow_shift", "sample_solver", "model_switch_phase",
                           "lora_names", "lora_multipliers"]:
                    if key in parsed and parsed[key] is not None:
                        orchestrator_payload[key] = parsed[key]
                        travel_logger.debug(f"[ORCHESTRATOR_PHASE_CONFIG] Added {key} to orchestrator_payload: {parsed[key]}")

                # Also update num_inference_steps
                orchestrator_payload["num_inference_steps"] = total_steps

                travel_logger.info(
                    f"phase_config parsed: {parsed['guidance_phases']} phases, "
                    f"steps={total_steps}, "
                    f"{len(parsed.get('lora_names', []))} LoRAs, "
                    f"lora_multipliers={parsed['lora_multipliers']}",
                    task_id=orchestrator_task_id_str
                )

            except (ValueError, KeyError, TypeError) as e:
                travel_logger.error(f"Failed to parse phase_config: {e}", task_id=orchestrator_task_id_str, exc_info=True)
                return TaskResult.failed(f"Failed to parse phase_config: {e}")

        # IDEMPOTENCY CHECK: Look for existing child tasks before creating new ones
        travel_logger.debug(f"[IDEMPOTENCY] Checking for existing child tasks for orchestrator {orchestrator_task_id_str}")
        existing_child_tasks = get_orchestrator_child_tasks(orchestrator_task_id_str)
        existing_segments = existing_child_tasks['segments']
        existing_stitch = existing_child_tasks['stitch']
        
        expected_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)
        # Some runs intentionally do NOT create a stitch task (e.g. chain_segments=False, or i2v non-SVI).
        # In those cases, the orchestrator should be considered complete once all segments complete.
        travel_mode = orchestrator_payload.get("model_type", "vace")
        _continuation_cfg = orchestrator_payload.get("continuation_config")
        use_svi = (
            bool(orchestrator_payload.get("use_svi", False))
            or (
                isinstance(_continuation_cfg, dict)
                and _continuation_cfg.get("strategy") == "svi_latent_chaining"
            )
        )
        orchestrator_payload["use_svi"] = use_svi

        # SVI is not compatible with LTX-2: it uses Wan-specific LoRAs and
        # destructively patches the model's native sliding window settings.
        _model_name_lower = (orchestrator_payload.get("model_name") or "").lower()
        if use_svi and "ltx2" in _model_name_lower:
            travel_logger.warning(
                "SVI mode requested but not supported for LTX-2 models — disabling. "
                "LTX-2 uses native sliding window instead.",
                task_id=orchestrator_task_id_str,
            )
            use_svi = False
            orchestrator_payload["use_svi"] = False
            # SVI unsupported on LTX, but LTX has its own continuation strategy.
            # Rewrite to prefix_video_source so _apply_continuation_config() still runs.
            if isinstance(_continuation_cfg, dict):
                _continuation_cfg["strategy"] = "prefix_video_source"
                _continuation_cfg["overlap_frames"] = 25  # LTX 8n+1: 3 latent frames + initial

        chain_segments = bool(orchestrator_payload.get("chain_segments", True))
        generation_policy = GenerationPolicy.from_payload(orchestrator_payload)
        should_create_stitch = bool(
            use_svi
            or generation_policy.continuation.enabled
        )
        required_stitch_count = 1 if should_create_stitch else 0
        
        # Also check for join_clips_orchestrator (created when stitch_config is present)
        existing_join_orchestrators = existing_child_tasks.get('join_clips_orchestrator', [])
        stitch_config_present = bool(orchestrator_payload.get("stitch_config"))
        required_join_orchestrator_count = 1 if stitch_config_present else 0

        if existing_segments or existing_stitch or existing_join_orchestrators:
            travel_logger.debug(f"[IDEMPOTENCY] Found existing child tasks: {len(existing_segments)} segments, {len(existing_stitch)} stitch tasks, {len(existing_join_orchestrators)} join orchestrators")

            # Check if we have the expected number of tasks already
            has_required_join_orchestrator = len(existing_join_orchestrators) >= required_join_orchestrator_count
            if len(existing_segments) >= expected_segments and len(existing_stitch) >= required_stitch_count and has_required_join_orchestrator:
                # Clean up any duplicates but don't create new tasks
                cleanup_summary = cleanup_duplicate_child_tasks(orchestrator_task_id_str, expected_segments)

                if cleanup_summary['duplicate_segments_removed'] > 0 or cleanup_summary['duplicate_stitch_removed'] > 0:
                    travel_logger.info(f"Cleaned up duplicates: {cleanup_summary['duplicate_segments_removed']} segments, {cleanup_summary['duplicate_stitch_removed']} stitch tasks", task_id=orchestrator_task_id_str)

                # CHECK: Are all child tasks actually complete?
                # If they are, we should mark orchestrator as complete instead of leaving it IN_PROGRESS
                # Also check for terminal failure states (failed/cancelled) that should mark orchestrator as failed

                def is_complete(task):
                    # DB stores statuses as "Complete" (capitalized). Compare case-insensitively.
                    return (task.get('status', '') or '').lower() == 'complete'

                def is_terminal_failure(task):
                    """Check if task is in a terminal failure state (failed, cancelled, etc.)"""
                    status = task.get('status', '').lower()
                    return status in ('failed', 'cancelled', 'canceled', 'error')

                all_segments_complete = all(is_complete(seg) for seg in existing_segments) if existing_segments else False
                all_stitch_complete = True if required_stitch_count == 0 else (all(is_complete(st) for st in existing_stitch) if existing_stitch else False)
                all_join_orchestrators_complete = True if required_join_orchestrator_count == 0 else (all(is_complete(jo) for jo in existing_join_orchestrators) if existing_join_orchestrators else False)

                any_segment_failed = any(is_terminal_failure(seg) for seg in existing_segments) if existing_segments else False
                any_stitch_failed = False if required_stitch_count == 0 else (any(is_terminal_failure(st) for st in existing_stitch) if existing_stitch else False)
                any_join_orchestrator_failed = False if required_join_orchestrator_count == 0 else (any(is_terminal_failure(jo) for jo in existing_join_orchestrators) if existing_join_orchestrators else False)

                # Also ensure we have the minimum required tasks
                has_required_segments = len(existing_segments) >= expected_segments
                has_required_stitch = True if required_stitch_count == 0 else (len(existing_stitch) >= 1)

                # If any child task failed/cancelled, mark orchestrator as failed
                if (any_segment_failed or any_stitch_failed or any_join_orchestrator_failed) and has_required_segments and has_required_stitch and has_required_join_orchestrator:
                    failed_segments = [seg for seg in existing_segments if is_terminal_failure(seg)]
                    failed_stitch = [st for st in existing_stitch if is_terminal_failure(st)]
                    failed_join_orchestrators = [jo for jo in existing_join_orchestrators if is_terminal_failure(jo)]

                    error_details = []
                    if failed_segments:
                        error_details.append(f"{len(failed_segments)} segment(s) failed/cancelled")
                    if failed_stitch:
                        error_details.append(f"{len(failed_stitch)} stitch task(s) failed/cancelled")
                    if failed_join_orchestrators:
                        error_details.append(f"{len(failed_join_orchestrators)} join orchestrator(s) failed/cancelled")

                    travel_logger.debug(f"[IDEMPOTENT_FAILED] Child tasks failed: {', '.join(error_details)}")
                    travel_logger.error(f"Child tasks in terminal failure state: {', '.join(error_details)}", task_id=orchestrator_task_id_str)

                    # Return failure so orchestrator is marked as failed
                    generation_success = False
                    output_message_for_orchestrator_db = f"[ORCHESTRATOR_FAILED] Child tasks failed: {', '.join(error_details)}"
                    return generation_success, output_message_for_orchestrator_db

                if all_segments_complete and all_stitch_complete and all_join_orchestrators_complete and has_required_segments and has_required_stitch and has_required_join_orchestrator:
                    # All children are done! Return with special "COMPLETE" marker
                    travel_logger.debug(f"[IDEMPOTENT_COMPLETE] All {len(existing_segments)} segments, {len(existing_stitch)} stitch, {len(existing_join_orchestrators)} join orchestrators complete")
                    travel_logger.info(f"All child tasks complete, orchestrator should be marked as complete", task_id=orchestrator_task_id_str)

                    # Get the final output:
                    # - If join_clips_orchestrator exists (stitch_config mode), use its output
                    # - Else if stitch exists, use its output
                    # - Otherwise (independent segments), use the last segment's output
                    final_output = None
                    if existing_join_orchestrators:
                        final_output = existing_join_orchestrators[0].get('output_location')
                    if not final_output and existing_stitch:
                        final_output = existing_stitch[0].get('output_location')
                    if not final_output and existing_segments:
                        def _seg_idx(seg):
                            try:
                                return int(seg.get('params', {}).get('segment_index', -1))
                            except (ValueError, TypeError):
                                return -1
                        last_seg = sorted(existing_segments, key=_seg_idx)[-1]
                        final_output = last_seg.get('output_location')
                    if not final_output:
                        final_output = 'Completed via idempotency'

                    # Return with special marker so worker knows to mark as COMPLETE instead of IN_PROGRESS
                    # We use a tuple with the marker to signal completion
                    generation_success = True
                    output_message_for_orchestrator_db = f"[ORCHESTRATOR_COMPLETE]{final_output}"  # Special prefix
                    travel_logger.info(f"All child tasks complete. Returning final output: {final_output}", task_id=orchestrator_task_id_str)
                    return generation_success, output_message_for_orchestrator_db
                else:
                    # Some children still in progress - report status and let worker keep waiting
                    segments_complete_count = sum(1 for seg in existing_segments if is_complete(seg))
                    stitch_complete_count = sum(1 for st in existing_stitch if is_complete(st))

                    generation_success = True
                    if required_stitch_count == 0:
                        output_message_for_orchestrator_db = f"[IDEMPOTENT] Child tasks already exist but not all complete: {segments_complete_count}/{len(existing_segments)} segments complete. Cleaned up {cleanup_summary['duplicate_segments_removed']} duplicate segments."
                    else:
                        output_message_for_orchestrator_db = f"[IDEMPOTENT] Child tasks already exist but not all complete: {segments_complete_count}/{len(existing_segments)} segments complete, {stitch_complete_count}/{len(existing_stitch)} stitch complete. Cleaned up {cleanup_summary['duplicate_segments_removed']} duplicate segments and {cleanup_summary['duplicate_stitch_removed']} duplicate stitch tasks."
                    travel_logger.info(output_message_for_orchestrator_db, task_id=orchestrator_task_id_str)
                    return generation_success, output_message_for_orchestrator_db
            else:
                # Partial completion - log and continue with missing tasks
                travel_logger.debug(f"[IDEMPOTENCY] Partial child tasks found: {len(existing_segments)}/{expected_segments} segments, {len(existing_stitch)}/{required_stitch_count} stitch. Will continue with orchestration.")
                travel_logger.warning(f"Partial child tasks found, continuing orchestration to create missing tasks", task_id=orchestrator_task_id_str)
        else:
            travel_logger.debug(f"[IDEMPOTENCY] No existing child tasks found. Proceeding with normal orchestration.")

        run_id = orchestrator_payload.get("run_id", orchestrator_task_id_str)
        base_dir_for_this_run_str = orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))

        # Convert to Path and resolve relative paths against project root (CWD)
        # This ensures './outputs/foo' resolves to '/project/outputs/foo', not '/project/outputs/outputs/foo'
        base_dir_path = Path(base_dir_for_this_run_str)
        if not base_dir_path.is_absolute():
            # Relative path - resolve against current working directory (project root)
            current_run_output_dir = (Path.cwd() / base_dir_path).resolve()
        else:
            # Already absolute - use as is
            current_run_output_dir = base_dir_path.resolve()

        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        travel_logger.debug(f"Orchestrator {orchestrator_task_id_str}: Base output directory for this run: {current_run_output_dir}")

        num_segments = orchestrator_payload.get("num_new_segments_to_generate", 0)
        if num_segments <= 0:
            msg = "no-op payload: no new segments to generate"
            travel_logger.warning(msg, task_id=orchestrator_task_id_str)
            return TaskResult.orchestrator_complete(msg)

        # Track actual DB row IDs by segment index to avoid mixing logical IDs
        actual_segment_db_id_by_index: dict[int, str] = {}

        # Track which segments already exist to avoid re-creating them
        existing_segment_indices = set()
        existing_segment_task_ids = {}  # index -> task_id mapping
        
        for segment in existing_segments:
            segment_idx = segment['params'].get('segment_index', -1)
            if segment_idx >= 0:
                existing_segment_indices.add(segment_idx)
                existing_segment_task_ids[segment_idx] = segment['id']
                # CRITICAL FIX: Pre-populate actual_segment_db_id_by_index with existing segments
                # so that new segments can correctly depend on existing ones
                actual_segment_db_id_by_index[segment_idx] = segment['id']
                
        # Check if stitch task already exists
        stitch_already_exists = len(existing_stitch) > 0
        existing_stitch_task_id = existing_stitch[0]['id'] if stitch_already_exists else None
        
        travel_logger.debug(f"[IDEMPOTENCY] Existing segment indices: {sorted(existing_segment_indices)}")
        travel_logger.debug(f"[IDEMPOTENCY] Stitch task exists: {stitch_already_exists} (ID: {existing_stitch_task_id})")

        # Image download directory is not needed for Supabase - images are already uploaded
        segment_image_download_dir_str : str | None = None

        # Expanded arrays from orchestrator payload
        expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
        expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
        expanded_segment_frames = orchestrator_payload["segment_frames_expanded"]
        expanded_frame_overlap = orchestrator_payload["frame_overlap_expanded"]
        vace_refs_instructions_all = orchestrator_payload.get("vace_image_refs_to_prepare_by_worker", [])

        # Per-segment parameter overrides
        phase_configs_expanded = orchestrator_payload.get("phase_configs_expanded", [])
        loras_per_segment_expanded = orchestrator_payload.get("loras_per_segment_expanded", [])

        # Log per-segment override summary
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] ========== Per-Segment Parameter Overrides ==========")
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] phase_configs_expanded: {len(phase_configs_expanded)} entries received")
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] loras_per_segment_expanded: {len(loras_per_segment_expanded)} entries received")

        # Count non-null overrides
        phase_config_overrides = sum(1 for pc in phase_configs_expanded if pc is not None)
        lora_overrides = sum(1 for l in loras_per_segment_expanded if l is not None)
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] Segments with phase_config override: {phase_config_overrides}/{num_segments}")
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] Segments with LoRA override: {lora_overrides}/{num_segments}")

        # Log per-segment detail
        for idx in range(num_segments):
            has_pc = idx < len(phase_configs_expanded) and phase_configs_expanded[idx] is not None
            has_lora = idx < len(loras_per_segment_expanded) and loras_per_segment_expanded[idx] is not None

            pc_info = "CUSTOM" if has_pc else "default"
            if has_pc:
                pc = phase_configs_expanded[idx]
                preset_name = pc.get("preset_name", pc.get("name", "unknown"))
                pc_info = f"CUSTOM ({preset_name})"

            lora_info = "default"
            if has_lora:
                lora_count = len(loras_per_segment_expanded[idx])
                lora_names = [l.get("name", l.get("path", "?")[:20]) for l in loras_per_segment_expanded[idx][:3]]
                lora_info = f"CUSTOM ({lora_count} LoRAs: {lora_names})"

            travel_logger.debug(f"[PER_SEGMENT_OVERRIDES]   Segment {idx}: phase_config={pc_info}, loras={lora_info}")
        travel_logger.debug(f"[PER_SEGMENT_OVERRIDES] =====================================================")

        # Normalize single int frame_overlap to array
        if isinstance(expanded_frame_overlap, int):
            single_overlap_value = expanded_frame_overlap
            # For N segments, we need N-1 overlap values (one for each transition)
            expanded_frame_overlap = [single_overlap_value] * max(0, num_segments - 1)
            orchestrator_payload["frame_overlap_expanded"] = expanded_frame_overlap
            travel_logger.debug(f"[NORMALIZE] Expanded single frame_overlap int {single_overlap_value} to array of {len(expanded_frame_overlap)} elements: {expanded_frame_overlap}")

        # [PAYLOAD_ORDER_DEBUG] Log the alignment of images and prompts from payload
        input_images_from_payload = orchestrator_payload.get("input_image_paths_resolved", [])
        travel_logger.debug(f"[PAYLOAD_ORDER_DEBUG] Orchestrator received: {len(input_images_from_payload)} images, {len(expanded_base_prompts)} prompts, {num_segments} segments")
        travel_logger.debug(f"[PAYLOAD_ORDER_DEBUG] Image → Prompt alignment check:")
        for i in range(max(len(input_images_from_payload), len(expanded_base_prompts))):
            img_name = Path(input_images_from_payload[i]).name if i < len(input_images_from_payload) else "NO_IMAGE"
            prompt_preview = expanded_base_prompts[i][:60] if i < len(expanded_base_prompts) and expanded_base_prompts[i] else "NO_PROMPT"
            transition_note = f"(seg {i}: img[{i}]→img[{i+1}])" if i < num_segments else "(no segment)"
            travel_logger.debug(f"[PAYLOAD_ORDER_DEBUG]   [{i}] {img_name} | '{prompt_preview}...' {transition_note}")

        # Preserve a copy of the original overlap list in case we need it later
        _orig_frame_overlap = list(expanded_frame_overlap)  # shallow copy

        # --- IDENTICAL PARAMETER DETECTION AND FRAME CONSOLIDATION ---
        def detect_identical_parameters(orchestrator_payload, num_segments):
            """
            Detect if all segments will have identical generation parameters.
            Returns analysis that enables both model caching and frame optimization.
            """
            # Extract parameter arrays
            expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
            expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
            lora_names = orchestrator_payload.get("lora_names", [])

            # Check parameter identity
            prompts_identical = len(set(expanded_base_prompts)) == 1
            negative_prompts_identical = len(set(expanded_negative_prompts)) == 1

            is_identical = prompts_identical and negative_prompts_identical

            if is_identical:
                travel_logger.debug(f"[IDENTICAL_DETECTION] All {num_segments} segments identical - enabling optimizations")
                travel_logger.debug(f"  - Unique prompt: '{expanded_base_prompts[0][:50]}...'")
                travel_logger.debug(f"  - LoRA count: {len(lora_names)}")

            return {
                "is_identical": is_identical,
                "can_optimize_frames": is_identical,  # Key for frame allocation optimization
                "can_reuse_model": is_identical,      # Key for model caching
                "unique_prompt": expanded_base_prompts[0] if prompts_identical else None
            }

        def validate_consolidation_safety(orchestrator_payload):
            """
            Verify that frame consolidation is safe by checking parameter identity.
            """
            # Get parameter arrays
            prompts = orchestrator_payload["base_prompts_expanded"]
            neg_prompts = orchestrator_payload["negative_prompts_expanded"]
            _lora_names = orchestrator_payload.get("lora_names", [])

            # Critical safety checks
            all_prompts_identical = len(set(prompts)) == 1
            all_neg_prompts_identical = len(set(neg_prompts)) == 1

            is_safe = all_prompts_identical and all_neg_prompts_identical

            if is_safe:
                travel_logger.debug(f"[CONSOLIDATION_SAFETY] ✅ Safe to consolidate - all parameters identical")
            else:
                travel_logger.debug(f"[CONSOLIDATION_SAFETY] ❌ NOT safe to consolidate:")
                if not all_prompts_identical:
                    travel_logger.debug(f"  - Prompts differ: {len(set(prompts))} unique prompts")
                if not all_neg_prompts_identical:
                    travel_logger.debug(f"  - Negative prompts differ: {len(set(neg_prompts))} unique")

            return {
                "is_safe": is_safe,
                "prompts_identical": all_prompts_identical,
                "negative_prompts_identical": all_neg_prompts_identical,
                "can_consolidate": is_safe
            }

        def optimize_frame_allocation_for_identical_params(orchestrator_payload, max_frames_per_segment=65):
            """
            When all parameters are identical, consolidate keyframes into fewer segments.

            Args:
                orchestrator_payload: Original orchestrator data
                max_frames_per_segment: Maximum frames per segment (model technical limit)

            Returns:
                Updated orchestrator_payload with optimized frame allocation
            """
            original_segment_frames = orchestrator_payload["segment_frames_expanded"]
            original_frame_overlaps = orchestrator_payload["frame_overlap_expanded"]
            original_base_prompts = orchestrator_payload["base_prompts_expanded"]

            travel_logger.debug(f"[FRAME_CONSOLIDATION] Original allocation: {len(original_segment_frames)} segments")
            travel_logger.debug(f"  - Segment frames: {original_segment_frames}")
            travel_logger.debug(f"  - Frame overlaps: {original_frame_overlaps}")

            # Calculate keyframe positions based on raw segment durations (no overlaps for consolidated videos)
            keyframe_positions = [0]  # Start with frame 0
            cumulative_pos = 0

            for segment_frames in original_segment_frames:
                cumulative_pos += segment_frames
                keyframe_positions.append(cumulative_pos)

            travel_logger.debug(f"[FRAME_CONSOLIDATION] Keyframe positions: {keyframe_positions}")

            # Simple consolidation: group keyframes into videos respecting frame limit
            optimized_segments = []
            optimized_overlaps = []
            optimized_prompts = []

            video_start = 0
            video_keyframes = [0]  # Always include first keyframe

            for i in range(1, len(keyframe_positions)):
                kf_pos = keyframe_positions[i]
                video_length_if_included = kf_pos - video_start + 1

                if video_length_if_included <= max_frames_per_segment:
                    # Keyframe fits in current video
                    video_keyframes.append(kf_pos)
                    travel_logger.debug(f"[CONSOLIDATION_LOGIC] Keyframe {kf_pos} fits in current video (length would be {video_length_if_included})")
                else:
                    # Current video is full, finalize it and start new one
                    final_frame = video_keyframes[-1]
                    raw_length = final_frame - video_start + 1
                    quantized_length = _quantize_frames(raw_length, frame_step)
                    optimized_segments.append(quantized_length)
                    optimized_prompts.append(original_base_prompts[0])

                    travel_logger.debug(f"[CONSOLIDATION_LOGIC] Video complete: frames {video_start}-{final_frame} (raw: {raw_length}, quantized: {quantized_length})")
                    travel_logger.debug(f"[CONSOLIDATION_LOGIC] Video keyframes: {[kf - video_start for kf in video_keyframes]}")

                    # Add overlap for the next video if there are more keyframes to process
                    # When we finalize a video because the next keyframe doesn't fit,
                    # we need overlap for the next video
                    if i < len(keyframe_positions):  # Still have more keyframes = need next video
                        # Use original overlap value instead of calculating new one
                        if isinstance(original_frame_overlaps, list) and original_frame_overlaps:
                            overlap = original_frame_overlaps[0]  # Use first value from array
                        elif isinstance(original_frame_overlaps, int):
                            overlap = original_frame_overlaps  # Use int value directly
                        else:
                            overlap = 4  # Default fallback if no overlap specified

                        optimized_overlaps.append(overlap)
                        travel_logger.debug(f"[CONSOLIDATION_LOGIC] Added overlap {overlap} frames for next video (from original settings)")

                    # Start new video
                    video_start = video_keyframes[-1]  # Start from last keyframe of previous video
                    video_keyframes = [video_start, kf_pos]
                    travel_logger.debug(f"[CONSOLIDATION_LOGIC] Starting new video at frame {video_start}")

            # Finalize the last video
            final_frame = video_keyframes[-1]
            raw_length = final_frame - video_start + 1
            quantized_length = _quantize_frames(raw_length, frame_step)
            optimized_segments.append(quantized_length)
            optimized_prompts.append(original_base_prompts[0])

            travel_logger.debug(f"[CONSOLIDATION_LOGIC] Final video: frames {video_start}-{final_frame} (raw: {raw_length}, quantized: {quantized_length})")
            travel_logger.debug(f"[CONSOLIDATION_LOGIC] Final video keyframes: {[kf - video_start for kf in video_keyframes]}")

            # SANITY CHECK: Consolidation should NEVER increase segment count
            original_num_segments = len(original_segment_frames)
            new_num_segments = len(optimized_segments)

            if new_num_segments > original_num_segments:
                # This should never happen - consolidation split segments instead of combining them!
                travel_logger.debug(f"[CONSOLIDATION_ERROR] ❌ Consolidation increased segments from {original_num_segments} to {new_num_segments} - ABORTING optimization")
                travel_logger.debug(f"[FRAME_CONSOLIDATION] ❌ ERROR: Consolidation would increase segments ({original_num_segments} → {new_num_segments}) - keeping original allocation")
                # Return early without modifying the payload
                return orchestrator_payload

            # Update orchestrator payload
            orchestrator_payload["segment_frames_expanded"] = optimized_segments
            orchestrator_payload["frame_overlap_expanded"] = optimized_overlaps
            orchestrator_payload["base_prompts_expanded"] = optimized_prompts
            orchestrator_payload["negative_prompts_expanded"] = [orchestrator_payload["negative_prompts_expanded"][0]] * len(optimized_segments)
            orchestrator_payload["num_new_segments_to_generate"] = len(optimized_segments)
            
            # CRITICAL FIX: Also remap enhanced_prompts_expanded to consolidated segment count
            # When segments are consolidated, the transitions are DIFFERENT (different start/end images),
            # so pre-existing enhanced prompts from original segments don't apply.
            # Set to empty strings to trigger VLM regeneration for the new consolidated transitions.
            original_enhanced_prompts = orchestrator_payload.get("enhanced_prompts_expanded", [])
            if original_enhanced_prompts and len(original_enhanced_prompts) != len(optimized_segments):
                travel_logger.debug(f"[FRAME_CONSOLIDATION] Remapping enhanced_prompts_expanded from {len(original_enhanced_prompts)} to {len(optimized_segments)} segments")
                travel_logger.debug(f"[FRAME_CONSOLIDATION] Setting enhanced_prompts to empty (consolidated transitions differ from originals, will regenerate via VLM)")
                orchestrator_payload["enhanced_prompts_expanded"] = [""] * len(optimized_segments)

            # CRITICAL: Store end anchor image indices for consolidated segments
            # This tells each consolidated segment which image should be its end anchor
            consolidated_end_anchors = []
            original_num_segments = len(original_segment_frames)

            # For consolidated segments, calculate the correct end anchor indices
            # Each consolidated segment should use the final image of its range
            # Use the simplified approach: track which images each segment should end with
            consolidated_end_anchors = []

            # First segment ends with the image at the last keyframe it contains
            if len(optimized_segments) >= 1:
                # First segment: determine which keyframes it contains based on consolidation logic
                # Recreate the consolidation to find the correct end images
                video_start = 0
                video_keyframes = [0]  # Always include first keyframe
                current_image_idx = 0

                for i in range(1, len(keyframe_positions)):
                    kf_pos = keyframe_positions[i]
                    video_length_if_included = kf_pos - video_start + 1

                    if video_length_if_included <= max_frames_per_segment:
                        # Keyframe fits in current video
                        video_keyframes.append(kf_pos)
                        current_image_idx = i  # This image index goes in current video
                    else:
                        # Finalize current video - end with current_image_idx
                        consolidated_end_anchors.append(current_image_idx)
                        travel_logger.debug(f"[FRAME_CONSOLIDATION] Segment {len(consolidated_end_anchors)-1}: end_anchor_image_index = {current_image_idx}")

                        # Start new video
                        video_start = video_keyframes[-1]
                        video_keyframes = [video_start, kf_pos]
                        current_image_idx = i  # Current keyframe goes in new video

                # Handle the final segment
                consolidated_end_anchors.append(current_image_idx)
                travel_logger.debug(f"[FRAME_CONSOLIDATION] Segment {len(consolidated_end_anchors)-1}: end_anchor_image_index = {current_image_idx}")

            # Store the end anchor mapping for use during segment creation
            orchestrator_payload["_consolidated_end_anchors"] = consolidated_end_anchors

            # Calculate relative keyframe positions AND image indices for each consolidated segment
            consolidated_keyframe_segments = []
            consolidated_keyframe_image_indices = []

            # Recreate the same consolidation logic to properly assign keyframes
            video_start = 0
            video_keyframes = [0]  # Always include first keyframe (absolute positions)
            video_image_indices = [0]  # Track which input images correspond to keyframes
            current_video_idx = 0

            for i in range(1, len(keyframe_positions)):
                kf_pos = keyframe_positions[i]
                video_length_if_included = kf_pos - video_start + 1

                if video_length_if_included <= max_frames_per_segment:
                    # Keyframe fits in current video
                    video_keyframes.append(kf_pos)
                    video_image_indices.append(i)  # Input image index corresponds to keyframe index
                else:
                    # Finalize current video and start new one
                    final_frame = video_keyframes[-1]

                    # Convert absolute keyframe positions to relative positions for this video
                    # BUT: adjust for quantization - keyframes must fit within quantized segment bounds
                    raw_length = final_frame - video_start + 1
                    quantized_length = _quantize_frames(raw_length, frame_step)

                    relative_keyframes = []
                    for kf_abs_pos in video_keyframes:
                        relative_pos = kf_abs_pos - video_start
                        # Ensure final keyframe fits within quantized bounds
                        if relative_pos >= quantized_length:
                            relative_pos = quantized_length - 1  # Last frame in quantized video
                        relative_keyframes.append(relative_pos)

                    consolidated_keyframe_segments.append(relative_keyframes)
                    consolidated_keyframe_image_indices.append(video_image_indices.copy())

                    travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx}: absolute start {video_start}, final frame {final_frame}")
                    travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} keyframes: {video_keyframes} → relative: {relative_keyframes}")
                    travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} image indices: {video_image_indices}")

                    # Start new video
                    current_video_idx += 1
                    video_start = final_frame  # Start from last keyframe (overlap)
                    # The overlap keyframe uses the same image as the final keyframe of previous segment
                    last_image_idx = video_image_indices[-1]
                    video_keyframes = [final_frame, kf_pos]  # Include overlap and current keyframe
                    video_image_indices = [last_image_idx, i]  # Include overlap image and current image

            # Handle the last video (make sure it has the correct final keyframes)
            if len(video_keyframes) > 0:
                # Convert absolute keyframe positions to relative positions for the final video
                # Adjust for quantization like the consolidation logic does
                final_frame = video_keyframes[-1]
                raw_length = final_frame - video_start + 1
                quantized_length = _quantize_frames(raw_length, frame_step)

                relative_keyframes = []
                for kf_abs_pos in video_keyframes:
                    relative_pos = kf_abs_pos - video_start
                    # Ensure final keyframe fits within quantized bounds
                    if relative_pos >= quantized_length:
                        relative_pos = quantized_length - 1  # Last frame in quantized video
                    relative_keyframes.append(relative_pos)

                consolidated_keyframe_segments.append(relative_keyframes)
                consolidated_keyframe_image_indices.append(video_image_indices.copy())

                travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx}: absolute start {video_start}")
                travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} keyframes: {video_keyframes} → relative: {relative_keyframes}")
                travel_logger.debug(f"[KEYFRAME_ASSIGNMENT] Video {current_video_idx} image indices: {video_image_indices}")

            # Store relative keyframe positions for guide video creation
            orchestrator_payload["_consolidated_keyframe_positions"] = consolidated_keyframe_segments

            travel_logger.debug(f"[FRAME_CONSOLIDATION] Optimized to {len(optimized_segments)} segments (was {len(original_segment_frames)})")
            travel_logger.debug(f"  - New segment frames: {optimized_segments}")
            travel_logger.debug(f"  - New overlaps: {optimized_overlaps}")
            travel_logger.debug(f"  - Efficiency: {(len(original_segment_frames) - len(optimized_segments))} fewer segments")

            return orchestrator_payload

        # --- SM_QUANTIZE_FRAMES_AND_OVERLAPS ---
        # Adjust all segment lengths to match model constraints (step*N+1 format).
        # Then, adjust overlap values to be even and not exceed the length of the
        # smaller of the two segments they connect. This prevents errors downstream
        # in guide video creation, generation, and stitching.

        travel_logger.debug(f"[FRAME_DEBUG] Orchestrator {orchestrator_task_id_str}: QUANTIZATION ANALYSIS")
        travel_logger.debug(f"[FRAME_DEBUG] Original segment_frames_expanded: {expanded_segment_frames}")
        travel_logger.debug(f"[FRAME_DEBUG] Original frame_overlap: {expanded_frame_overlap}")
        
        quantized_segment_frames = []
        travel_logger.debug(f"Orchestrator: Quantizing frame counts. Original segment_frames_expanded: {expanded_segment_frames}")
        for i, frames in enumerate(expanded_segment_frames):
            # Quantize to step*N+1 format to match model constraints (e.g. 4N+1 for Wan, 8N+1 for LTX-2)
            new_frames = _quantize_frames(frames, frame_step)
            travel_logger.debug(f"[FRAME_DEBUG] Segment {i}: {frames} -> {new_frames} ({frame_step}N+1 quantization)")
            if new_frames != frames:
                travel_logger.debug(f"Orchestrator: Quantized segment {i} length from {frames} to {new_frames} ({frame_step}*N+1 format).")
            quantized_segment_frames.append(new_frames)
        
        travel_logger.debug(f"[FRAME_DEBUG] Quantized segment_frames: {quantized_segment_frames}")
        travel_logger.debug(f"Orchestrator: Finished quantizing frame counts. New quantized_segment_frames: {quantized_segment_frames}")
        
        quantized_frame_overlap = []
        # There are N-1 overlaps for N segments. The loop must not iterate more times than this.
        num_overlaps_to_process = len(quantized_segment_frames) - 1
        travel_logger.debug(f"[FRAME_DEBUG] Processing {num_overlaps_to_process} overlap values")

        if num_overlaps_to_process > 0:
            for i in range(num_overlaps_to_process):
                # Gracefully handle if the original overlap array is longer than expected.
                if i < len(expanded_frame_overlap):
                    original_overlap = expanded_frame_overlap[i]
                else:
                    # This case should not happen if client is correct, but as a fallback.
                    travel_logger.debug(f"Orchestrator: Overlap at index {i} missing. Defaulting to 0.")
                    original_overlap = 0
                
                # Overlap connects segment i and i+1.
                # It cannot be larger than the shorter of the two segments.
                max_possible_overlap = min(quantized_segment_frames[i], quantized_segment_frames[i+1])

                # Quantize original overlap to be even, then cap it.
                new_overlap = (original_overlap // 2) * 2
                new_overlap = min(new_overlap, max_possible_overlap)
                if new_overlap < 0: new_overlap = 0

                travel_logger.debug(f"[FRAME_DEBUG] Overlap {i} (segments {i}->{i+1}): {original_overlap} -> {new_overlap}")
                travel_logger.debug(f"[FRAME_DEBUG]   Segment lengths: {quantized_segment_frames[i]}, {quantized_segment_frames[i+1]}")
                travel_logger.debug(f"[FRAME_DEBUG]   Max possible overlap: {max_possible_overlap}")
                
                if new_overlap != original_overlap:
                    travel_logger.debug(f"Orchestrator: Adjusted overlap between segments {i}-{i+1} from {original_overlap} to {new_overlap}.")
                
                quantized_frame_overlap.append(new_overlap)
        
        travel_logger.debug(f"[FRAME_DEBUG] Final quantized_frame_overlap: {quantized_frame_overlap}")
        
        # Persist quantised results back to orchestrator_payload so all downstream tasks see them
        orchestrator_payload["segment_frames_expanded"] = quantized_segment_frames
        orchestrator_payload["frame_overlap_expanded"] = quantized_frame_overlap
        
        # Calculate expected final length
        total_input_frames = sum(quantized_segment_frames)
        total_overlaps = sum(quantized_frame_overlap)
        expected_final_length = total_input_frames - total_overlaps
        travel_logger.debug(f"[FRAME_DEBUG] EXPECTED FINAL VIDEO:")
        travel_logger.debug(f"[FRAME_DEBUG]   Total input frames: {total_input_frames}")
        travel_logger.debug(f"[FRAME_DEBUG]   Total overlaps: {total_overlaps}")
        travel_logger.debug(f"[FRAME_DEBUG]   Expected final length: {expected_final_length} frames")
        travel_logger.debug(f"[FRAME_DEBUG]   Expected duration: {expected_final_length / orchestrator_payload.get('fps_helpers', 16):.2f}s")
        
        # Replace original lists with the new quantized ones for all subsequent logic
        expanded_segment_frames = quantized_segment_frames
        expanded_frame_overlap = quantized_frame_overlap
        # --- END SM_QUANTIZE_FRAMES_AND_OVERLAPS ---

        # If quantisation resulted in an empty overlap list (e.g. single-segment run) but the
        # original payload DID contain an overlap value, restore that so the first segment
        # can still reuse frames from the previous/continued video.  This is crucial for
        # continue-video journeys where we expect `frame_overlap_from_previous` > 0.
        if (not expanded_frame_overlap) and _orig_frame_overlap:
            expanded_frame_overlap = _orig_frame_overlap

        # --- FRAME CONSOLIDATION OPTIMIZATION (DISABLED) ---
        # This optimization was consolidating multiple segments into fewer when all prompts
        # were identical. Commented out to ensure all requested segments are created.
        # To re-enable, uncomment this section.
        #
        # # Store original values for comparison logging
        # original_num_segments = num_segments
        # original_segment_frames = list(expanded_segment_frames)
        # original_frame_overlap = list(expanded_frame_overlap)
        #
        # # Check if all prompts and LoRAs are identical to enable frame consolidation
        # # IMPORTANT: Disable consolidation if enhance_prompt is enabled, because VLM will
        # # generate different prompts for each segment AFTER consolidation would have run.
        # # This would create consolidated segments with different prompts, breaking the
        # # consolidation assumption that all segments have identical parameters.
        # enhance_prompt_enabled = orchestrator_payload.get("enhance_prompt", False)
        # if enhance_prompt_enabled:
        #     travel_logger.debug(f"[FRAME_CONSOLIDATION] enhance_prompt=True detected - DISABLING frame consolidation")
        #     travel_logger.debug(f"[FRAME_CONSOLIDATION] Reason: VLM will generate unique prompts per segment, breaking identity assumption")
        #     identity_analysis = {
        #         "can_optimize_frames": False,
        #         "can_reuse_model": False,
        #         "is_identical": False
        #     }
        # else:
        #     identity_analysis = detect_identical_parameters(orchestrator_payload, num_segments)
        #
        # if identity_analysis["can_optimize_frames"]:
        #     # Only run consolidation if there are multiple segments to consolidate
        #     num_segments = len(orchestrator_payload["segment_frames_expanded"])
        #     if num_segments <= 1:
        #         travel_logger.debug(f"[FRAME_CONSOLIDATION] ⏭️  Skipping optimization - only {num_segments} segment(s), nothing to consolidate")
        #         travel_logger.info(f"Frame consolidation: Only {num_segments} segment(s) - no consolidation needed", task_id=orchestrator_task_id_str)
        #     else:
        #         # Run safety validation before optimization
        #         safety_check = validate_consolidation_safety(orchestrator_payload)
        #
        #         if safety_check["is_safe"]:
        #             travel_logger.debug(f"[FRAME_CONSOLIDATION] ✅ Triggering optimization for identical parameters")
        #             travel_logger.info("Frame consolidation: All parameters identical - enabling optimization", task_id=orchestrator_task_id_str)
        #
        #             # Apply frame consolidation optimization
        #             orchestrator_payload = optimize_frame_allocation_for_identical_params(
        #                 orchestrator_payload,
        #                 max_frames_per_segment=81,  # Max 81 frames per video
        #             )
        #
        #             # Update variables with optimized values
        #             expanded_segment_frames = orchestrator_payload["segment_frames_expanded"]
        #             expanded_frame_overlap = orchestrator_payload["frame_overlap_expanded"]
        #             expanded_base_prompts = orchestrator_payload["base_prompts_expanded"]
        #             expanded_negative_prompts = orchestrator_payload["negative_prompts_expanded"]
        #             num_segments = orchestrator_payload["num_new_segments_to_generate"]
        #
        #             # CRITICAL: Update VACE image references for consolidated segments
        #             # When segments are consolidated, we need to reassign all VACE image refs
        #             # to the new consolidated segments based on their new indices
        #             if vace_refs_instructions_all:
        #                 travel_logger.debug(f"[VACE_REFS_CONSOLIDATION] Updating VACE image refs for {num_segments} consolidated segments")
        #                 travel_logger.debug(f"[VACE_REFS_CONSOLIDATION] Original VACE refs count: {len(vace_refs_instructions_all)}")
        #
        #                 # For consolidated segments, reassign all VACE refs to the appropriate new segment
        #                 # based on the original keyframe positions and new segment boundaries
        #                 for ref_idx, ref_instr in enumerate(vace_refs_instructions_all):
        #                     original_segment_idx = ref_instr.get("segment_idx_for_naming", 0)
        #
        #                     # For now, assign all refs to the first (and often only) consolidated segment
        #                     # This ensures the consolidated segment gets all the keyframe images
        #                     new_segment_idx = 0 if num_segments == 1 else min(original_segment_idx, num_segments - 1)
        #
        #                     if original_segment_idx != new_segment_idx:
        #                         travel_logger.debug(f"[VACE_REFS_CONSOLIDATION] VACE ref {ref_idx}: segment {original_segment_idx} → {new_segment_idx}")
        #                         ref_instr["segment_idx_for_naming"] = new_segment_idx
        #
        #                 travel_logger.debug(f"[VACE_REFS_CONSOLIDATION] VACE refs updated for consolidated segments")
        #
        #             # Summary logging for optimization results
        #             segments_saved = original_num_segments - num_segments
        #             travel_logger.info(f"Frame consolidation optimization: {original_num_segments} → {num_segments} segments (saved {segments_saved})", task_id=orchestrator_task_id_str)
        #             travel_logger.debug(f"Original allocation: {original_segment_frames}, Optimized: {expanded_segment_frames}", task_id=orchestrator_task_id_str)
        #             travel_logger.debug(f"Original overlaps: {original_frame_overlap}, Optimized: {expanded_frame_overlap}", task_id=orchestrator_task_id_str)
        #
        #             travel_logger.debug(f"[FRAME_CONSOLIDATION] Successfully updated to {num_segments} optimized segments")
        #         else:
        #             travel_logger.warning("Frame consolidation: Safety validation failed - parameters not identical enough", task_id=orchestrator_task_id_str)
        #             travel_logger.debug(f"[FRAME_CONSOLIDATION] Safety check failed - keeping original allocation")
        # else:
        #     travel_logger.info("Frame consolidation: Parameters not identical - using standard allocation", task_id=orchestrator_task_id_str)
        #     travel_logger.debug(f"[FRAME_CONSOLIDATION] Parameters not identical - keeping original allocation")
        
        travel_logger.debug(f"[FRAME_CONSOLIDATION] Consolidation disabled - using original {num_segments} segments as specified")
        # --- END FRAME CONSOLIDATION OPTIMIZATION ---

        # =============================================================================
        # STRUCTURE VIDEO PROCESSING (Single or Multi-Source Composite)
        # =============================================================================
        using_travel_guidance = isinstance(orchestrator_payload.get("travel_guidance"), dict)
        travel_guidance_config: Optional[TravelGuidanceConfig]
        structure_config: Optional[StructureGuidanceConfig]

        if using_travel_guidance:
            travel_guidance_config = TravelGuidanceConfig.from_payload(
                orchestrator_payload,
                orchestrator_payload.get("model_name", ""),
            )
            structure_config = (
                travel_guidance_config.to_structure_guidance_config()
                if travel_guidance_config.kind in {"vace", "uni3c"}
                else None
            )
            travel_logger.debug(f"[TRAVEL_GUIDANCE] Parsed: {travel_guidance_config}")
        else:
            structure_config = StructureGuidanceConfig.from_params(orchestrator_payload)
            travel_logger.debug(f"[STRUCTURE_CONFIG] Parsed: {structure_config}")
            if structure_config.has_guidance:
                travel_guidance_config = TravelGuidanceConfig(
                    kind="uni3c" if structure_config.is_uni3c else "vace",
                    videos=list(structure_config.videos),
                    strength=structure_config.strength,
                    mode=(
                        ""
                        if structure_config.is_uni3c
                        else (
                            "raw"
                            if structure_config.preprocessing == "none"
                            else structure_config.preprocessing
                        )
                    ),
                    canny_intensity=structure_config.canny_intensity,
                    depth_contrast=structure_config.depth_contrast,
                    step_window=structure_config.step_window,
                    frame_policy=structure_config.frame_policy,
                    zero_empty_frames=structure_config.zero_empty_frames,
                    keep_on_gpu=structure_config.keep_on_gpu,
                )
            else:
                travel_guidance_config = TravelGuidanceConfig(kind="none")

        # Extract values for guidance processing
        if using_travel_guidance and travel_guidance_config.kind != "none":
            structure_videos = [video.to_dict() for video in travel_guidance_config.videos]
            structure_video_path = None
            structure_type = travel_guidance_config.legacy_structure_type
            motion_strength = (
                travel_guidance_config.control_strength
                if travel_guidance_config.is_ltx_hybrid
                else travel_guidance_config.strength
            )
            canny_intensity = travel_guidance_config.canny_intensity
            depth_contrast = travel_guidance_config.depth_contrast
        else:
            structure_videos = orchestrator_payload.get("structure_videos", [])
            if not structure_videos and structure_config and structure_config.videos:
                structure_videos = [v.to_dict() for v in structure_config.videos]
                travel_logger.debug(
                    f"[STRUCTURE_CONFIG] Extracted {len(structure_videos)} videos from structure_guidance.videos"
                )
            structure_video_path = orchestrator_payload.get("structure_video_path")
            structure_type = (
                structure_config.legacy_structure_type
                if structure_config and structure_config.has_guidance
                else None
            )
            motion_strength = structure_config.strength if structure_config else 1.0
            canny_intensity = structure_config.canny_intensity if structure_config else 1.0
            depth_contrast = structure_config.depth_contrast if structure_config else 1.0

        segment_flow_offsets = []
        total_flow_frames = 0
        
        # =============================================================================
        # Calculate TWO different timelines:
        # 1. STITCHED TIMELINE: Final output length after overlaps removed (what user sees)
        #    - Used for multi-structure-video (start_frame/end_frame are in this space)
        # 2. GUIDANCE TIMELINE: Internal length where overlaps are "reused" 
        #    - Used for legacy single structure video (backwards compat)
        # =============================================================================
        
        # STITCHED TIMELINE: sum(frames) - sum(overlaps)
        # This is what the user sees and where image keyframes are positioned
        segment_stitched_offsets, total_stitched_frames = _calculate_segment_stitched_offsets(
            expanded_segment_frames,
            expanded_frame_overlap,
        )
        
        # GUIDANCE TIMELINE: Legacy calculation for backwards compatibility
        # This has overlapping regions that segments "reuse"
        for idx in range(num_segments):
            segment_total_frames = expanded_segment_frames[idx]
            if idx == 0 and not orchestrator_payload.get("continue_from_video_resolved_path"):
                segment_flow_offsets.append(0)
                total_flow_frames = segment_total_frames
            else:
                overlap = expanded_frame_overlap[idx - 1] if idx > 0 else 0
                segment_offset = total_flow_frames - overlap
                segment_flow_offsets.append(segment_offset)
                total_flow_frames += segment_total_frames
        
        travel_logger.debug(f"[STRUCTURE_VIDEO] Stitched timeline: {total_stitched_frames} frames")
        travel_logger.debug(f"[STRUCTURE_VIDEO] Stitched segment offsets: {segment_stitched_offsets}")
        travel_logger.debug(f"[STRUCTURE_VIDEO] Guidance timeline (legacy): {total_flow_frames} frames")
        travel_logger.debug(f"[STRUCTURE_VIDEO] Guidance segment offsets (legacy): {segment_flow_offsets}")
        
        # Determines which timeline offsets segments use for guidance slicing.
        # True  → stitched timeline (multi-video / travel_guidance)
        # False → legacy guidance timeline (single video / no guidance)
        use_stitched_offsets = False

        # =============================================================================
        # PATH A: travel_guidance-guided travel (vace / uni3c / ltx_control)
        # =============================================================================
        if using_travel_guidance:
            if travel_guidance_config.kind == "none":
                use_stitched_offsets = False
                travel_logger.info("travel_guidance kind=none: skipping guidance video processing", task_id=orchestrator_task_id_str)
            else:
                travel_logger.info(
                    f"travel_guidance mode: kind={travel_guidance_config.kind}, videos={len(structure_videos)}",
                    task_id=orchestrator_task_id_str,
                )

                try:
                    from source.media.structure import create_composite_guidance_video

                    structure_type = travel_guidance_config.get_preprocessor_type()

                    target_resolution_raw = orchestrator_payload["parsed_resolution_wh"]
                    target_fps = orchestrator_payload.get("fps_helpers", 16)

                    if isinstance(target_resolution_raw, str):
                        parsed_res = parse_resolution(target_resolution_raw)
                        if parsed_res is None:
                            raise ValueError(f"Invalid resolution format: {target_resolution_raw}")
                        grid_size = get_model_grid_size(orchestrator_payload.get("model_name"))
                        target_resolution = snap_resolution_to_model_grid(parsed_res, grid_size)
                        orchestrator_payload["parsed_resolution_wh"] = f"{target_resolution[0]}x{target_resolution[1]}"
                        travel_logger.debug(
                            f"[TRAVEL_GUIDANCE] Resolution snapped (grid={grid_size}): "
                            f"{target_resolution_raw} → {orchestrator_payload['parsed_resolution_wh']}"
                        )
                    else:
                        target_resolution = target_resolution_raw

                    timestamp_short = datetime.now().strftime("%H%M%S")
                    unique_suffix = uuid.uuid4().hex[:6]
                    composite_filename = (
                        f"travel_guidance_{travel_guidance_config.kind}_{structure_type}_"
                        f"{timestamp_short}_{unique_suffix}.mp4"
                    )

                    composite_guidance_path = create_composite_guidance_video(
                        structure_configs=structure_videos,
                        total_frames=total_stitched_frames,
                        structure_type=structure_type,
                        target_resolution=target_resolution,
                        target_fps=target_fps,
                        output_path=current_run_output_dir / composite_filename,
                        motion_strength=motion_strength,
                        canny_intensity=canny_intensity,
                        depth_contrast=depth_contrast,
                        download_dir=current_run_output_dir,
                    )

                    use_stitched_offsets = True
                    structure_guidance_video_url = upload_and_get_final_output_location(
                        local_file_path=composite_guidance_path,
                        initial_db_location=str(composite_guidance_path),
                    )

                    guidance_frame_count, _ = get_video_frame_count_and_fps(composite_guidance_path)
                    travel_logger.success(
                        f"Travel guidance video created: {guidance_frame_count} frames",
                        task_id=orchestrator_task_id_str,
                    )

                    travel_guidance_config._guidance_video_url = structure_guidance_video_url
                    if structure_config is not None:
                        structure_config._guidance_video_url = structure_guidance_video_url
                except (OSError, ValueError, RuntimeError) as e:
                    travel_logger.error(
                        f"Failed to create travel guidance video: {e}",
                        task_id=orchestrator_task_id_str,
                        exc_info=True,
                    )
                    travel_logger.warning(
                        "Travel guidance will not be available for this generation",
                        task_id=orchestrator_task_id_str,
                    )
                    travel_guidance_config._guidance_video_url = None
                    if structure_config is not None:
                        structure_config._guidance_video_url = None

        # =============================================================================
        # PATH B: Multi-Structure Video (legacy/new internal structure_guidance path)
        # =============================================================================
        elif structure_videos and len(structure_videos) > 0:
            travel_logger.info(f"Multi-structure video mode: {len(structure_videos)} configs", task_id=orchestrator_task_id_str)
            
            try:
                from source.media.structure import (
                    create_composite_guidance_video)
                
                # Validate and extract structure_type from configs (must all match)
                # Prefer structure_config.legacy_structure_type (set earlier), then check per-video configs
                if structure_type:
                    # Already have structure_type from structure_config (new format)
                    travel_logger.debug(f"[STRUCTURE_VIDEO] Using structure_type from config: {structure_type}")
                else:
                    # Legacy format: extract from per-video configs
                    structure_types_found = set()
                    for cfg in structure_videos:
                        cfg_type = cfg.get("structure_type", cfg.get("type", "flow"))
                        structure_types_found.add(cfg_type)

                    if len(structure_types_found) > 1:
                        raise ValueError(f"All structure_videos must have same type, found: {structure_types_found}")

                    structure_type = structure_types_found.pop() if structure_types_found else "flow"

                # Validate structure_type
                if structure_type not in ["flow", "canny", "depth", "raw", "uni3c"]:
                    raise ValueError(f"Invalid structure_type: {structure_type}. Must be 'flow', 'canny', 'depth', 'raw', or 'uni3c'")
                
                # Use strength parameters from structure_config (already extracted earlier)
                # These handle both new format (structure_guidance.strength) and legacy formats
                travel_logger.debug(f"[STRUCTURE_VIDEO] Using strength params: motion={motion_strength}, canny={canny_intensity}, depth={depth_contrast}")
                
                # Log configs
                for i, cfg in enumerate(structure_videos):
                    cfg_motion = cfg.get("motion_strength", motion_strength)
                    travel_logger.debug(f"[STRUCTURE_VIDEO] Config {i}: frames [{cfg.get('start_frame')}, {cfg.get('end_frame')}) "
                           f"from {Path(cfg.get('path', 'unknown')).name}, "
                           f"source_range=[{cfg.get('source_start_frame', 0)}, {cfg.get('source_end_frame', 'end')}), "
                           f"treatment={cfg.get('treatment', 'adjust')}, motion_strength={cfg_motion}")
                
                # Get resolution and FPS
                target_resolution_raw = orchestrator_payload["parsed_resolution_wh"]
                target_fps = orchestrator_payload.get("fps_helpers", 16)
                
                if isinstance(target_resolution_raw, str):
                    parsed_res = parse_resolution(target_resolution_raw)
                    if parsed_res is None:
                        raise ValueError(f"Invalid resolution format: {target_resolution_raw}")
                    grid_size = get_model_grid_size(orchestrator_payload.get("model_name"))
                    target_resolution = snap_resolution_to_model_grid(parsed_res, grid_size)
                    orchestrator_payload["parsed_resolution_wh"] = f"{target_resolution[0]}x{target_resolution[1]}"
                    travel_logger.debug(f"[STRUCTURE_VIDEO] Resolution snapped (grid={grid_size}): {target_resolution_raw} → {orchestrator_payload['parsed_resolution_wh']}")
                else:
                    target_resolution = target_resolution_raw
                
                # Generate unique filename
                timestamp_short = datetime.now().strftime("%H%M%S")
                unique_suffix = uuid.uuid4().hex[:6]
                composite_filename = f"structure_composite_{structure_type}_{timestamp_short}_{unique_suffix}.mp4"
                
                travel_logger.info(f"Creating composite guidance video ({structure_type})...", task_id=orchestrator_task_id_str)
                
                # Create composite guidance video
                # Use STITCHED timeline for multi-structure video
                # This ensures start_frame/end_frame match user's mental model (image positions)
                composite_guidance_path = create_composite_guidance_video(
                    structure_configs=structure_videos,
                    total_frames=total_stitched_frames,  # Stitched timeline, not guidance timeline!
                    structure_type=structure_type,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    output_path=current_run_output_dir / composite_filename,
                    motion_strength=motion_strength,
                    canny_intensity=canny_intensity,
                    depth_contrast=depth_contrast,
                    download_dir=current_run_output_dir)
                
                # Flag to use stitched offsets for segment payloads
                use_stitched_offsets = True
                
                # Upload composite
                structure_guidance_video_url = upload_and_get_final_output_location(
                    local_file_path=composite_guidance_path,
                    initial_db_location=str(composite_guidance_path))
                
                # Get frame count for logging
                guidance_frame_count, _ = get_video_frame_count_and_fps(composite_guidance_path)
                travel_logger.success(
                    f"Composite guidance video created: {guidance_frame_count} frames, {len(structure_videos)} sources",
                    task_id=orchestrator_task_id_str
                )
                
                # Store guidance URL in config (unified format)
                if structure_config is not None:
                    structure_config._guidance_video_url = structure_guidance_video_url
                
            except (OSError, ValueError, RuntimeError) as e:
                travel_logger.error(f"Failed to create composite guidance video: {e}", task_id=orchestrator_task_id_str, exc_info=True)
                travel_logger.warning("Structure guidance will not be available for this generation", task_id=orchestrator_task_id_str)
                if structure_config is not None:
                    structure_config._guidance_video_url = None

        # =============================================================================
        # PATH B: Legacy Single Structure Video (structure_video_path)
        # =============================================================================
        elif structure_video_path:
            # Legacy single structure video - uses guidance timeline (not stitched)
            use_stitched_offsets = False
            structure_video_treatment = orchestrator_payload.get("structure_video_treatment", "adjust")
            structure_type = orchestrator_payload.get("structure_video_type", orchestrator_payload.get("structure_type", "flow"))
            travel_logger.info(f"Single structure video mode: type={structure_type}, treatment={structure_video_treatment}", task_id=orchestrator_task_id_str)

            # Extract strength parameters
            motion_strength = orchestrator_payload.get("structure_video_motion_strength", 1.0)
            canny_intensity = orchestrator_payload.get("structure_canny_intensity", 1.0)
            depth_contrast = orchestrator_payload.get("structure_depth_contrast", 1.0)
            
            # Capture original URL if remote
            if isinstance(structure_video_path, str) and structure_video_path.startswith(("http://", "https://")):
                 orchestrator_payload["structure_original_video_url"] = structure_video_path

            # Download if URL
            from ...utils import download_video_if_url
            structure_video_path = download_video_if_url(
                structure_video_path,
                download_target_dir=current_run_output_dir,
                task_id_for_logging=orchestrator_task_id_str,
                descriptive_name="structure_video"
            )
            
            # Validate
            if not Path(structure_video_path).exists():
                raise ValueError(f"Structure video not found: {structure_video_path}")
            if structure_video_treatment not in ["adjust", "clip"]:
                raise ValueError(f"Invalid structure_video_treatment: {structure_video_treatment}")
            if structure_type not in ["flow", "canny", "depth", "raw", "uni3c"]:
                raise ValueError(f"Invalid structure_type: {structure_type}. Must be 'flow', 'canny', 'depth', 'raw', or 'uni3c'")

            travel_logger.info(f"Structure video processing: {total_flow_frames} total frames needed", task_id=orchestrator_task_id_str)

            # Create guidance video
            try:
                from source.media.structure import create_structure_guidance_video, create_trimmed_structure_video
                
                target_resolution_raw = orchestrator_payload["parsed_resolution_wh"]
                target_fps = orchestrator_payload.get("fps_helpers", 16)
                
                if isinstance(target_resolution_raw, str):
                    parsed_res = parse_resolution(target_resolution_raw)
                    if parsed_res is None:
                        raise ValueError(f"Invalid resolution format: {target_resolution_raw}")
                    grid_size = get_model_grid_size(orchestrator_payload.get("model_name"))
                    target_resolution = snap_resolution_to_model_grid(parsed_res, grid_size)
                    orchestrator_payload["parsed_resolution_wh"] = f"{target_resolution[0]}x{target_resolution[1]}"
                else:
                    target_resolution = target_resolution_raw
                
                timestamp_short = datetime.now().strftime("%H%M%S")
                unique_suffix = uuid.uuid4().hex[:6]
                
                # Create trimmed video
                trimmed_filename = f"structure_trimmed_{timestamp_short}_{unique_suffix}.mp4"
                trimmed_video_path = create_trimmed_structure_video(
                    structure_video_path=structure_video_path,
                    max_frames_needed=total_flow_frames,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    output_path=current_run_output_dir / trimmed_filename,
                    treatment=structure_video_treatment)
                
                trimmed_video_url = upload_and_get_final_output_location(
                    local_file_path=trimmed_video_path,
                    initial_db_location=str(trimmed_video_path))
                orchestrator_payload["structure_trimmed_video_url"] = trimmed_video_url

                # Create guidance video
                structure_guidance_filename = f"structure_{structure_type}_{timestamp_short}_{unique_suffix}.mp4"
                structure_guidance_video_path = create_structure_guidance_video(
                    structure_video_path=structure_video_path,
                    max_frames_needed=total_flow_frames,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                    output_path=current_run_output_dir / structure_guidance_filename,
                    structure_type=structure_type,
                    motion_strength=motion_strength,
                    canny_intensity=canny_intensity,
                    depth_contrast=depth_contrast,
                    treatment=structure_video_treatment)
                
                structure_guidance_video_url = upload_and_get_final_output_location(
                    local_file_path=structure_guidance_video_path,
                    initial_db_location=str(structure_guidance_video_path))

                guidance_frame_count, _ = get_video_frame_count_and_fps(structure_guidance_video_path)
                travel_logger.success(f"Structure guidance video created: {guidance_frame_count} frames", task_id=orchestrator_task_id_str)

                # Store guidance URL in config (unified format)
                if structure_config is not None:
                    structure_config._guidance_video_url = structure_guidance_video_url

            except (OSError, ValueError, RuntimeError) as e:
                travel_logger.error(f"Failed to create structure guidance video: {e}", task_id=orchestrator_task_id_str, exc_info=True)
                travel_logger.warning("Structure guidance will not be available", task_id=orchestrator_task_id_str)
                if structure_config is not None:
                    structure_config._guidance_video_url = None

        # =============================================================================
        # PATH C: No Structure Video
        # =============================================================================
        else:
            use_stitched_offsets = False  # Doesn't matter, but initialize for consistency
            travel_logger.info("No structure video configured", task_id=orchestrator_task_id_str)

        # --- ENHANCED PROMPTS HANDLING ---
        # First, load any pre-existing enhanced prompts from the payload (regardless of enhance_prompt flag)
        # Then, if enhance_prompt=True, run VLM only for segments that don't have prompts yet
        vlm_enhanced_prompts = {}  # Dict: segment_idx -> enhanced_prompt

        # Always use pre-existing enhanced prompts from payload if available
        payload_enhanced_prompts = orchestrator_payload.get("enhanced_prompts_expanded", []) or []
        if payload_enhanced_prompts:
            for idx, prompt in enumerate(payload_enhanced_prompts):
                if prompt and prompt.strip():
                    vlm_enhanced_prompts[idx] = prompt
            if vlm_enhanced_prompts:
                travel_logger.debug(f"[ENHANCED_PROMPTS] Loaded {len(vlm_enhanced_prompts)} pre-existing enhanced prompts from payload")
                for idx, prompt in vlm_enhanced_prompts.items():
                    travel_logger.debug(f"[ENHANCED_PROMPTS]   Segment {idx}: '{prompt[:80]}...'")

        # Run VLM for segments that still need prompts (only if enhance_prompt is enabled)
        segments_needing_vlm = [idx for idx in range(num_segments) if idx not in vlm_enhanced_prompts]
        if orchestrator_payload.get("enhance_prompt", False) and not segments_needing_vlm:
            travel_logger.debug(f"[VLM_BATCH] enhance_prompt enabled but all {num_segments} segments already have prompts - skipping VLM")
        elif orchestrator_payload.get("enhance_prompt", False) and segments_needing_vlm:
            travel_logger.debug(f"[VLM_BATCH] enhance_prompt enabled - {len(segments_needing_vlm)} segments need VLM enrichment")
            log_ram_usage("Before VLM loading", task_id=orchestrator_task_id_str)
            try:
                # Import VLM helper
                from ...media.vlm import generate_transition_prompts_batch
                from ...utils import download_image_if_url

                # Get input images
                input_images_resolved = orchestrator_payload.get("input_image_paths_resolved", [])
                vlm_device = orchestrator_payload.get("vlm_device", "cuda")
                base_prompt = orchestrator_payload.get("base_prompt", "")
                _fps_helpers = orchestrator_payload.get("fps_helpers", 16)

                # [VLM_INPUT_DEBUG] Log the source images array to verify ordering
                travel_logger.debug(f"[VLM_INPUT_DEBUG] input_images_resolved from payload ({len(input_images_resolved)} images):")
                for i, img in enumerate(input_images_resolved):
                    img_name = Path(img).name if img else 'NONE'
                    travel_logger.debug(f"[VLM_INPUT_DEBUG]   [{i}]: {img_name}")

                if base_prompt:
                    travel_logger.debug(f"[VLM_BATCH] Base prompt from payload: '{base_prompt[:80]}...'")

                # Detect single-image mode: only 1 image, no transition to describe
                is_single_image_mode = len(input_images_resolved) == 1

                if is_single_image_mode:
                    travel_logger.debug(f"[VLM_SINGLE] Detected single-image mode - using single-image VLM prompt generation")

                    # Import single-image VLM helper
                    from ...media.vlm import generate_single_image_prompts_batch

                    # Build lists for single-image batch processing
                    single_images = []
                    single_base_prompts = []
                    single_indices = []

                    for idx in segments_needing_vlm:
                        
                        # For single-image mode, use the only image we have
                        image_path = input_images_resolved[0]
                        
                        # Download image if it's a URL
                        image_path = download_image_if_url(
                            image_path,
                            current_run_output_dir,
                            f"vlm_single_{idx}",
                            debug_mode=False,
                            descriptive_name=f"vlm_single_seg{idx}"
                        )
                        
                        single_images.append(image_path)
                        segment_base_prompt = expanded_base_prompts[idx] if expanded_base_prompts[idx] and expanded_base_prompts[idx].strip() else base_prompt
                        single_base_prompts.append(segment_base_prompt)
                        single_indices.append(idx)
                    
                    # Generate prompts for single images
                    if single_images:
                        travel_logger.debug(f"[VLM_SINGLE] Processing {len(single_images)} segment(s) with single-image VLM...")
                        
                        enhanced_prompts = generate_single_image_prompts_batch(
                            image_paths=single_images,
                            base_prompts=single_base_prompts,
                            device=vlm_device)
                        
                        # Map results back to segment indices
                        for idx, enhanced in zip(single_indices, enhanced_prompts):
                            vlm_enhanced_prompts[idx] = enhanced
                            travel_logger.debug(f"[VLM_SINGLE] Segment {idx}: {enhanced[:80]}...")
                    
                    # Skip the transition-based processing below
                    image_pairs = []
                    segment_indices = []
                
                else:
                    # Multi-image mode: build lists of image pairs for transitions
                    image_pairs = []
                    base_prompts_for_batch = []
                    segment_indices = []  # Track which segment each pair belongs to

                    for idx in segments_needing_vlm:
                        # Determine which images this segment transitions between
                        if orchestrator_payload.get("_consolidated_end_anchors"):
                            consolidated_end_anchors = orchestrator_payload["_consolidated_end_anchors"]
                            if idx < len(consolidated_end_anchors):
                                end_anchor_idx = consolidated_end_anchors[idx]
                                start_anchor_idx = 0 if idx == 0 else consolidated_end_anchors[idx - 1]
                            else:
                                start_anchor_idx = idx
                                end_anchor_idx = idx + 1
                        else:
                            start_anchor_idx = idx
                            end_anchor_idx = idx + 1

                        # Ensure indices are within bounds
                        if (start_anchor_idx < len(input_images_resolved) and
                            end_anchor_idx < len(input_images_resolved)):

                            start_image_url = input_images_resolved[start_anchor_idx]
                            end_image_url = input_images_resolved[end_anchor_idx]
                            
                            # [VLM_URL_DEBUG] Log the FULL source URLs (clickable in logs)
                            start_url_name = Path(start_image_url).name if start_image_url else 'NONE'
                            end_url_name = Path(end_image_url).name if end_image_url else 'NONE'
                            travel_logger.debug(f"[VLM_URL_DEBUG] ═══════════════════════════════════════════════════════")
                            travel_logger.debug(f"[VLM_URL_DEBUG] Segment {idx}: Downloading images for VLM")
                            travel_logger.debug(f"[VLM_URL_DEBUG]   START (array idx={start_anchor_idx}): {start_url_name}")
                            travel_logger.debug(f"[VLM_URL_DEBUG]   START FULL URL: {start_image_url}")
                            travel_logger.debug(f"[VLM_URL_DEBUG]   END   (array idx={end_anchor_idx}): {end_url_name}")
                            travel_logger.debug(f"[VLM_URL_DEBUG]   END   FULL URL: {end_image_url}")

                            # Download images if they're URLs
                            start_image_path = download_image_if_url(
                                start_image_url,
                                current_run_output_dir,
                                f"vlm_start_{idx}",
                                debug_mode=False,
                                descriptive_name=f"vlm_start_seg{idx}"
                            )
                            end_image_path = download_image_if_url(
                                end_image_url,
                                current_run_output_dir,
                                f"vlm_end_{idx}",
                                debug_mode=False,
                                descriptive_name=f"vlm_end_seg{idx}"
                            )
                            
                            # [VLM_URL_DEBUG] Log the downloaded local paths
                            travel_logger.debug(f"[VLM_URL_DEBUG]   START downloaded to: {Path(start_image_path).name}")
                            travel_logger.debug(f"[VLM_URL_DEBUG]   END   downloaded to: {Path(end_image_path).name}")

                            image_pairs.append((start_image_path, end_image_path))
                            # Use segment-specific base_prompt if available, otherwise use overall base_prompt
                            segment_base_prompt = expanded_base_prompts[idx] if expanded_base_prompts[idx] and expanded_base_prompts[idx].strip() else base_prompt
                            base_prompts_for_batch.append(segment_base_prompt)
                            segment_indices.append(idx)
                        else:
                            travel_logger.debug(f"[VLM_BATCH] Segment {idx}: Skipping - image indices out of bounds (start={start_anchor_idx}, end={end_anchor_idx}, available={len(input_images_resolved)})")

                # Generate all prompts in one batch (reuses VLM model)
                if image_pairs:
                    # [VLM_IMAGE_DEBUG] Log exactly what image pairs VLM will process
                    travel_logger.debug(f"[VLM_IMAGE_DEBUG] About to call VLM with {len(image_pairs)} image pairs:")
                    for i, ((start, end), base_prompt) in enumerate(zip(image_pairs, base_prompts_for_batch)):
                        seg_idx = segment_indices[i]
                        start_name = Path(start).name if start else 'NONE'
                        end_name = Path(end).name if end else 'NONE'
                        prompt_preview = base_prompt[:50] if base_prompt else 'EMPTY'
                        travel_logger.debug(f"[VLM_IMAGE_DEBUG]   Pair {i} (segment {seg_idx}): {start_name} → {end_name}")
                        travel_logger.debug(f"[VLM_IMAGE_DEBUG]     Base prompt: '{prompt_preview}...'")

                    enhanced_prompts = generate_transition_prompts_batch(
                        image_pairs=image_pairs,
                        base_prompts=base_prompts_for_batch,
                        device=vlm_device,
                        task_id=orchestrator_task_id_str,
                        upload_debug_images=True  # Upload VLM debug images for remote inspection
                    )

                    # Map results back to segment indices
                    for idx, enhanced in zip(segment_indices, enhanced_prompts):
                        vlm_enhanced_prompts[idx] = enhanced
                        travel_logger.debug(f"[VLM_BATCH] Segment {idx}: {enhanced[:80]}...")

                travel_logger.debug(f"[VLM_BATCH] Generated {len(vlm_enhanced_prompts)} enhanced prompts")
                log_ram_usage("After VLM cleanup", task_id=orchestrator_task_id_str)

                # Call Supabase edge function to update shot_generations with newly enriched prompts
                try:
                    import httpx

                    # Build complete enhanced_prompts array (empty strings for non-enriched segments)
                    complete_enhanced_prompts = []
                    for idx in range(num_segments):
                        if idx in vlm_enhanced_prompts:
                            complete_enhanced_prompts.append(vlm_enhanced_prompts[idx])
                        else:
                            complete_enhanced_prompts.append("")

                    # Only call if we have SUPABASE configured and generated any new prompts
                    # Use SERVICE_KEY if available (admin), otherwise use ACCESS_TOKEN (user with ownership check)
                    auth_token = db_config.SUPABASE_SERVICE_KEY or db_config.SUPABASE_ACCESS_TOKEN
                    if db_config.SUPABASE_URL and auth_token and len(complete_enhanced_prompts) > 0:
                        # Extract shot_id from orchestrator_payload
                        shot_id = orchestrator_payload.get("shot_id")
                        if not shot_id:
                            travel_logger.debug(f"[VLM_BATCH] WARNING: No shot_id found in orchestrator_payload, skipping edge function call")
                        else:
                            # Call edge function to update shot_generations with enhanced prompts
                            edge_url = f"{db_config.SUPABASE_URL.rstrip('/')}/functions/v1/update-shot-pair-prompts"
                            headers = {"Content-Type": "application/json"}
                            if auth_token:
                                headers["Authorization"] = f"Bearer {auth_token}"

                            payload = {
                                "shot_id": shot_id,
                                "task_id": orchestrator_task_id_str,  # Links logs to orchestrator task
                                "enhanced_prompts": complete_enhanced_prompts
                            }

                            travel_logger.debug(f"[VLM_BATCH] Calling edge function to update shot_generations with enhanced prompts...")
                            travel_logger.debug(f"[VLM_BATCH] Payload: shot_id={shot_id}, task_id={orchestrator_task_id_str}, enhanced_prompts={len(complete_enhanced_prompts)} items")
                            
                            # [EDGE_FUNC_DEBUG] Log what we're sending to the edge function
                            # Edge function will store enhanced_prompts[i] to imageGenerations[i] (ordered by timeline_frame)
                            # This MUST match input_image_paths_resolved ordering!
                            travel_logger.debug(f"[EDGE_FUNC_DEBUG] Sending {len(complete_enhanced_prompts)} prompts to edge function:")
                            for i, prompt in enumerate(complete_enhanced_prompts):
                                img_name = Path(input_images_resolved[i]).name if i < len(input_images_resolved) else "NO_IMAGE"
                                prompt_preview = prompt[:60] if prompt else "EMPTY"
                                travel_logger.debug(f"[EDGE_FUNC_DEBUG]   [{i}] → {img_name} | '{prompt_preview}...'")
                            travel_logger.debug(f"[EDGE_FUNC_DEBUG] WARNING: If images above don't match timeline_frame order in shot_generations, prompts will be misaligned!")
                            travel_logger.debug("[VLM_BATCH] Edge function auth configured")

                            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)

                            if resp.status_code == 200:
                                travel_logger.debug(f"[VLM_BATCH] Successfully updated shot_generations via edge function")
                                resp_json = resp.json()
                                travel_logger.debug(f"[VLM_BATCH] Edge function response: {resp_json}")
                            else:
                                travel_logger.debug(f"[VLM_BATCH] WARNING: Edge function call failed: {resp.status_code} - {resp.text}")
                    else:
                        travel_logger.debug(
                            f"[VLM_BATCH] Skipping edge function call (generated={len(complete_enhanced_prompts)} prompts)"
                        )

                except (httpx.HTTPError, OSError, ValueError) as e_edge:
                    travel_logger.debug(f"[VLM_BATCH] WARNING: Failed to call edge function: {e_edge}", exc_info=True)
                    # Non-fatal - continue with task creation

            except (RuntimeError, ValueError, OSError) as e_vlm_batch:
                travel_logger.debug(f"[VLM_BATCH] ERROR during batch VLM processing: {e_vlm_batch}", exc_info=True)
                travel_logger.debug(f"[VLM_BATCH] Falling back to original prompts for all segments")
                vlm_enhanced_prompts = {}

        # Update orchestrator_payload with VLM enhanced prompts so they appear in debug output/DB
        if vlm_enhanced_prompts:
            enhanced_list = orchestrator_payload.get("enhanced_prompts_expanded", [])
            # Resize if needed (should be initialized to correct size but be safe)
            if len(enhanced_list) < num_segments:
                enhanced_list.extend([""] * (num_segments - len(enhanced_list)))
            
            # Fill in values
            for idx_prom, prompt in vlm_enhanced_prompts.items():
                if idx_prom < len(enhanced_list):
                    enhanced_list[idx_prom] = prompt
            
            orchestrator_payload["enhanced_prompts_expanded"] = enhanced_list
            travel_logger.debug(f"[VLM_ENHANCE] Updated orchestrator_payload enhanced_prompts_expanded with {len(vlm_enhanced_prompts)} prompts")

        # Loop to queue all segment tasks (skip existing ones for idempotency)
        segments_created = 0
        for idx in range(num_segments):
            # Get travel mode for dependency logic
            travel_mode = orchestrator_payload.get("model_type", "vace")
            chain_segments = orchestrator_payload.get("chain_segments", True)
            use_svi = orchestrator_payload.get("use_svi", False)

            # Determine dependency strictly from previously resolved actual DB IDs
            # SVI MODE: Sequential segments (each depends on previous for end frame chaining)
            # I2V MODE (non-SVI): Independent segments (no dependency on previous task)
            # VACE MODE: Sequential by default (chain_segments=True), independent if chain_segments=False

            if use_svi:
                # SVI mode: ALWAYS sequential - each segment needs previous output for start frame
                previous_segment_task_id = actual_segment_db_id_by_index.get(idx - 1) if idx > 0 else None
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} (SVI mode): Sequential dependency on previous segment: {previous_segment_task_id}")
            elif generation_policy.continuation.enabled:
                previous_segment_task_id = actual_segment_db_id_by_index.get(idx - 1) if idx > 0 else None
                travel_logger.debug(
                    f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} ({generation_policy.continuation.strategy} continuation): "
                    f"Sequential dependency on previous segment: {previous_segment_task_id}"
                )
            elif travel_mode == "i2v":
                previous_segment_task_id = None
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} (i2v mode): No dependency on previous segment")
            elif travel_mode == "vace" and not chain_segments:
                previous_segment_task_id = None
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Segment {idx} (vace independent mode): No dependency on previous segment")
            else:
                # VACE MODE (Sequential): Dependent on previous segment
                previous_segment_task_id = actual_segment_db_id_by_index.get(idx - 1) if idx > 0 else None

            # Defensive fallback for sequential modes (SVI or VACE with chaining)
            if use_svi or generation_policy.continuation.enabled:
                if idx > 0 and not previous_segment_task_id:
                    fallback_prev = existing_segment_task_ids.get(idx - 1)
                    if fallback_prev:
                        travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Fallback resolved previous DB ID for seg {idx-1} from existing_segment_task_ids: {fallback_prev}")
                        actual_segment_db_id_by_index[idx - 1] = fallback_prev
                        previous_segment_task_id = fallback_prev
                    else:
                        try:
                            child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
                            for seg in child_tasks.get('segments', []):
                                if seg.get('params', {}).get('segment_index') == idx - 1:
                                    prev_from_db = seg.get('id')
                                    if prev_from_db:
                                        travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] DB fallback resolved previous DB ID for seg {idx-1}: {prev_from_db}")
                                        actual_segment_db_id_by_index[idx - 1] = prev_from_db
                                        previous_segment_task_id = prev_from_db
                                    break
                        except (RuntimeError, ValueError, OSError) as e_depdb:
                            travel_logger.debug(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not resolve previous DB ID for seg {idx-1} via DB fallback: {e_depdb}")

            # Skip if this segment already exists
            if idx in existing_segment_indices:
                existing_db_id = existing_segment_task_ids[idx]
                travel_logger.debug(f"[IDEMPOTENCY] Skipping segment {idx} - already exists with ID {existing_db_id}")
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Using existing DB ID for segment {idx}: {existing_db_id}; next segment will depend on this")
                continue
                
            segments_created += 1
            
            # Note: segment handler now manages its own output paths using prepare_output_path()

            # Determine frame_overlap_from_previous for current segment `idx`
            current_frame_overlap_from_previous = 0
            if idx == 0 and orchestrator_payload.get("continue_from_video_resolved_path"):
                current_frame_overlap_from_previous = expanded_frame_overlap[0] if expanded_frame_overlap else 0
            elif idx > 0:
                # SM_RESTRUCTURE_FIX_OVERLAP_IDX: Use idx-1 for subsequent segments
                current_frame_overlap_from_previous = expanded_frame_overlap[idx-1] if len(expanded_frame_overlap) > (idx-1) else 0
            
            # VACE refs for this specific segment
            # Ensure vace_refs_instructions_all is a list, default to empty list if None
            vace_refs_safe = vace_refs_instructions_all if vace_refs_instructions_all is not None else []
            vace_refs_for_this_segment = [
                ref_instr for ref_instr in vace_refs_safe
                if ref_instr.get("segment_idx_for_naming") == idx
            ]

            # [DEEP_DEBUG] Log orchestrator payload values BEFORE creating segment payload
            travel_logger.debug(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: CREATING SEGMENT {idx} PAYLOAD")
            
            # Use centralized extraction to get all parameters that should be at top level
            from ...utils import extract_orchestrator_parameters
            
            # Extract parameters using centralized function
            task_params_for_extraction = {
                "orchestrator_details": orchestrator_payload
            }
            extracted_params = extract_orchestrator_parameters(
                task_params_for_extraction,
                task_id=f"seg_{idx}_{orchestrator_task_id_str[:8]}")

            # VLM-enhanced prompt retrieval
            # Prompts were pre-generated in batch processing (lines 845-924) for performance.
            # This avoids reloading the VLM model for each segment.
            segment_base_prompt = expanded_base_prompts[idx]
            prompt_source = "expanded_base_prompts"
            if idx in vlm_enhanced_prompts:
                segment_base_prompt = vlm_enhanced_prompts[idx]
                prompt_source = "vlm_enhanced_prompts"
                travel_logger.debug(f"[VLM_ENHANCE] Segment {idx}: Using pre-generated enhanced prompt")
            
            # [SEGMENT_PROMPT_DEBUG] Log the prompt assignment for each segment
            input_images = orchestrator_payload.get("input_image_paths_resolved", [])
            img_start_name = Path(input_images[idx]).name if idx < len(input_images) else "OUT_OF_BOUNDS"
            img_end_name = Path(input_images[idx + 1]).name if idx + 1 < len(input_images) else "OUT_OF_BOUNDS"
            travel_logger.debug(f"[SEGMENT_PROMPT_DEBUG] Segment {idx}: images {img_start_name} → {img_end_name}")
            travel_logger.debug(f"[SEGMENT_PROMPT_DEBUG]   Prompt source: {prompt_source}")
            travel_logger.debug(f"[SEGMENT_PROMPT_DEBUG]   Prompt: '{segment_base_prompt[:80]}...'" if segment_base_prompt else "[SEGMENT_PROMPT_DEBUG]   Prompt: EMPTY")
            
            # Fallback to orchestrator's base_prompt if segment prompt is empty
            if not segment_base_prompt or not segment_base_prompt.strip():
                segment_base_prompt = orchestrator_payload.get("base_prompt", "")
                if segment_base_prompt:
                    travel_logger.debug(f"[PROMPT_FALLBACK] Segment {idx}: Using orchestrator base_prompt (segment prompt was empty)")

            # Apply text_before_prompts and text_after_prompts wrapping (after enrichment)
            text_before = orchestrator_payload.get("text_before_prompts", "").strip()
            text_after = orchestrator_payload.get("text_after_prompts", "").strip()

            if text_before or text_after:
                # Build the wrapped prompt, ensuring clean spacing
                parts = []
                if text_before:
                    parts.append(text_before)
                parts.append(segment_base_prompt)
                if text_after:
                    parts.append(text_after)
                segment_base_prompt = " ".join(parts)
                travel_logger.debug(f"[TEXT_WRAP] Segment {idx}: Applied text_before/after wrapping")

            # Get negative prompt with fallback
            segment_negative_prompt = expanded_negative_prompts[idx] if idx < len(expanded_negative_prompts) else ""
            if not segment_negative_prompt or not segment_negative_prompt.strip():
                segment_negative_prompt = orchestrator_payload.get("negative_prompt", "")
                if segment_negative_prompt:
                    travel_logger.debug(f"[PROMPT_FALLBACK] Segment {idx}: Using orchestrator negative_prompt (segment negative_prompt was empty)")

            # Calculate segment_frames_target with context frames for segments after the first.
            # Context frames are ONLY needed for sequential VACE segments that continue from the
            # previous segment's VIDEO. For SVI chaining, we continue from the previous segment's
            # LAST FRAME as an image, so we should NOT inflate the frame count with overlap here.
            base_segment_frames = expanded_segment_frames[idx]
            if (
                idx > 0
                and current_frame_overlap_from_previous > 0
                and (not use_svi)
                and generation_policy.continuation.enabled
            ):
                # Sequential mode: add context frames for continuity with previous segment
                segment_frames_target_with_context = base_segment_frames + current_frame_overlap_from_previous
                travel_logger.debug(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, context={current_frame_overlap_from_previous}, total={segment_frames_target_with_context}")
            else:
                # First segment OR independent mode: no context frames needed
                segment_frames_target_with_context = base_segment_frames
                if use_svi and idx > 0:
                    travel_logger.debug(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, no context (SVI mode)")
                elif travel_mode != "vace" and idx > 0:
                    travel_logger.debug(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, no context ({travel_mode} mode)")
                elif not chain_segments and idx > 0:
                    travel_logger.debug(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, no context (independent mode)")
                else:
                    travel_logger.debug(f"[CONTEXT_FRAMES] Segment {idx}: base={base_segment_frames}, no context needed")
            
            # Ensure frame count is valid step*N+1 (VAE temporal quantization requirement)
            # Invalid counts cause mask/guide vs output frame count mismatches
            if (segment_frames_target_with_context - 1) % frame_step != 0:
                old_count = segment_frames_target_with_context
                if (
                    idx > 0
                    and current_frame_overlap_from_previous > 0
                    and (not use_svi)
                    and generation_policy.continuation.enabled
                ):
                    segment_frames_target_with_context = _quantize_frames_up(segment_frames_target_with_context, frame_step)
                else:
                    segment_frames_target_with_context = _quantize_frames(segment_frames_target_with_context, frame_step)
                travel_logger.debug(f"[FRAME_QUANTIZATION] Segment {idx}: {old_count} -> {segment_frames_target_with_context} (enforcing {frame_step}N+1 rule)")
            
            # Consolidated segment frame count log for easy debugging
            travel_logger.debug(f"[SEGMENT_FRAMES] Segment {idx}: FINAL frame target = {segment_frames_target_with_context} (valid {frame_step}N+1: ✓)")

            # Resolve start/end image URLs for this segment
            segment_start_image_url = input_images[idx] if idx < len(input_images) else None
            segment_end_image_url = input_images[idx + 1] if idx + 1 < len(input_images) else None

            segment_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id, # Added project_id
                # Start/end image URLs (required by create-task edge function resolver)
                "start_image_url": segment_start_image_url,
                **({"end_image_url": segment_end_image_url} if segment_end_image_url else {}),
                # Parent generation ID for linking to shot_generations
                "parent_generation_id": (
                    task_params_from_db.get("parent_generation_id")
                    or orchestrator_payload.get("parent_generation_id")
                    or orchestrator_payload.get("orchestrator_details", {}).get("parent_generation_id")
                ),
                "segment_index": idx,
                "is_first_segment": (idx == 0),
                "is_last_segment": (idx == num_segments - 1),
                # Standardized fields for completion handler
                "child_order": idx,
                "is_single_item": (num_segments == 1),

                "current_run_base_output_dir": str(current_run_output_dir.resolve()), # Base for segment's own output folder creation

                "base_prompt": segment_base_prompt,
                "negative_prompt": segment_negative_prompt,
                "segment_frames_target": segment_frames_target_with_context,
                "frame_overlap_from_previous": current_frame_overlap_from_previous,
                "frame_overlap_with_next": expanded_frame_overlap[idx] if len(expanded_frame_overlap) > idx else 0,
                
                "vace_image_refs_to_prepare_by_worker": vace_refs_for_this_segment, # Already filtered for this segment

                "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
                "model_name": orchestrator_payload["model_name"],
                "seed_to_use": orchestrator_payload.get("seed_base", DEFAULT_SEED_BASE),
                "cfg_star_switch": orchestrator_payload.get("cfg_star_switch", 0),
                "cfg_zero_step": orchestrator_payload.get("cfg_zero_step", -1),
                "params_json_str_override": orchestrator_payload.get("params_json_str_override"),
                "fps_helpers": orchestrator_payload.get("fps_helpers", 16),
                "subsequent_starting_strength_adjustment": orchestrator_payload.get("subsequent_starting_strength_adjustment", 0.0),
                "desaturate_subsequent_starting_frames": orchestrator_payload.get("desaturate_subsequent_starting_frames", 0.0),
                "adjust_brightness_subsequent_starting_frames": orchestrator_payload.get("adjust_brightness_subsequent_starting_frames", 0.0),
                "after_first_post_generation_saturation": orchestrator_payload.get("after_first_post_generation_saturation"),
                "after_first_post_generation_brightness": orchestrator_payload.get("after_first_post_generation_brightness"),
                
                "segment_image_download_dir": segment_image_download_dir_str, # Add the download dir path string
                
                "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
                "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False),
                "continue_from_video_resolved_path_for_guide": orchestrator_payload.get("continue_from_video_resolved_path") if idx == 0 else None,
                "consolidated_end_anchor_idx": orchestrator_payload.get("_consolidated_end_anchors", [None] * num_segments)[idx] if orchestrator_payload.get("_consolidated_end_anchors") else None,
                "consolidated_keyframe_positions": orchestrator_payload.get("_consolidated_keyframe_positions", [None] * num_segments)[idx] if orchestrator_payload.get("_consolidated_end_anchors") else None,
                "orchestrator_details": orchestrator_payload, # Canonical name for full orchestrator payload
                
                # SVI (Stable Video Infinity) end frame chaining
                "use_svi": use_svi,
            }

            # =============================================================================
            # Travel / Structure Guidance
            # =============================================================================
            segment_frame_offset = (
                segment_stitched_offsets[idx]
                if (
                    use_stitched_offsets
                    or (using_travel_guidance and travel_guidance_config.is_ltx_hybrid)
                )
                else (segment_flow_offsets[idx] if idx < len(segment_flow_offsets) else 0)
            )

            segment_has_guidance = False
            segment_travel_guidance_config = travel_guidance_config
            if using_travel_guidance:
                if travel_guidance_config.is_ltx_hybrid:
                    segment_travel_guidance_config = _build_segment_hybrid_guidance_config(
                        config=travel_guidance_config,
                        segment_index=idx,
                        segment_frames_expanded=expanded_segment_frames,
                        segment_stitched_offsets=segment_stitched_offsets,
                        total_stitched_frames=total_stitched_frames,
                    )
                    segment_has_guidance = segment_travel_guidance_config.kind != "none"
                else:
                    segment_has_guidance = _segment_has_travel_guidance_overlap(
                        segment_index=idx,
                        segment_frames_expanded=expanded_segment_frames,
                        segment_stitched_offsets=segment_stitched_offsets,
                        total_stitched_frames=total_stitched_frames,
                        travel_guidance_config=travel_guidance_config,
                    )
                if not segment_has_guidance and travel_guidance_config.kind != "none":
                    travel_logger.debug(
                        f"[TRAVEL_GUIDANCE] Segment {idx}: no overlap on stitched timeline, sending kind=none"
                    )
            elif structure_config is not None:
                segment_has_guidance = structure_config.has_guidance
                if structure_videos and segment_has_guidance:
                    from source.media.structure import segment_has_structure_overlap

                    segment_has_guidance = segment_has_structure_overlap(
                        segment_index=idx,
                        segment_frames_expanded=expanded_segment_frames,
                        frame_overlap_expanded=expanded_frame_overlap,
                        structure_videos=structure_videos,
                    )
                    if not segment_has_guidance:
                        travel_logger.debug(
                            f"[STRUCTURE_VIDEO] Segment {idx}: No overlap with structure_videos, skipping structure guidance"
                        )

            segment_payload["travel_guidance"] = _build_segment_travel_guidance_payload(
                segment_travel_guidance_config,
                frame_offset=segment_frame_offset,
                has_guidance=segment_has_guidance,
            )

            if segment_has_guidance and structure_config is not None:
                segment_payload["structure_guidance"] = structure_config.to_segment_payload(
                    frame_offset=segment_frame_offset
                )["structure_guidance"]
            else:
                segment_payload["structure_guidance"] = None

            # =============================================================================
            # Per-Segment Parameter Overrides
            # =============================================================================
            # Build individual_segment_params dict with per-segment overrides
            individual_segment_params = {}

            # Add per-segment phase_config if available
            if idx < len(phase_configs_expanded) and phase_configs_expanded[idx] is not None:
                individual_segment_params["phase_config"] = phase_configs_expanded[idx]
                travel_logger.debug(f"[PER_SEGMENT_PARAMS] Segment {idx}: Using per-segment phase_config override")

            # Add per-segment LoRAs if available
            segment_loras_for_payload = None
            if idx < len(loras_per_segment_expanded) and loras_per_segment_expanded[idx] is not None:
                segment_loras_for_payload = list(loras_per_segment_expanded[idx])
                travel_logger.debug(
                    f"[PER_SEGMENT_PARAMS] Segment {idx}: Using per-segment LoRA override "
                    f"({len(segment_loras_for_payload)} LoRAs)"
                )

            segment_loras_for_payload = _auto_inject_travel_guidance_lora(
                segment_loras_for_payload,
                (
                    segment_travel_guidance_config
                    if segment_payload["travel_guidance"].get("kind") != "none"
                    else None
                ),
            )
            if segment_loras_for_payload:
                individual_segment_params["segment_loras"] = segment_loras_for_payload
                travel_logger.debug(
                    f"[PER_SEGMENT_PARAMS] Segment {idx}: Final segment_loras count = "
                    f"{len(segment_loras_for_payload)}"
                )

            # Only add individual_segment_params if it has content
            if individual_segment_params:
                segment_payload["individual_segment_params"] = individual_segment_params
                travel_logger.debug(f"[PER_SEGMENT_PARAMS] Segment {idx}: Added individual_segment_params with keys: {list(individual_segment_params.keys())}")

            # Add extracted parameters at top level for queue processing
            segment_payload.update(extracted_params)
            
            # SVI-specific configuration: merge SVI LoRAs and set parameters
            # IMPORTANT: First segment (idx == 0) does NOT use SVI mode - it generates normally.
            # Only subsequent segments use SVI end frame chaining from the previous segment's output.
            if use_svi and idx > 0:
                travel_logger.debug(f"[SVI_CONFIG] Segment {idx}: Configuring SVI mode")

                # Force SVI generation parameters (SVI LoRAs are 2-phase and expect these defaults)
                for key, value in SVI_DEFAULT_PARAMS.items():
                    prev_val = segment_payload.get(key, None)
                    if prev_val != value:
                        segment_payload[key] = value
                        travel_logger.debug(f"[SVI_CONFIG] Segment {idx}: Set {key}={value} (was {prev_val})")

                # SVI requires svi2pro=True for encoding mode
                segment_payload["svi2pro"] = True

                # SVI requires video_prompt_type="I" to enable image_refs
                segment_payload["video_prompt_type"] = "I"

                # For SVI, use smaller overlap since end/start frames should mostly match
                segment_payload["frame_overlap_with_next"] = SVI_STITCH_OVERLAP if idx < (num_segments - 1) else 0
                segment_payload["frame_overlap_from_previous"] = SVI_STITCH_OVERLAP if idx > 0 else 0
                travel_logger.debug(
                    f"[SVI_CONFIG] Segment {idx}: "
                    f"frame_overlap_from_previous={segment_payload['frame_overlap_from_previous']} "
                    f"frame_overlap_with_next={segment_payload['frame_overlap_with_next']} (SVI mode)"
                )
            elif use_svi and idx == 0:
                # First segment: disable SVI mode - generate normally from start image
                segment_payload["use_svi"] = False
                segment_payload["svi2pro"] = False
                travel_logger.debug(f"[SVI_CONFIG] Segment {idx}: First segment - SVI disabled (use_svi=False, svi2pro=False)")

            # [DEEP_DEBUG] Log segment payload values AFTER creation
            travel_logger.debug(f"[ORCHESTRATOR_DEBUG] {orchestrator_task_id_str}: SEGMENT {idx} PAYLOAD CREATED")
            travel_logger.debug(f"[DEEP_DEBUG] Segment payload keys: {list(segment_payload.keys())}")

            # === CANCELLATION CHECK: Abort if orchestrator was cancelled ===
            orchestrator_current_status = db_ops.get_task_current_status(orchestrator_task_id_str)
            if orchestrator_current_status and orchestrator_current_status.lower() in ('cancelled', 'canceled'):
                travel_logger.debug(f"[CANCELLATION] Orchestrator {orchestrator_task_id_str} was cancelled - aborting segment creation at index {idx}")
                travel_logger.essential(f"Orchestrator cancelled, stopping segment creation at segment {idx}", task_id=orchestrator_task_id_str)
                # Cancel any child tasks that were already created in earlier iterations
                db_ops.cancel_orchestrator_children(orchestrator_task_id_str, reason="Orchestrator cancelled by user")
                return TaskResult.failed(f"Orchestrator cancelled before segment {idx} could be created ({segments_created} segments were already created and have been cancelled)")

            travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Creating new segment {idx}, depends_on (prev idx {idx-1}): {previous_segment_task_id}")
            travel_logger.essential(f"Creating segment {idx} task...", task_id=orchestrator_task_id_str)
            actual_db_row_id = db_ops.add_task_to_db(
                task_payload=segment_payload, 
                task_type_str="travel_segment",
                dependant_on=previous_segment_task_id
            )
            # Record the actual DB ID so subsequent segments depend on the real DB row ID
            actual_segment_db_id_by_index[idx] = actual_db_row_id
            travel_logger.essential(f"Segment {idx} created: task_id={actual_db_row_id}", task_id=orchestrator_task_id_str)
            travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] New segment {idx} created with actual DB ID: {actual_db_row_id}; next segment will depend on this")
            # Post-insert verification of dependency from DB
            try:
                dep_saved = db_ops.get_task_dependency(actual_db_row_id)
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN][VERIFY] Segment {idx} saved dependant_on={dep_saved} (expected {previous_segment_task_id})")
                travel_logger.debug(f"Segment {idx} dependency verified: dependant_on={dep_saved}", task_id=orchestrator_task_id_str)
            except (RuntimeError, ValueError, OSError) as e_ver:
                travel_logger.debug(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on for seg {idx} ({actual_db_row_id}): {e_ver}")
                travel_logger.warning(f"Segment {idx} dependency verification failed (likely replication lag): {e_ver}", task_id=orchestrator_task_id_str)
        
        # After loop, enqueue the stitch task (check for idempotency)
        # SKIP if independent segments or non-SVI I2V mode
        # SVI mode REQUIRES stitching since segments are chained sequentially
        stitch_created = 0
        
        # Determine if we should create a stitch task
        should_create_stitch = False
        if use_svi:
            # SVI mode: Always create stitch task (segments are sequential with end frame chaining)
            should_create_stitch = True
            # For SVI, use the small overlap value
            stitch_overlap_settings = [SVI_STITCH_OVERLAP] * (num_segments - 1) if num_segments > 1 else []
            travel_logger.debug(f"[STITCHING] SVI mode: Creating stitch task with overlap={SVI_STITCH_OVERLAP}")
        elif generation_policy.continuation.enabled:
            # Normalized continuation mode: Create stitch task with configured overlaps
            should_create_stitch = True
            stitch_overlap_settings = expanded_frame_overlap
            travel_logger.debug(
                f"[STITCHING] {generation_policy.continuation.strategy}: "
                f"Creating stitch task with overlaps={expanded_frame_overlap}"
            )
        else:
            travel_logger.debug(f"[STITCHING] Skipping stitch task creation (mode={travel_mode}, chain_segments={chain_segments}, use_svi={use_svi})")
        
        if should_create_stitch and not stitch_already_exists:
            final_stitched_video_name = f"travel_final_stitched_{run_id}.mp4"
            # Stitcher saves its final primary output directly under main_output_dir (e.g., ./steerable_motion_output/)
            # NOT under current_run_output_dir (which is .../travel_run_XYZ/)
            final_stitched_output_path = Path(orchestrator_payload.get("main_output_dir_for_run", str(main_output_dir_base.resolve()))) / final_stitched_video_name

            stitch_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id,
                # IMPORTANT: generation.ts / complete_task expects this at the TOP LEVEL for travel_stitch.
                # (Some older payloads also include it under orchestrator_details; we keep full_orchestrator_payload
                # for backward compatibility, but top-level is the correct contract.)
                "parent_generation_id": (
                    task_params_from_db.get("parent_generation_id")
                    or orchestrator_payload.get("parent_generation_id")
                    or orchestrator_payload.get("orchestrator_details", {}).get("parent_generation_id")
                ),
                "num_total_segments_generated": num_segments,
                "current_run_base_output_dir": str(current_run_output_dir.resolve()),
                "frame_overlap_settings_expanded": stitch_overlap_settings,  # Use mode-specific overlap
                "crossfade_sharp_amt": orchestrator_payload.get("crossfade_sharp_amt", 0.3),
                "parsed_resolution_wh": orchestrator_payload["parsed_resolution_wh"],
                "fps_final_video": orchestrator_payload.get("fps_helpers", 16),
                "upscale_factor": orchestrator_payload.get("upscale_factor", 0.0),
                "upscale_model_name": orchestrator_payload.get("upscale_model_name"),
                "seed_for_upscale": orchestrator_payload.get("seed_base", DEFAULT_SEED_BASE) + UPSCALE_SEED_OFFSET,
                "debug_mode_enabled": orchestrator_payload.get("debug_mode_enabled", False),
                "skip_cleanup_enabled": orchestrator_payload.get("skip_cleanup_enabled", False),
                "initial_continued_video_path": orchestrator_payload.get("continue_from_video_resolved_path"),
                "final_stitched_output_path": str(final_stitched_output_path.resolve()),
                "poll_interval_from_orchestrator": orchestrator_payload.get("original_common_args", {}).get("poll_interval", 15),
                "poll_timeout_from_orchestrator": orchestrator_payload.get("original_common_args", {}).get("poll_timeout", 1800),
                "orchestrator_details": orchestrator_payload,  # Canonical name
                "use_svi": use_svi,  # Pass SVI flag to stitch task
            }
            
            # Stitch should depend on the last segment's actual DB row ID
            last_segment_task_id = actual_segment_db_id_by_index.get(num_segments - 1)
            travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Creating stitch task, depends_on (last seg idx {num_segments-1}): {last_segment_task_id}")
            actual_stitch_db_row_id = db_ops.add_task_to_db(
                task_payload=stitch_payload, 
                task_type_str="travel_stitch",
                dependant_on=last_segment_task_id
            )
            travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN] Stitch task created with actual DB ID: {actual_stitch_db_row_id}")
            
            # Post-insert verification of dependency from DB
            try:
                dep_saved = db_ops.get_task_dependency(actual_stitch_db_row_id)
                travel_logger.debug(f"[DEBUG_DEPENDENCY_CHAIN][VERIFY] Stitch saved dependant_on={dep_saved} (expected {last_segment_task_id})")
                travel_logger.essential(f"Stitch task created: task_id={actual_stitch_db_row_id}", task_id=orchestrator_task_id_str)
            except (RuntimeError, ValueError, OSError) as e_ver2:
                travel_logger.debug(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on for stitch ({actual_stitch_db_row_id}): {e_ver2}")
            stitch_created = 1
        elif stitch_already_exists:
            travel_logger.debug(f"[IDEMPOTENCY] Skipping stitch task creation - already exists with ID {existing_stitch_task_id}")

        # === JOIN CLIPS ORCHESTRATOR (for AI-generated transitions) ===
        # If stitch_config is provided, create a join_clips_orchestrator that will generate
        # smooth AI transitions between segments using VACE, instead of simple crossfades
        stitch_config = orchestrator_payload.get("stitch_config")
        join_orchestrator_created = False

        # Check if join orchestrator already exists (idempotency)
        existing_join_orchestrators = existing_child_tasks.get('join_clips_orchestrator', [])
        join_orchestrator_already_exists = len(existing_join_orchestrators) > 0

        if stitch_config and not join_orchestrator_already_exists:
            travel_logger.debug(f"[JOIN_STITCH] stitch_config detected - creating join_clips_orchestrator for AI transitions")
            travel_logger.debug(f"[JOIN_STITCH] stitch_config: {stitch_config}")

            # Collect ALL segment task IDs for multi-dependency
            all_segment_task_ids = [actual_segment_db_id_by_index[i] for i in range(num_segments)]
            travel_logger.debug(f"[JOIN_STITCH] All segment task IDs ({len(all_segment_task_ids)}): {all_segment_task_ids}")

            # Build join_clips_orchestrator payload from stitch_config
            join_orchestrator_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id,
                "parent_generation_id": (
                    task_params_from_db.get("parent_generation_id")
                    or orchestrator_payload.get("parent_generation_id")
                ),

                # Dynamic clip_list will be built from segment outputs when join orchestrator runs
                "segment_task_ids": all_segment_task_ids,

                # Join settings from stitch_config
                "context_frame_count": stitch_config.get("context_frame_count", 12),
                "gap_frame_count": stitch_config.get("gap_frame_count", 19),
                "replace_mode": stitch_config.get("replace_mode", True),
                "prompt": stitch_config.get("prompt", "smooth seamless transition"),
                "negative_prompt": stitch_config.get("negative_prompt", ""),
                "enhance_prompt": stitch_config.get("enhance_prompt", False),
                "keep_bridging_images": stitch_config.get("keep_bridging_images", False),

                # Model and generation settings
                "model": stitch_config.get("model", orchestrator_payload.get("model", "wan_2_2_vace_lightning_baseline_2_2_2")),
                "phase_config": stitch_config.get("phase_config", orchestrator_payload.get("phase_config")),
                "additional_loras": stitch_config.get("loras", []),
                "seed": -1 if stitch_config.get("random_seed", True) else stitch_config.get("seed", orchestrator_payload.get("seed_base", -1)),

                # Resolution/FPS from original orchestrator
                "resolution": orchestrator_payload.get("parsed_resolution_wh"),
                "fps": orchestrator_payload.get("fps_helpers", 16),
                "use_input_video_resolution": True,
                "use_input_video_fps": True,

                # Audio (if provided)
                "audio_url": orchestrator_payload.get("audio_url"),

                # Output configuration
                "output_base_dir": str(current_run_output_dir.resolve()),

                # Use parallel join pattern (better quality)
                "use_parallel_joins": True,
            }

            # Create join_clips_orchestrator with multi-dependency on ALL segments
            travel_logger.debug(f"[JOIN_STITCH] Creating join_clips_orchestrator dependent on {len(all_segment_task_ids)} segments")
            join_orchestrator_task_id = db_ops.add_task_to_db(
                task_payload={"orchestrator_details": join_orchestrator_payload},
                task_type_str="join_clips_orchestrator",
                dependant_on=all_segment_task_ids  # Multi-dependency: all segments must complete
            )
            travel_logger.debug(f"[JOIN_STITCH] ✅ join_clips_orchestrator created: {join_orchestrator_task_id}")
            travel_logger.info(f"Created join_clips_orchestrator {join_orchestrator_task_id} for AI-stitching {num_segments} segments", task_id=orchestrator_task_id_str)
            join_orchestrator_created = True

        elif stitch_config and join_orchestrator_already_exists:
            travel_logger.debug(f"[IDEMPOTENCY] Skipping join_clips_orchestrator creation - already exists")

        if segments_created > 0:
            extra_info = ""
            if join_orchestrator_created:
                extra_info = " + join_clips_orchestrator for AI transitions"
            elif stitch_created:
                extra_info = " + travel_stitch task"
            msg = f"Successfully enqueued {segments_created} new segment tasks for run {run_id}{extra_info}. (Total expected: {num_segments} segments)"
        else:
            msg = f"All child tasks already exist for run {run_id}. No new tasks created."
        travel_logger.info(msg, task_id=orchestrator_task_id_str)
        log_ram_usage("Orchestrator end (success)", task_id=orchestrator_task_id_str)
        return TaskResult.orchestrating(msg)

    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e:
        msg = f"Failed during travel orchestration processing: {e}"
        travel_logger.error(msg, task_id=orchestrator_task_id_str, exc_info=True)
        log_ram_usage("Orchestrator end (error)", task_id=orchestrator_task_id_str)
        return TaskResult.failed(msg)
