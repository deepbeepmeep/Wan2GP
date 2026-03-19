"""
Task Registry Module

This module defines the TaskRegistry class and the TASK_HANDLERS dictionary.
It allows for a cleaner way to route tasks to their appropriate handlers
instead of using a massive if/elif block.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import time
import json
import uuid

if TYPE_CHECKING:
    from source.core.params.contracts import TaskDispatchContext
    from source.task_handlers.queue.task_queue import HeadlessTaskQueue

from source.core.log import headless_logger, task_logger
from source.task_handlers.worker.worker_utils import log_ram_usage
from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
from source.core.params.phase_config_parser import parse_phase_config
from source.core.params.structure_guidance import StructureGuidanceConfig
from source.core.params.travel_guidance import TravelGuidanceConfig
from source.task_handlers.contracts.dispatch import normalize_task_dispatch_payload
from source.task_handlers.tasks.travel_segment_types import IndividualSegmentParams
from source.core.params.contracts import validate_orchestrator_details

# Import task handlers
# These imports should be available from the environment where this module is used
from source.task_handlers.extract_frame import handle_extract_frame_task
from source.task_handlers.rife_interpolate import handle_rife_interpolate_task
from source.models.comfy.comfy_handler import handle_comfy_task
from source.task_handlers.travel import orchestrator as travel_orchestrator
from source.task_handlers.travel.stitch import handle_travel_stitch_task
from source.task_handlers import magic_edit as me
from source.task_handlers.join.generation import handle_join_clips_task
from source.task_handlers.join.final_stitch import handle_join_final_stitch
from source.task_handlers.join.orchestrator import handle_join_clips_orchestrator_task
from source.task_handlers.edit_video_orchestrator import handle_edit_video_orchestrator_task
from source.task_handlers.inpaint_frames import handle_inpaint_frames_task
from source.task_handlers.create_visualization import handle_create_visualization_task
from source.task_handlers.travel.segment_processor import TravelSegmentProcessor, TravelSegmentContext
from source.media.video import extract_last_frame_as_image
from source.core.db.dependencies.task_dependencies_queries import get_predecessor_output_via_edge_function
from source.core.db.task_polling import get_task_params
from source.utils.download_utils import download_file, download_image_if_url
from source.utils.prompt_utils import ensure_valid_negative_prompt, ensure_valid_prompt
from source.utils.resolution_utils import (
    get_model_grid_size,
    parse_resolution,
    snap_resolution_to_model_grid,
)

# Import centralized task type definitions
from source.task_handlers.tasks.task_types import DIRECT_QUEUE_TASK_TYPES
from source.task_handlers.tasks.dispatch_manifest import HANDLER_IMPORT_SPECS as _HANDLER_SPECS


# ─── Coordination dataclasses for handle_travel_segment_via_queue ────────────

@dataclass
class SegmentContext:
    """Mode, orchestrator details, and segment identity."""
    mode: str  # "orchestrator" or "standalone"
    orchestrator_details: dict
    individual_params: dict
    segment_idx: int
    segment_params: dict  # alias for the top-level task_params_dict
    orchestrator_task_id_ref: Optional[str] = None
    orchestrator_run_id: Optional[str] = None

@dataclass
class GenerationInputs:
    """Model, prompt, resolution, frames, and flags."""
    model_name: str
    prompt_for_wgp: str
    negative_prompt_for_wgp: str
    parsed_res_wh: Tuple[int, int]
    total_frames_for_segment: int
    current_run_base_output_dir: Path
    segment_processing_dir: Path
    debug_enabled: bool
    travel_mode: str

@dataclass
class ImageRefs:
    """Image reference paths for the segment."""
    start_ref_path: Optional[str] = None
    end_ref_path: Optional[str] = None
    svi_predecessor_video_for_source: Optional[str] = None
    use_svi: bool = False
    is_continuing: bool = False

@dataclass
class StructureOutputs:
    """Results from structure guidance processing."""
    guide_video_path: Optional[str] = None
    mask_video_path_for_wgp: Optional[Path] = None
    video_prompt_type_str: Optional[str] = None
    structure_config: Optional[StructureGuidanceConfig | TravelGuidanceConfig] = None
    image_refs_paths: Optional[list[str]] = None
    frames_positions: Optional[str] = None
    image_refs_strengths: Optional[list[float]] = None
    audio_guide: Optional[str] = None
    audio_prompt_type: Optional[str] = None
    audio_scale: Optional[float] = None
    denoising_strength: Optional[float] = None


_MISSING = object()

_TRAVEL_SEGMENT_HANDLER_SPECS = {
    "travel_segment": (
        "source.task_handlers.travel.segments.segment_queue",
        "handle_travel_segment_via_queue",
    ),
    "individual_travel_segment": (
        "source.task_handlers.travel.segments.segment_queue",
        "handle_travel_segment_via_queue",
    ),
}


def _parse_handler_spec(spec) -> tuple[str, str]:
    if isinstance(spec, tuple) and len(spec) == 2:
        return spec
    if isinstance(spec, str) and "." in spec:
        module_path, attr_name = spec.rsplit(".", 1)
        return module_path, attr_name
    raise ValueError(f"Invalid handler import spec: {spec!r}")


@lru_cache(maxsize=None)
def _load_handler_callable(task_type: str):
    module_path, attr_name = _parse_handler_spec(_HANDLER_SPECS[task_type])
    module = import_module(module_path)
    return getattr(module, attr_name)

def _get_param(key, *sources, default=_MISSING, prefer_truthy: bool = False):
    """
    Get a parameter from multiple sources with precedence (first source wins).

    By default, only skips None values.
    If prefer_truthy=True, also skips falsy non-bool values (e.g. "", {}, [], 0),
    matching historical `a or b` fallback semantics while still preserving explicit booleans.
    """
    for source in sources:
        if not source or key not in source:
            continue

        value = source[key]
        if value is None:
            continue

        if prefer_truthy and not isinstance(value, bool) and not value:
            continue

        return value

    return None if default is _MISSING else default

def _resolve_segment_context(task_params_dict: dict, is_standalone: bool, task_id: str) -> SegmentContext:
    """Resolve mode, orchestrator_details, individual_params, and segment identity.

    Handles orchestrator vs standalone mode, fetching orchestrator_details from the DB
    if not provided inline.

    Raises ValueError if required fields are missing.
    """
    segment_params = task_params_dict
    orchestrator_task_id_ref = segment_params.get("orchestrator_task_id_ref")
    orchestrator_run_id = segment_params.get("orchestrator_run_id")
    segment_idx = segment_params.get("segment_index")

    # Get orchestrator_details (canonical name) or full_orchestrator_payload (legacy alias)
    orchestrator_details: dict = segment_params.get("orchestrator_details") or segment_params.get("full_orchestrator_payload")

    if is_standalone:
        # Standalone mode: use provided payload, default segment_idx to 0
        if segment_idx is None:
            segment_idx = 0
        if not orchestrator_details:
            raise ValueError(f"Individual travel segment {task_id} missing orchestrator_details")
        headless_logger.debug(f"Running in standalone mode (individual_travel_segment)", task_id=task_id)
    else:
        # Orchestrator mode: require segment_index and either inline orchestrator_details or ability to fetch
        if segment_idx is None:
            raise ValueError(f"Travel segment {task_id} missing segment_index")

        # If orchestrator_details not inline, try to fetch from parent task
        if not orchestrator_details:
            if not orchestrator_task_id_ref:
                raise ValueError(f"Travel segment {task_id} missing orchestrator_details and orchestrator_task_id_ref")
            orchestrator_task_raw_params_json = get_task_params(orchestrator_task_id_ref)
            if orchestrator_task_raw_params_json:
                fetched_params = json.loads(orchestrator_task_raw_params_json) if isinstance(orchestrator_task_raw_params_json, str) else orchestrator_task_raw_params_json
                orchestrator_details = fetched_params.get("orchestrator_details")

        if not orchestrator_details:
            raise ValueError(f"Travel segment {task_id}: Could not retrieve orchestrator_details")
        validate_orchestrator_details(
            orchestrator_details,
            context="travel_segment",
            task_id=task_id,
        )

    # individual_segment_params has highest priority for segment-specific overrides
    individual_params: IndividualSegmentParams = segment_params.get("individual_segment_params", {})
    orchestrator_details = MappingProxyType(dict(orchestrator_details))

    return SegmentContext(
        mode="standalone" if is_standalone else "orchestrator",
        orchestrator_details=orchestrator_details,
        individual_params=individual_params,
        segment_idx=segment_idx,
        segment_params=segment_params,
        orchestrator_task_id_ref=orchestrator_task_id_ref,
        orchestrator_run_id=orchestrator_run_id,
    )


def _resolve_generation_inputs(ctx: SegmentContext, task_id: str, main_output_dir_base: Path) -> GenerationInputs:
    """Resolve model, prompt, resolution, frame count, output directory, and flags.

    Returns a GenerationInputs dataclass with all resolved values.
    Raises ValueError if resolution format is invalid.
    """
    segment_params = ctx.segment_params
    orchestrator_details = ctx.orchestrator_details
    individual_params = ctx.individual_params
    segment_idx = ctx.segment_idx

    # Model name: segment > orchestrator (required)
    model_name = segment_params.get("model_name") or orchestrator_details.get("model_name")
    if not model_name:
        raise ValueError(f"Travel segment {task_id}: model_name missing from both segment_params and orchestrator_details")

    # Prompts: enhanced_prompt preferred > base_prompt fallback
    # Frontend may provide both: enhanced_prompt (AI-enhanced) and base_prompt (user's original)
    enhanced_prompt = _get_param("enhanced_prompt", individual_params, segment_params, default=None)
    base_prompt = _get_param("base_prompt", individual_params, segment_params, default=" ")

    # Use enhanced_prompt if present and non-empty, otherwise base_prompt
    effective_prompt = enhanced_prompt if enhanced_prompt and enhanced_prompt.strip() else base_prompt
    prompt_for_wgp = ensure_valid_prompt(effective_prompt)

    if enhanced_prompt and enhanced_prompt.strip():
        task_logger.debug(f"[PROMPT] Task {task_id}: Using enhanced_prompt")
    negative_prompt_for_wgp = ensure_valid_negative_prompt(
        _get_param("negative_prompt", individual_params, segment_params, default=" ")
    )

    # Resolution: segment > orchestrator
    parsed_res_wh_str = segment_params.get("parsed_resolution_wh") or orchestrator_details.get("parsed_resolution_wh")
    if not parsed_res_wh_str:
        raise ValueError(f"Travel segment {task_id}: parsed_resolution_wh missing from both segment_params and orchestrator_details")
    parsed_res_raw = parse_resolution(parsed_res_wh_str)
    if parsed_res_raw is None:
        raise ValueError(f"Travel segment {task_id}: Invalid resolution format {parsed_res_wh_str}")
    model_name = segment_params.get("model_name") or orchestrator_details.get("model_name")
    grid_size = get_model_grid_size(model_name)
    parsed_res_wh = snap_resolution_to_model_grid(parsed_res_raw, grid_size)

    # Frame count: individual_segment_params.num_frames > top-level num_frames > segment_frames_target > segment_frames_expanded[idx]
    total_frames_for_segment = (
        individual_params.get("num_frames") or
        segment_params.get("num_frames") or
        segment_params.get("segment_frames_target")
    )
    if not total_frames_for_segment:
        segment_frames = orchestrator_details.get("segment_frames_expanded", [])
        if segment_idx < len(segment_frames):
            total_frames_for_segment = segment_frames[segment_idx]
        else:
            raise ValueError(
                f"Travel segment {task_id}: no frame count found (segment_idx={segment_idx}, "
                f"segment_frames_expanded has {len(segment_frames)} entries)"
            )
    try:
        total_frames_for_segment = int(total_frames_for_segment)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Travel segment {task_id}: invalid frame count {total_frames_for_segment!r}"
        ) from exc
    if total_frames_for_segment <= 0:
        raise ValueError(
            f"Travel segment {task_id}: invalid frame count {total_frames_for_segment!r}"
        )

    current_run_base_output_dir_str = _get_param(
        "current_run_base_output_dir",
        segment_params,
        default=orchestrator_details.get("main_output_dir_for_run", str(main_output_dir_base.resolve())),
        prefer_truthy=True)

    # Convert to Path and resolve relative paths against main_output_dir_base
    base_dir_path = Path(current_run_base_output_dir_str)
    if not base_dir_path.is_absolute():
        # Relative path - resolve against main_output_dir_base
        current_run_base_output_dir = main_output_dir_base / base_dir_path
    else:
        # Already absolute - use as is
        current_run_base_output_dir = base_dir_path

    segment_processing_dir = current_run_base_output_dir
    segment_processing_dir.mkdir(parents=True, exist_ok=True)

    debug_enabled = _get_param("debug_mode_enabled", segment_params, orchestrator_details, default=False)
    travel_mode = orchestrator_details.get("model_type", "vace")

    return GenerationInputs(
        model_name=model_name,
        prompt_for_wgp=prompt_for_wgp,
        negative_prompt_for_wgp=negative_prompt_for_wgp,
        parsed_res_wh=parsed_res_wh,
        total_frames_for_segment=total_frames_for_segment,
        current_run_base_output_dir=current_run_base_output_dir,
        segment_processing_dir=segment_processing_dir,
        debug_enabled=debug_enabled,
        travel_mode=travel_mode,
    )


def _resolve_image_references(ctx: SegmentContext, gen: GenerationInputs, task_id: str, is_standalone: bool) -> ImageRefs:
    """Resolve start/end image reference paths and SVI predecessor video.

    Handles SVI chaining (downloading predecessor video, extracting last frames),
    non-SVI image resolution from multiple sources, and downloading URLs to local paths.
    """
    segment_params = ctx.segment_params
    orchestrator_details = ctx.orchestrator_details
    individual_params = ctx.individual_params
    segment_idx = ctx.segment_idx
    segment_processing_dir = gen.segment_processing_dir
    debug_enabled = gen.debug_enabled

    start_ref_path = None
    end_ref_path = None

    # Image resolution priority:
    # 1. individual_segment_params.start_image_url / end_image_url
    # 2. individual_segment_params.input_image_paths_resolved (array)
    # 3. top-level input_image_paths_resolved
    # 4. orchestrator_details images (indexed by segment_idx)

    individual_images = individual_params.get("input_image_paths_resolved", [])
    top_level_images = segment_params.get("input_image_paths_resolved", [])
    orchestrator_images = orchestrator_details.get("input_image_paths_resolved", [])

    task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: segment_idx={segment_idx}, individual_images={len(individual_images)}, top_level_images={len(top_level_images)}, orchestrator_images={len(orchestrator_images)}")

    is_continuing = orchestrator_details.get("continue_from_video_resolved_path") is not None
    # Check use_svi with explicit False handling (False at segment level should override True at orchestrator)
    use_svi = segment_params["use_svi"] if "use_svi" in segment_params else orchestrator_details.get("use_svi", False)
    svi_predecessor_video_url = _get_param(
        "svi_predecessor_video_url", segment_params, orchestrator_details, prefer_truthy=True
    )

    if use_svi:
        task_logger.debug(f"[SVI_MODE] Task {task_id}: SVI mode enabled for segment {segment_idx}")

    # =============================================================================
    # SVI MODE: Chain segments using predecessor video for overlapped_latents
    # SVI Pro uses the last ~9 frames (5 + sliding_window_overlap) from predecessor
    # to create temporal continuity via overlapped_latents in the VAE
    # =============================================================================
    svi_predecessor_video_for_source = None  # Will be set for SVI continuation

    if use_svi and (segment_idx > 0 or svi_predecessor_video_url):
        task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Using SVI end frame chaining mode")

        predecessor_output_url = None

        # Priority 1: Manually specified predecessor video URL
        if svi_predecessor_video_url:
            task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Using manually specified predecessor video: {svi_predecessor_video_url}")
            predecessor_output_url = svi_predecessor_video_url
        # Priority 2: Fetch from dependency chain (for segment_idx > 0)
        elif segment_idx > 0:
            task_dependency_id, predecessor_output_url = get_predecessor_output_via_edge_function(task_id)
            if task_dependency_id and predecessor_output_url:
                task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Found predecessor {task_dependency_id} with output: {predecessor_output_url}")
            else:
                task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: ERROR - Could not fetch predecessor output (dep_id={task_dependency_id})")

        if predecessor_output_url:
            # Download predecessor video if it's a URL
            predecessor_video_path = predecessor_output_url
            if predecessor_output_url.startswith("http"):
                try:
                    local_filename = Path(predecessor_output_url).name
                    local_download_path = segment_processing_dir / f"svi_predecessor_{segment_idx:02d}_{local_filename}"

                    if not local_download_path.exists():
                        download_file(predecessor_output_url, segment_processing_dir, local_download_path.name)
                        task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Downloaded predecessor video to {local_download_path}")
                    else:
                        task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Predecessor video already exists at {local_download_path}")

                    predecessor_video_path = str(local_download_path)
                except (OSError, ValueError, RuntimeError) as e_dl:
                    task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Failed to download predecessor video: {e_dl}")
                    predecessor_video_path = None

            # SVI CRITICAL: Extract only the last ~9 frames (5 + overlap_size) from predecessor
            # WGP uses prefix_video[:, -(5 + overlap_size):] to create overlapped_latents,
            # but then prepends the ENTIRE prefix_video to output. By extracting only the
            # last frames ourselves, we limit what gets prepended.
            if predecessor_video_path and Path(predecessor_video_path).exists():
                from source.media.video import (
                    get_video_frame_count_and_fps as get_video_frame_count_and_fps,
                    extract_frame_range_to_video as extract_frame_range_to_video
                )

                # Get predecessor video frame count
                pred_frames, pred_fps = get_video_frame_count_and_fps(predecessor_video_path)
                if pred_frames and pred_frames > 0:
                    # IMPORTANT GROUND TRUTH:
                    # Wan2GP SVI continuation uses the last (5 + overlap_size) frames of prefix_video
                    # to build `overlapped_latents`. The extra 5 frames are *model context*; the
                    # overlap_size (4) is the actual stitch overlap we keep in the final segment.
                    #
                    # So we intentionally extract MORE than 4 frames here (usually 9), but after WGP
                    # generation we trim the output so only the last 4 prefix frames remain as the
                    # overlap with the previous segment.
                    prefix_min_context = 5
                    overlap_size = 4  # SVI_STITCH_OVERLAP (pixel-frame overlap)
                    frames_needed = prefix_min_context + overlap_size  # 9 frames total
                    start_frame = max(0, int(pred_frames) - frames_needed)

                    # GROUND TRUTH LOG: Predecessor video analysis
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: ========== PREDECESSOR ANALYSIS ==========")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Predecessor video: {predecessor_video_path}")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Predecessor total frames: {pred_frames}")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: prefix_min_context={prefix_min_context}, overlap_size={overlap_size} => frames_needed={frames_needed}")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Extracting frames [{start_frame}:{pred_frames}] (last {frames_needed} frames)")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Frame range breakdown:")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}:   - Frames 0-{start_frame-1}: Will be DISCARDED (not needed)")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}:   - Frames {start_frame}-{pred_frames-1}: Will be EXTRACTED (last {frames_needed} frames)")
                    task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}:   - Last 4 frames [{pred_frames-4}:{pred_frames-1}]: OVERLAP frames for stitching")

                    trimmed_prefix_filename = f"svi_prefix_{segment_idx:02d}_last{frames_needed}frames_{uuid.uuid4().hex[:6]}.mp4"
                    trimmed_prefix_path = segment_processing_dir / trimmed_prefix_filename

                    try:
                        trimmed_result = extract_frame_range_to_video(
                            input_video_path=predecessor_video_path,
                            output_video_path=str(trimmed_prefix_path),
                            start_frame=start_frame,
                            end_frame=None,  # To end
                            fps=float(pred_fps) if pred_fps and pred_fps > 0 else 16.0)
                        # Verify extracted prefix video
                        prefix_frames, prefix_fps = get_video_frame_count_and_fps(trimmed_result)
                        svi_predecessor_video_for_source = str(trimmed_result)
                        task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Extracted prefix video: {prefix_frames} frames (expected: {frames_needed})")
                        task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: Prefix video path: {trimmed_result}")
                        task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: This prefix will be passed to WGP as video_source")
                        if prefix_frames != frames_needed:
                            task_logger.debug(f"[SVI_GROUND_TRUTH] Seg {segment_idx}: WARNING: Extracted {prefix_frames} frames, expected {frames_needed}")
                    except (OSError, ValueError, RuntimeError) as e_trim:
                        # Fallback: use full video if extraction failed
                        svi_predecessor_video_for_source = predecessor_video_path
                        task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: WARNING - Failed to extract last frames ({e_trim}), using full predecessor video")
                else:
                    # Fallback: use full video if we can't get frame count
                    svi_predecessor_video_for_source = predecessor_video_path
                    task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Could not get predecessor frame count, using full video")

                # Still extract last frame for image_refs (anchor reference)
                start_ref_path = extract_last_frame_as_image(
                    predecessor_video_path,
                    segment_processing_dir,
                    task_id
                )
                task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: Extracted last frame as anchor start_ref: {start_ref_path}")
            else:
                task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: ERROR - Predecessor video not available at {predecessor_video_path}")

        # For SVI, end_ref is the target image from input array
        target_end_idx = segment_idx + 1 if segment_idx > 0 else 1
        if svi_predecessor_video_url and segment_idx == 0:
            target_end_idx = 1 if len(orchestrator_images) > 1 else 0

        if len(orchestrator_images) > target_end_idx:
            end_ref_path = orchestrator_images[target_end_idx]
            task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: end_ref from input_images[{target_end_idx}]: {end_ref_path}")
        elif len(orchestrator_images) > 0:
            end_ref_path = orchestrator_images[-1]
            task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: end_ref fallback to last input image: {end_ref_path}")

    # =============================================================================
    # SVI MODE: First segment uses input images normally (no manual predecessor)
    # =============================================================================
    elif use_svi and segment_idx == 0:
        task_logger.debug(f"[SVI_CHAINING] Seg {segment_idx}: First segment in SVI mode - using input images")
        if len(orchestrator_images) > 0:
            start_ref_path = orchestrator_images[0]
        if len(orchestrator_images) > 1:
            end_ref_path = orchestrator_images[1]

    # =============================================================================
    # NON-SVI MODES: Original logic
    # =============================================================================
    # Check individual_segment_params first (highest priority for standalone)
    elif individual_params.get("start_image_url") or individual_params.get("end_image_url"):
        start_ref_path = individual_params.get("start_image_url")
        end_ref_path = individual_params.get("end_image_url")
        task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: Using individual_segment_params URLs: start={start_ref_path}")
    elif len(individual_images) >= 2:
        start_ref_path = individual_images[0]
        end_ref_path = individual_images[1]
        task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: Using individual_segment_params array: start={start_ref_path}")
    elif is_standalone and len(top_level_images) >= 2:
        start_ref_path = top_level_images[0]
        end_ref_path = top_level_images[1]
        task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: Using top-level images directly: start={start_ref_path}")
    elif is_continuing:
        if segment_idx == 0:
            continued_video_path = orchestrator_details.get("continue_from_video_resolved_path")
            if continued_video_path and Path(continued_video_path).exists():
                start_ref_path = extract_last_frame_as_image(continued_video_path, segment_processing_dir, task_id)
            if orchestrator_images:
                end_ref_path = orchestrator_images[0]
        else:
            if len(orchestrator_images) > segment_idx:
                start_ref_path = orchestrator_images[segment_idx - 1]
                end_ref_path = orchestrator_images[segment_idx]
    else:
        task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: Using orchestrator images (from scratch)")
        if len(orchestrator_images) > segment_idx:
            start_ref_path = orchestrator_images[segment_idx]

        if len(orchestrator_images) > segment_idx + 1:
            end_ref_path = orchestrator_images[segment_idx + 1]
    task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: start_ref_path after logic: {start_ref_path}")

    if start_ref_path:
        start_ref_path = download_image_if_url(start_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)
        task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: start_ref_path AFTER DOWNLOAD: {start_ref_path}")
    if end_ref_path:
        end_ref_path = download_image_if_url(end_ref_path, segment_processing_dir, task_id, debug_mode=debug_enabled)

    return ImageRefs(
        start_ref_path=start_ref_path,
        end_ref_path=end_ref_path,
        svi_predecessor_video_for_source=svi_predecessor_video_for_source,
        use_svi=use_svi,
        is_continuing=is_continuing,
    )


def _process_structure_guidance(ctx: SegmentContext, gen: GenerationInputs, task_id: str, main_output_dir_base: Path, mask_active_frames: bool) -> StructureOutputs:
    """Run TravelSegmentProcessor for VACE mode or any mode with structure guidance.

    Returns StructureOutputs with guide/mask video paths and prompt type.
    Raises RuntimeError/ValueError/OSError if the processor fails.
    """
    segment_params = ctx.segment_params
    orchestrator_details = ctx.orchestrator_details
    segment_idx = ctx.segment_idx
    segment_processing_dir = gen.segment_processing_dir
    travel_mode = gen.travel_mode
    debug_enabled = gen.debug_enabled

    guide_video_path = None
    mask_video_path_for_wgp = None
    video_prompt_type_str = None
    image_refs_paths = None
    frames_positions = None
    image_refs_strengths = None
    audio_guide = None
    audio_prompt_type = None
    audio_scale = None
    denoising_strength = None

    travel_guidance_payload = segment_params.get("travel_guidance")
    if isinstance(travel_guidance_payload, dict):
        structure_config: StructureGuidanceConfig | TravelGuidanceConfig
        travel_guidance_config = TravelGuidanceConfig.from_payload(
            travel_guidance_payload,
            gen.model_name,
        )
        if travel_guidance_config.kind in {"vace", "uni3c"}:
            structure_config = travel_guidance_config.to_structure_guidance_config()
        else:
            structure_config = travel_guidance_config
        has_structure_guidance = structure_config.has_guidance
    else:
        # Backward-compatible path for legacy callers and non-travel structure guidance.
        structure_config = StructureGuidanceConfig.from_params({
            **orchestrator_details,
            **segment_params
        })
        has_structure_guidance = structure_config.has_guidance

    # Run TravelSegmentProcessor for:
    # 1. VACE mode (always - requires masks and guides)
    # 2. Any mode with structure guidance configured (enables uni3c/flow guidance for i2v, etc.)
    if travel_mode == "vace" or has_structure_guidance:
        if has_structure_guidance and travel_mode != "vace":
            task_logger.debug(
                f"[STRUCTURE_GUIDANCE] Task {task_id}: Running TravelSegmentProcessor for "
                f"{travel_mode} mode (guidance configured)"
            )
        try:
            processor_context = TravelSegmentContext(
                task_id=task_id,
                segment_idx=segment_idx,
                model_name=gen.model_name,
                total_frames_for_segment=gen.total_frames_for_segment,
                parsed_res_wh=gen.parsed_res_wh,
                segment_processing_dir=segment_processing_dir,
                main_output_dir_base=main_output_dir_base,
                orchestrator_details=orchestrator_details,
                segment_params=segment_params,
                mask_active_frames=mask_active_frames,
                debug_enabled=debug_enabled)
            processor = TravelSegmentProcessor(processor_context)
            segment_outputs = processor.process_segment()

            guide_video_path = segment_outputs.get("video_guide")
            mask_video_path_for_wgp = Path(segment_outputs["video_mask"]) if segment_outputs.get("video_mask") else None
            video_prompt_type_str = segment_outputs["video_prompt_type"]
            image_refs_paths = segment_outputs.get("image_refs_paths")
            frames_positions = segment_outputs.get("frames_positions")
            image_refs_strengths = segment_outputs.get("image_refs_strengths")
            audio_guide = segment_outputs.get("audio_guide")
            audio_prompt_type = segment_outputs.get("audio_prompt_type")
            audio_scale = segment_outputs.get("audio_scale")
            denoising_strength = segment_outputs.get("denoising_strength")
            detected_structure_type = segment_outputs.get("structure_type")

            # Debug: Log segment_outputs keys and structure_type
            task_logger.debug(f"[UNI3C_DEBUG] Task {task_id}: segment_outputs keys: {list(segment_outputs.keys())}")
            task_logger.debug(f"[UNI3C_DEBUG] Task {task_id}: detected_structure_type={repr(detected_structure_type)}, guide_video_path={bool(guide_video_path)}")

            # If config targets uni3c and we have a guide video, update the config's guidance URL
            # This is used later by the UNI3C MODE section
            if structure_config.is_uni3c and guide_video_path:
                task_logger.debug(f"[UNI3C_AUTO] Task {task_id}: Uni3C mode with guide from processor: {guide_video_path}")
                # Store the processor's guide video path in the config
                structure_config._guidance_video_url = guide_video_path

        except (RuntimeError, ValueError, OSError) as e_shared_processor:
            task_logger.error(f"Shared processor failed: {e_shared_processor}", exc_info=True)
            raise RuntimeError(f"Shared processor failed: {e_shared_processor}") from e_shared_processor

    return StructureOutputs(
        guide_video_path=guide_video_path,
        mask_video_path_for_wgp=mask_video_path_for_wgp,
        video_prompt_type_str=video_prompt_type_str,
        structure_config=structure_config,
        image_refs_paths=image_refs_paths,
        frames_positions=frames_positions,
        image_refs_strengths=image_refs_strengths,
        audio_guide=audio_guide,
        audio_prompt_type=audio_prompt_type,
        audio_scale=audio_scale,
        denoising_strength=denoising_strength,
    )


def _build_generation_params(ctx: SegmentContext, gen: GenerationInputs, image_refs: ImageRefs, structure: StructureOutputs, task_id: str) -> dict:
    """Build the WGP generation_params dict from resolved inputs.

    Handles base params, image paths, additional_loras, explicit overrides
    (num_inference_steps, guidance_scale, flow_shift), per-segment LoRAs,
    phase_config parsing, and guide/mask video attachment.
    """
    segment_params = ctx.segment_params
    orchestrator_details = ctx.orchestrator_details
    individual_params = ctx.individual_params
    debug_enabled = gen.debug_enabled
    model_name = gen.model_name
    start_ref_path = image_refs.start_ref_path
    end_ref_path = image_refs.end_ref_path
    guide_video_path = structure.guide_video_path
    mask_video_path_for_wgp = structure.mask_video_path_for_wgp
    video_prompt_type_str = structure.video_prompt_type_str
    structure_config = structure.structure_config

    # Seed: individual_segment_params > top-level > default
    seed_to_use = individual_params.get("seed_to_use") or segment_params.get("seed_to_use", 12345)

    generation_params = {
        "model_name": model_name,
        "negative_prompt": gen.negative_prompt_for_wgp,
        "resolution": f"{gen.parsed_res_wh[0]}x{gen.parsed_res_wh[1]}",
        "video_length": gen.total_frames_for_segment,
        "seed": seed_to_use,
    }

    # Always pass images if available, regardless of specific travel_mode string (hybrid models need them)
    if start_ref_path:
        generation_params["image_start"] = str(Path(start_ref_path).resolve())
    if end_ref_path:
        generation_params["image_end"] = str(Path(end_ref_path).resolve())

    task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: generation_params image_start: {generation_params.get('image_start')}")
    task_logger.debug(f"[IMG_RESOLVE] Task {task_id}: generation_params image_end: {generation_params.get('image_end')}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Parameter extraction with precedence: individual > segment > orchestrator
    # Using _get_param() helper for consistent fallback behavior
    # ═══════════════════════════════════════════════════════════════════════════

    # Travel-only: pass `additional_loras` through on the segment task payload if present,
    # but DO NOT propagate into `generation_params` (so it won't trigger LoRA downloads/application).
    additional_loras = _get_param(
        "additional_loras",
        individual_params,
        segment_params,
        orchestrator_details,
        default={},
        prefer_truthy=True,  # match previous `a or b or c` behavior for empty dicts
    )
    if additional_loras and debug_enabled:
        task_logger.debug(
            f"[TRAVEL_LORA] Task {task_id}: additional_loras present on payload but intentionally ignored for generation ({len(additional_loras)} entries)"
        )

    # IMPORTANT: Do NOT treat orchestrator/UI `steps` as diffusion `num_inference_steps`.
    # In travel payloads, `steps` often means "timeline steps" (UI concept), whereas
    # diffusion steps must come from `num_inference_steps` (or `phase_config.steps_per_phase`).
    explicit_steps = _get_param("num_inference_steps", segment_params, orchestrator_details, prefer_truthy=True)
    if explicit_steps:
        generation_params["num_inference_steps"] = explicit_steps

    explicit_guidance = _get_param("guidance_scale", segment_params, orchestrator_details, prefer_truthy=True)
    if explicit_guidance:
        generation_params["guidance_scale"] = explicit_guidance

    explicit_flow_shift = _get_param("flow_shift", segment_params, orchestrator_details, prefer_truthy=True)
    if explicit_flow_shift:
        generation_params["flow_shift"] = explicit_flow_shift

    # =============================================================================
    # Per-Segment Parameter Overrides
    # =============================================================================
    # Log what per-segment overrides were received
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: ========== Per-Segment Override Check ==========")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: individual_params keys: {list(individual_params.keys()) if individual_params else 'None'}")

    # Check for segment-specific LoRAs first (highest priority)
    # This allows per-segment LoRA overrides to take precedence over phase_config LoRAs
    segment_loras = _get_param("segment_loras", individual_params, default=None)
    segment_lora_config = None

    if segment_loras:
        task_logger.debug(f"[PER_SEGMENT_LORAS] Task {task_id}: Using segment-specific LoRAs ({len(segment_loras)} LoRAs)")
        # Pass through URLs/filenames without downloading - convert_to_wgp_task_impl() handles resolution
        segment_lora_config = True  # Flag to skip phase_config LoRAs below
        activated = []
        mults = []
        for lora_dict in segment_loras:
            path = lora_dict.get("path", "")
            strength = lora_dict.get("strength", 1.0)
            if path:
                activated.append(path)
                mults.append(str(strength))
        if activated:
            generation_params["activated_loras"] = activated
            generation_params["loras_multipliers"] = " ".join(mults)
            headless_logger.info(f"Set {len(activated)} segment-specific LoRAs (download deferred)", task_id=task_id)
    else:
        task_logger.debug(f"[PER_SEGMENT_LORAS] Task {task_id}: No segment-specific LoRAs - will use phase_config/shot-level LoRAs if present")

    # Determine phase_config source for logging
    phase_config_from_individual = individual_params.get("phase_config") if individual_params else None
    phase_config_from_segment = segment_params.get("phase_config") if segment_params else None
    _phase_config_from_orchestrator = orchestrator_details.get("phase_config") if orchestrator_details else None

    phase_config_source = _get_param(
        "phase_config", individual_params, segment_params, orchestrator_details, prefer_truthy=True
    )

    # Log which source phase_config came from
    if phase_config_source:
        if phase_config_from_individual:
            source_name = "PER-SEGMENT (individual_params)"
        elif phase_config_from_segment:
            source_name = "segment_params"
        else:
            source_name = "SHOT-LEVEL (orchestrator_details)"

        preset_name = phase_config_source.get("preset_name", phase_config_source.get("name", "unknown"))
        task_logger.debug(f"[PER_SEGMENT_PHASE_CONFIG] Task {task_id}: Using phase_config from {source_name}")
        task_logger.debug(f"[PER_SEGMENT_PHASE_CONFIG] Task {task_id}: preset_name={preset_name}")

    if phase_config_source:
        try:
            steps_per_phase = phase_config_source.get("steps_per_phase", [2, 2, 2])
            phase_config_steps = sum(steps_per_phase)
            task_logger.debug(f"[PER_SEGMENT_PHASE_CONFIG] Task {task_id}: steps_per_phase={steps_per_phase} (total={phase_config_steps})")

            parsed_phase_config = parse_phase_config(
                phase_config=phase_config_source,
                num_inference_steps=phase_config_steps,
                task_id=task_id,
                model_name=generation_params.get("model_name"),
                debug_mode=debug_enabled
            )

            generation_params["num_inference_steps"] = phase_config_steps

            # Copy non-LoRA params from phase_config
            for key in ["guidance_phases", "switch_threshold", "switch_threshold2",
                       "guidance_scale", "guidance2_scale", "guidance3_scale",
                       "flow_shift", "sample_solver", "model_switch_phase"]:
                if key in parsed_phase_config and parsed_phase_config[key] is not None:
                    generation_params[key] = parsed_phase_config[key]

            # Copy LoRA params from phase_config ONLY if no segment-specific LoRAs
            # (segment_loras takes precedence over phase_config LoRAs)
            # Pass through URLs without downloading - convert_to_wgp_task_impl() handles resolution
            if not segment_lora_config:
                if "lora_names" in parsed_phase_config:
                    generation_params["activated_loras"] = parsed_phase_config["lora_names"]
                if "lora_multipliers" in parsed_phase_config:
                    generation_params["loras_multipliers"] = " ".join(str(m) for m in parsed_phase_config["lora_multipliers"])
                lora_count = len(parsed_phase_config.get("lora_names", []))
                if lora_count:
                    headless_logger.info(f"Set {lora_count} phase_config LoRAs (download deferred)", task_id=task_id)
            else:
                task_logger.debug(f"[PER_SEGMENT_PHASE_CONFIG] Task {task_id}: Skipping phase_config LoRAs (segment-specific LoRAs take precedence)")

            # Pass phase_config patch data through to task_processor.py for proper apply+restore lifecycle
            # (instead of applying directly here where it would never be cleaned up)
            if "_patch_config" in parsed_phase_config:
                generation_params["_parsed_phase_config"] = parsed_phase_config
                generation_params["_phase_config_model_name"] = model_name

        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Task {task_id}: Invalid phase_config: {e}") from e
    else:
        task_logger.debug(f"[PER_SEGMENT_PHASE_CONFIG] Task {task_id}: No phase_config found - using explicit params only")

    # Summary log for per-segment overrides
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: ========== Per-Segment Override Summary ==========")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: LoRAs applied: {generation_params.get('activated_loras', [])}")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: LoRA multipliers: {generation_params.get('loras_multipliers', '')}")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: num_inference_steps: {generation_params.get('num_inference_steps', 'not set')}")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: guidance_scale: {generation_params.get('guidance_scale', 'not set')}")
    task_logger.debug(f"[PER_SEGMENT_HANDLER] Task {task_id}: ================================================")

    if guide_video_path: generation_params["video_guide"] = str(guide_video_path)
    if mask_video_path_for_wgp: generation_params["video_mask"] = str(mask_video_path_for_wgp.resolve())
    generation_params["video_prompt_type"] = video_prompt_type_str

    # Log the complete guidance state being sent to WGP
    task_logger.debug(
        f"[LTX_WGP_HANDOFF] Task {task_id}: "
        f"video_guide={'SET' if guide_video_path else 'NONE'}, "
        f"video_mask={'SET' if mask_video_path_for_wgp else 'NONE'}, "
        f"video_prompt_type='{video_prompt_type_str}', "
        f"use_uni3c={generation_params.get('use_uni3c', False)}, "
        f"uni3c_guide={'SET' if generation_params.get('uni3c_guide_video') else 'NONE'}, "
        f"activated_loras={generation_params.get('activated_loras', [])}"
    )
    if structure.image_refs_paths:
        generation_params["image_refs_paths"] = structure.image_refs_paths
    if structure.frames_positions:
        generation_params["frames_positions"] = structure.frames_positions
    if structure.image_refs_strengths:
        generation_params["image_refs_strengths"] = structure.image_refs_strengths
    if structure.audio_guide:
        generation_params["audio_guide"] = structure.audio_guide
    if structure.audio_prompt_type:
        generation_params["audio_prompt_type"] = structure.audio_prompt_type
    if structure.audio_scale is not None:
        generation_params["audio_scale"] = structure.audio_scale
    if structure.denoising_strength is not None:
        generation_params["denoising_strength"] = structure.denoising_strength

    # Diagnostic: Log guidance configuration when uni3c is enabled
    if structure_config.is_uni3c and guide_video_path:
        task_logger.debug(f"[UNI3C_GUIDANCE_CHECK] Task {task_id}: Checking guidance configuration...")
        task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   video_guide path: {guide_video_path}")
        task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   structure_config.guidance_video_url: {structure_config.guidance_video_url}")
        task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   image_end (i2v): {generation_params.get('image_end', 'None')}")
        task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   video_prompt_type: '{video_prompt_type_str}'")
        if "V" in video_prompt_type_str:
            task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   🔴 WARNING: video_prompt_type has 'V' - VACE will ALSO use video_guide!")
            task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   This causes double-feeding: both VACE and Uni3C use same guide")
        else:
            task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   ✅ OK: video_prompt_type='{video_prompt_type_str}' (no 'V')")
            task_logger.debug(f"[UNI3C_GUIDANCE_CHECK]   Uni3C handles motion guidance, VACE only uses mask for frame preservation")

    return generation_params


def _apply_svi_config(generation_params: dict, ctx: SegmentContext, gen: GenerationInputs, image_refs: ImageRefs, task_id: str) -> None:
    """Apply SVI-specific generation parameters (mutates generation_params in place).

    Enables SVI encoding mode, sets video_prompt_type, configures predecessor video
    as video_source, adjusts video_length for continuation frame accounting,
    copies SVI generation params, and merges SVI LoRAs.
    """
    if not image_refs.use_svi:
        return

    segment_params = ctx.segment_params
    orchestrator_details = ctx.orchestrator_details
    start_ref_path = image_refs.start_ref_path
    svi_predecessor_video_for_source = image_refs.svi_predecessor_video_for_source
    total_frames_for_segment = gen.total_frames_for_segment

    task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Configuring WGP payload for SVI mode")

    # Enable SVI encoding mode
    generation_params["svi2pro"] = True

    # SVI requires video_prompt_type="I" to enable image_refs passthrough
    generation_params["video_prompt_type"] = "I"

    # Set image_refs to start image (anchor for SVI encoding)
    if start_ref_path:
        generation_params["image_refs_paths"] = [str(Path(start_ref_path).resolve())]
        task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Set image_refs_paths to start image: {start_ref_path}")

    # CRITICAL: Pass predecessor video as video_source for SVI continuation
    # WGP uses prefix_video[:, -(5 + overlap_size):] to create overlapped_latents
    # This provides temporal continuity between segments (uses last ~9 frames)
    # IMPORTANT: Must NOT set image_start when video_source is set, otherwise
    # wgp.py prioritizes image_start and creates only a 1-frame prefix_video!
    if svi_predecessor_video_for_source:
        from source.media.video import get_video_frame_count_and_fps as get_video_frame_count_and_fps

        video_source_path = str(Path(svi_predecessor_video_for_source).resolve())
        generation_params["video_source"] = video_source_path

        # GROUND TRUTH LOG: What WGP will receive
        try:
            wgp_input_frames, wgp_input_fps = get_video_frame_count_and_fps(video_source_path)
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: ========== WGP INPUT ANALYSIS ==========")
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: video_source: {video_source_path}")
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: video_source frames: {wgp_input_frames}")
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: WGP will extract last 9 frames from this for overlapped_latents")
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: WGP will prepend ALL {wgp_input_frames} frames to output")
        except (OSError, ValueError, RuntimeError) as e_wgp_log:
            task_logger.debug(f"[SVI_GROUND_TRUTH] Task {task_id}: Could not analyze video_source: {e_wgp_log}")

        # Remove image_start so WGP uses video_source for multi-frame prefix_video
        # The anchor image is provided via image_refs instead
        if "image_start" in generation_params:
            del generation_params["image_start"]
            task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Removed image_start (anchor provided via image_refs)")
        # CRITICAL: Set image_prompt_type to include "V" to enable video_source usage
        # SVI2Pro allows "SVL" - we use "SV" for start+video continuation
        generation_params["image_prompt_type"] = "SV"
        task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Set image_prompt_type='SV' for video continuation")
        task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Set video_source for SVI continuation: {svi_predecessor_video_for_source}")

    # SVI Pro sliding window overlap = 4 frames (standard for SVI)
    # Note: WGP applies latent alignment formula (x-1)//4*4+1, but we bypass it
    # by patching sliding_window_defaults.overlap_default=4 in headless_model_management.py
    generation_params["sliding_window_overlap"] = 4
    task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Set sliding_window_overlap=4 for SVI")

    # IMPORTANT (SVI continuation frame accounting):
    # When `video_source` is provided, WGP will:
    #   - generate `video_length` frames total
    #   - discard the first `sliding_window_overlap` frames of the generated sample
    #   - prepend the prefix video frames (pixels) to the output
    #
    # Therefore the number of **new** frames produced by diffusion is:
    #   new_frames = video_length - sliding_window_overlap
    #
    # In our travel pipeline, `segment_frames_target` / `segment_frames_expanded[idx]` represent
    # the desired number of **new** frames for the segment (4N+1). To actually get that many
    # new frames while still maintaining overlap for stitching, we must request:
    #   video_length = desired_new_frames + sliding_window_overlap
    if svi_predecessor_video_for_source:
        try:
            desired_new_frames = int(total_frames_for_segment) if total_frames_for_segment else 0
            overlap_size = int(generation_params.get("sliding_window_overlap") or 4)
            current_video_length = int(generation_params.get("video_length") or 0)
            if desired_new_frames > 0 and overlap_size > 0:
                # Only bump if the payload is still "new-frames based" (avoid double-bumping).
                if current_video_length == desired_new_frames:
                    generation_params["video_length"] = desired_new_frames + overlap_size
                    task_logger.debug(
                        f"[SVI_PAYLOAD] Task {task_id}: SVI continuation => bump video_length "
                        f"{current_video_length} -> {generation_params['video_length']} "
                        f"(desired_new_frames={desired_new_frames}, overlap_size={overlap_size})"
                    )
                else:
                    task_logger.debug(
                        f"[SVI_PAYLOAD] Task {task_id}: SVI continuation => NOT bumping video_length "
                        f"(current={current_video_length}, desired_new_frames={desired_new_frames}, overlap_size={overlap_size})"
                    )
        except (ValueError, TypeError) as e_len_bump:
            task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: WARNING: Could not adjust video_length for SVI continuation: {e_len_bump}")

    # Add SVI generation parameters from segment_params (set by orchestrator)
    for key in ["guidance_phases", "num_inference_steps", "guidance_scale", "guidance2_scale",
               "flow_shift", "switch_threshold", "model_switch_phase", "sample_solver"]:
        if key in segment_params and segment_params[key] is not None:
            generation_params[key] = segment_params[key]
            task_logger.debug(f"[SVI_PAYLOAD] Task {task_id}: Set {key}={segment_params[key]}")

    # Merge SVI LoRAs with existing LoRAs
    from source.task_handlers.travel.svi_config import merge_svi_into_generation_params

    svi_strength = _get_param("svi_strength", segment_params, orchestrator_details)
    svi_strength_1 = _get_param("svi_strength_1", segment_params, orchestrator_details)
    svi_strength_2 = _get_param("svi_strength_2", segment_params, orchestrator_details)

    merge_svi_into_generation_params(
        generation_params,
        svi_strength=svi_strength,
        svi_strength_1=svi_strength_1,
        svi_strength_2=svi_strength_2)

    task_logger.debug(
        f"[SVI_PAYLOAD] Task {task_id}: Merged SVI LoRAs "
        f"(svi_strength={svi_strength}, svi_strength_1={svi_strength_1}, svi_strength_2={svi_strength_2})"
    )


def _apply_uni3c_config(generation_params: dict, ctx: SegmentContext, gen: GenerationInputs, structure: StructureOutputs, task_id: str) -> None:
    """Apply UNI3C-specific generation parameters (mutates generation_params in place).

    Downloads the guide video if it is a URL, then injects all uni3c_* parameters
    into generation_params.
    """
    structure_config = structure.structure_config
    segment_params = ctx.segment_params
    segment_processing_dir = gen.segment_processing_dir

    # Use unified structure_config exclusively
    use_uni3c = structure_config.is_uni3c

    task_logger.debug(f"[UNI3C_DEBUG] Task {task_id}: structure_config.is_uni3c={structure_config.is_uni3c}, use_uni3c={use_uni3c}")

    if not use_uni3c:
        return

    # Get guide video from config (may have been set by processor)
    uni3c_guide = structure_config.guidance_video_url

    # If Uni3C is enabled but there's no guide video, skip to avoid downstream crashes.
    if not uni3c_guide:
        task_logger.debug(f"[UNI3C] Task {task_id}: Uni3C requested but no guide video provided; skipping Uni3C injection")
        return

    # Use config values (already handles legacy param fallback)
    uni3c_strength = structure_config.strength
    uni3c_start = structure_config.step_window[0]
    uni3c_end = structure_config.step_window[1]
    uni3c_keep_gpu = structure_config.keep_on_gpu
    uni3c_frame_policy = structure_config.frame_policy
    uni3c_zero_empty = structure_config.zero_empty_frames

    # Download guide video if URL
    if uni3c_guide.startswith(("http://", "https://")):
        local_filename = Path(uni3c_guide).name or "uni3c_guide_video.mp4"
        local_download_path = segment_processing_dir / f"uni3c_{local_filename}"
        if not local_download_path.exists():
            download_file(uni3c_guide, segment_processing_dir, local_download_path.name)
            task_logger.debug(f"[UNI3C] Task {task_id}: Downloaded guide video to {local_download_path}")
        else:
            task_logger.debug(f"[UNI3C] Task {task_id}: Guide video already exists at {local_download_path}")
        uni3c_guide = str(local_download_path)

    # Layer 2 logging
    task_logger.debug(f"[UNI3C] Task {task_id}: Uni3C ENABLED (via structure_guidance config)")
    task_logger.debug(f"[UNI3C] Task {task_id}:   guide_video={uni3c_guide}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   strength={uni3c_strength}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   start_percent={uni3c_start}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   end_percent={uni3c_end}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   frame_policy={uni3c_frame_policy}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   keep_on_gpu={uni3c_keep_gpu}")
    task_logger.debug(f"[UNI3C] Task {task_id}:   zero_empty_frames={uni3c_zero_empty}")

    # Inject into generation_params
    generation_params["use_uni3c"] = True
    generation_params["uni3c_guide_video"] = uni3c_guide
    generation_params["uni3c_strength"] = uni3c_strength
    generation_params["uni3c_start_percent"] = uni3c_start
    generation_params["uni3c_end_percent"] = uni3c_end
    generation_params["uni3c_keep_on_gpu"] = uni3c_keep_gpu
    generation_params["uni3c_frame_policy"] = uni3c_frame_policy
    generation_params["uni3c_zero_empty_frames"] = uni3c_zero_empty

    # For i2v with end anchor: blackout last uni3c frame so i2v handles it alone
    # This prevents uni3c from "overcooking" the final frame
    is_last_segment = segment_params.get("is_last_segment", True)
    has_end_anchor = generation_params.get("image_end") is not None
    if is_last_segment and has_end_anchor:
        generation_params["uni3c_blackout_last_frame"] = True
        task_logger.debug(f"[UNI3C] Task {task_id}: Will blackout last frame (is_last_segment + has end anchor)")


def _handle_travel_segment_via_queue_impl(task_params_dict: dict, main_output_dir_base: Path, task_id: str, colour_match_videos: bool, mask_active_frames: bool, task_queue: HeadlessTaskQueue, is_standalone: bool = False):
    """Handle travel segment tasks via direct queue integration to eliminate blocking waits.

    Orchestrator mode (is_standalone=False) requires orchestrator_task_id_ref, orchestrator_run_id, segment_index.
    Standalone mode (is_standalone=True) uses orchestrator_details provided directly in params.
    Parameter precedence: individual_segment_params > segment_params > orchestrator_details.
    """
    headless_logger.debug(f"Starting travel segment queue processing (standalone={is_standalone})", task_id=task_id)
    log_ram_usage("Segment via queue - start", task_id=task_id)

    try:
        from source.task_handlers.travel.chaining import handle_travel_chaining_after_wgp

        # 1. Resolve segment context (mode, orchestrator_details, individual_params, segment_idx)
        ctx = _resolve_segment_context(task_params_dict, is_standalone, task_id)

        # 2. Resolve generation inputs (model, prompt, resolution, frames, output dir, flags)
        gen = _resolve_generation_inputs(ctx, task_id, main_output_dir_base)

        # 3. Resolve image references (SVI chaining, non-SVI image resolution, downloading)
        image_refs = _resolve_image_references(ctx, gen, task_id, is_standalone)

        # 4. Process structure guidance (TravelSegmentProcessor)
        structure = _process_structure_guidance(ctx, gen, task_id, main_output_dir_base, mask_active_frames)

        # 5. Build generation_params dict
        generation_params = _build_generation_params(ctx, gen, image_refs, structure, task_id)

        # 6. Apply SVI-specific configuration
        _apply_svi_config(generation_params, ctx, gen, image_refs, task_id)
        # 7. Apply UNI3C-specific configuration
        _apply_uni3c_config(generation_params, ctx, gen, structure, task_id)

        # Log diagnostic summary before WGP submission
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: ========== WGP GENERATION REQUEST ==========")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_length (target frames): {generation_params.get('video_length')}")
        is_valid_4n1 = (generation_params.get('video_length', 0) - 1) % 4 == 0
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: Valid 4N+1: {is_valid_4n1} {'✓' if is_valid_4n1 else '✗ WARNING'}")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_guide: {generation_params.get('video_guide', 'None')}")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: video_mask: {generation_params.get('video_mask', 'None')}")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: model: {gen.model_name}")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: resolution: {generation_params.get('resolution')}")
        task_logger.debug(f"[WGP_SUBMIT] Task {task_id}: =============================================")

        # IMPORTANT: Use the DB task_id as the queue task id.
        # This keeps logs, fatal error handling, and debug tooling consistent (no "travel_seg_" indirection).
        # We still include a hint in parameters so the queue can apply any task-type specific behavior.
        generation_params["_source_task_type"] = "travel_segment"
        from source.task_handlers.queue.task_queue import GenerationTask

        generation_task = GenerationTask(
            id=task_id,
            model=gen.model_name,
            prompt=gen.prompt_for_wgp,
            parameters=generation_params
        )

        _submitted_task_id = task_queue.submit_task(generation_task)

        max_wait_time = 1800
        wait_interval = 2
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            status = task_queue.get_task_status(task_id)
            if status is None: return False, f"Travel segment {task_id}: Task status became None"

            if status.status == "completed":
                # In standalone mode, skip chaining (no orchestrator to coordinate with)
                if is_standalone:
                    headless_logger.debug(f"Standalone segment completed, skipping chaining", task_id=task_id)
                    return True, status.result_path

                # Orchestrator mode: run chaining
                chain_success, chain_message, final_chained_path = handle_travel_chaining_after_wgp(
                    wgp_task_params={
                        # Provide minimal ground-truth frame params for downstream debug logging / trimming analysis.
                        "use_svi": image_refs.use_svi,
                        "video_length": generation_params.get("video_length"),
                        "sliding_window_overlap": generation_params.get("sliding_window_overlap"),
                        "video_source": generation_params.get("video_source"),
                        "travel_chain_details": {
                            "orchestrator_task_id_ref": ctx.orchestrator_task_id_ref,
                            "orchestrator_run_id": ctx.orchestrator_run_id,
                            "segment_index_completed": ctx.segment_idx,
                            "orchestrator_details": ctx.orchestrator_details,
                            "segment_processing_dir_for_saturation": str(gen.segment_processing_dir),
                            "is_first_new_segment_after_continue": ctx.segment_params.get("is_first_segment", False) and ctx.orchestrator_details.get("continue_from_video_resolved_path"),
                            "is_subsequent_segment": not ctx.segment_params.get("is_first_segment", True),
                            "colour_match_videos": colour_match_videos,
                            "cm_start_ref_path": None, "cm_end_ref_path": None, "show_input_images": False, "start_image_path": None, "end_image_path": None,
                        }
                    },
                    actual_wgp_output_video_path=status.result_path,
                    image_download_dir=gen.segment_processing_dir,
                    main_output_dir_base=main_output_dir_base)
                if chain_success and final_chained_path:
                    return True, final_chained_path
                else:
                    return True, status.result_path
            elif status.status == "failed":
                return False, f"Travel segment {task_id}: Generation failed: {status.error_message}"

            time.sleep(wait_interval)
            elapsed_time += wait_interval

        return False, f"Travel segment {task_id}: Generation timeout"

    except (RuntimeError, ValueError, OSError, KeyError) as e:
        task_logger.error(f"Travel segment {task_id}: Exception: {e}", exc_info=True)
        return False, f"Travel segment {task_id}: Exception: {str(e)}"

class TaskRegistry:
    """Registry for task handlers."""

    @staticmethod
    def _resolve_handler(task_type: str):
        """Resolve a registered handler callable through the lazy import manifest."""
        return _load_handler_callable(task_type)

    @staticmethod
    def validate_registered_handlers() -> dict[str, str]:
        failures: dict[str, str] = {}
        for task_type in _HANDLER_SPECS:
            try:
                _load_handler_callable(task_type)
            except Exception as exc:
                failures[task_type] = str(exc)
        return failures
    
    @staticmethod
    def dispatch(task_type: str, context: "TaskDispatchContext") -> Tuple[bool, Optional[str]]:
        """Dispatch a task to the appropriate handler.

        Args:
            task_type: The type of task to execute.
            context: TaskDispatchContext with task_id, task_params_dict,
                     main_output_dir_base, project_id, task_queue, etc.

        Returns:
            Tuple (success, output_location)
        """
        task_id = context["task_id"]
        params = context["task_params_dict"]

        # 1. Direct Queue Tasks
        if task_type in DIRECT_QUEUE_TASK_TYPES and context["task_queue"]:
            return TaskRegistry._handle_direct_queue_task(task_type, context)

        # 2. Orchestrator & Specialized Handlers
        handlers = {
            "travel_orchestrator": lambda: travel_orchestrator.handle_travel_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"]),
            "travel_segment": lambda: _load_handler_callable("travel_segment")(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                is_standalone=False
            ),
            "individual_travel_segment": lambda: _load_handler_callable("individual_travel_segment")(
                task_params_dict=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                colour_match_videos=context["colour_match_videos"],
                mask_active_frames=context["mask_active_frames"],
                task_queue=context["task_queue"],
                is_standalone=True
            ),
            "travel_stitch": lambda: handle_travel_stitch_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                stitch_task_id_str=task_id),
            "magic_edit": lambda: me.handle_magic_edit_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id),
            "join_clips_orchestrator": lambda: handle_join_clips_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"]),
            "edit_video_orchestrator": lambda: handle_edit_video_orchestrator_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                orchestrator_task_id_str=task_id,
                orchestrator_project_id=context["project_id"]),
            "join_clips_segment": lambda: handle_join_clips_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"]),
            "join_final_stitch": lambda: handle_join_final_stitch(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id),
            "inpaint_frames": lambda: handle_inpaint_frames_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id,
                task_queue=context["task_queue"]),
            "create_visualization": lambda: handle_create_visualization_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                viz_task_id_str=task_id),
            "extract_frame": lambda: handle_extract_frame_task(
                params, context["main_output_dir_base"], task_id
            ),
            "rife_interpolate_images": lambda: handle_rife_interpolate_task(
                params, context["main_output_dir_base"], task_id, task_queue=context["task_queue"]
            ),
            "comfy": lambda: handle_comfy_task(
                task_params_from_db=params,
                main_output_dir_base=context["main_output_dir_base"],
                task_id=task_id)
        }

        if task_type in handlers:
            if task_type not in ["travel_orchestrator", "join_clips_orchestrator", "edit_video_orchestrator"]:
                return handlers[task_type]()

            dispatch_params = normalize_task_dispatch_payload(params, task_id=task_id)
            orchestrator_handlers = {
                "travel_orchestrator": lambda: travel_orchestrator.handle_travel_orchestrator_task(
                    task_params_from_db=dispatch_params,
                    main_output_dir_base=context["main_output_dir_base"],
                    orchestrator_task_id_str=task_id,
                    orchestrator_project_id=context["project_id"]),
                "join_clips_orchestrator": lambda: handle_join_clips_orchestrator_task(
                    task_params_from_db=dispatch_params,
                    main_output_dir_base=context["main_output_dir_base"],
                    orchestrator_task_id_str=task_id,
                    orchestrator_project_id=context["project_id"]),
                "edit_video_orchestrator": lambda: handle_edit_video_orchestrator_task(
                    task_params_from_db=dispatch_params,
                    main_output_dir_base=context["main_output_dir_base"],
                    orchestrator_task_id_str=task_id,
                    orchestrator_project_id=context["project_id"]),
            }
            return orchestrator_handlers[task_type]()

        # Default fallthrough to queue
        if context["task_queue"]:
             return TaskRegistry._handle_direct_queue_task(task_type, context)
        
        raise ValueError(f"Unknown task type {task_type} and no queue available")

    @staticmethod
    def _handle_direct_queue_task(task_type: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        task_id = context["task_id"]
        task_queue = context["task_queue"]
        params = context["task_params_dict"]
        
        try:
            generation_task = db_task_to_generation_task(
                params, task_id, task_type, context["wan2gp_path"], context["debug_mode"]
            )
            
            if task_type == "wan_2_2_t2i":
                generation_task.parameters["video_length"] = 1
            
            if context["colour_match_videos"]:
                generation_task.parameters["colour_match_videos"] = True
            if context["mask_active_frames"]:
                generation_task.parameters["mask_active_frames"] = True
            
            task_queue.submit_task(generation_task)
            
            # Wait for completion
            max_wait_time = 3600
            elapsed = 0
            while elapsed < max_wait_time:
                status = task_queue.get_task_status(task_id)
                if not status: return False, "Task status became None"
                
                if status.status == "completed":
                    return True, status.result_path
                elif status.status == "failed":
                    return False, status.error_message or "Failed without message"
                
                time.sleep(2)
                elapsed += 2
            
            return False, "Timeout"
            
        except (RuntimeError, ValueError, OSError) as e:
            task_logger.error(f"Queue error: {e}", exc_info=True)
            return False, f"Queue error: {e}"
