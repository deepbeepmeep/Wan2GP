"""TypedDict contracts for major implicit dict interfaces.

These TypedDicts document the contracts that were previously implicit dict
structures. They enable IDE autocomplete, catch key typos at definition time,
and provide a single source of truth for what keys each dict must/may contain.

Usage:
    from source.core.params.contracts import TaskDispatchContext, OrchestratorDetails

    def dispatch(task_type: str, context: TaskDispatchContext) -> ...:
        task_id = context["task_id"]  # IDE knows this is str
"""

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from source.task_handlers.queue.task_queue import HeadlessTaskQueue


# ---------------------------------------------------------------------------
# TaskDispatchContext — passed from worker.py → TaskRegistry.dispatch()
# ---------------------------------------------------------------------------

class TaskDispatchContext(TypedDict):
    """Context dict constructed in worker.py process_single_task()."""

    task_id: str
    task_params_dict: dict
    main_output_dir_base: Path
    project_id: Optional[str]
    task_queue: Optional["HeadlessTaskQueue"]
    colour_match_videos: bool
    mask_active_frames: bool
    debug_mode: bool
    wan2gp_path: str


# ---------------------------------------------------------------------------
# OrchestratorDetails — the big dict inside task_params["orchestrator_details"]
# Split into Required (always present) and full (with optional keys).
# ---------------------------------------------------------------------------

class _OrchestratorDetailsRequired(TypedDict):
    """Keys that MUST be present in orchestrator_details for travel orchestration."""

    model_name: str
    parsed_resolution_wh: str
    segment_frames_expanded: List[int]
    num_new_segments_to_generate: int
    base_prompts_expanded: List[str]
    negative_prompts_expanded: List[str]
    frame_overlap_expanded: List[int]
    input_image_paths_resolved: List[str]


class OrchestratorDetails(_OrchestratorDetailsRequired, total=False):
    """Full orchestrator_details dict with all optional keys."""

    # Segment chaining
    chain_segments: bool  # default True
    continue_from_video_resolved_path: str
    continuation_config: dict

    # Model / generation
    model_type: str  # "vace", "i2v", etc.
    use_svi: bool
    seed_base: int
    fps_helpers: int
    num_inference_steps: int

    # Phase config (parsed from phase_config dict)
    phase_config: dict
    guidance_phases: list
    switch_threshold: float
    switch_threshold2: float
    guidance_scale: float
    guidance2_scale: float
    guidance3_scale: float
    flow_shift: float
    sample_solver: str
    model_switch_phase: int
    lora_multipliers: list

    # Prompts
    base_prompt: str
    negative_prompt: str
    text_before_prompts: str
    text_after_prompts: str
    enhanced_prompts_expanded: List[str]
    enhance_prompt: bool
    vlm_device: str

    # Per-segment overrides
    phase_configs_expanded: list
    loras_per_segment_expanded: list
    lora_names: list

    # Structure guidance
    travel_guidance: Optional[dict]
    structure_video_path: Optional[str]
    structure_video_treatment: str
    structure_type: str
    structure_video_type: str
    structure_video_motion_strength: float
    structure_canny_intensity: float
    structure_depth_contrast: float
    structure_videos: list
    structure_original_video_url: str
    structure_trimmed_video_url: str

    # VACE
    vace_image_refs_to_prepare_by_worker: list

    # Output / run management
    run_id: str
    main_output_dir_for_run: str
    shot_id: Optional[str]
    parent_generation_id: Optional[str]

    # Post-processing
    crossfade_sharp_amt: float
    upscale_factor: float
    upscale_model_name: Optional[str]
    subsequent_starting_strength_adjustment: float
    desaturate_subsequent_starting_frames: float
    adjust_brightness_subsequent_starting_frames: float
    after_first_post_generation_saturation: Optional[float]
    after_first_post_generation_brightness: Optional[float]

    # Debug / cleanup
    debug_mode_enabled: bool
    skip_cleanup_enabled: bool

    # WGP override
    cfg_star_switch: int
    cfg_zero_step: int
    params_json_str_override: Optional[str]

    # Polling (nested in original_common_args, also at top level for stitch)
    poll_interval: int
    poll_timeout: int

    # Internal (set during orchestration)
    _consolidated_end_anchors: list
    _consolidated_keyframe_positions: list
    orchestrator_task_id: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_ORCHESTRATOR_KEYS = frozenset(_OrchestratorDetailsRequired.__annotations__.keys())
REQUIRED_ORCHESTRATOR_KEYS = _REQUIRED_ORCHESTRATOR_KEYS


def validate_orchestrator_details(
    details: dict,
    *,
    context: str = "orchestrator",
    task_id: str = "unknown",
) -> None:
    """Validate that all required keys are present in orchestrator_details.

    Raises ValueError with a clear message listing missing keys.
    """
    missing = _REQUIRED_ORCHESTRATOR_KEYS - details.keys()
    if missing:
        raise ValueError(
            f"{context} (task {task_id}): orchestrator_details missing required keys: "
            f"{sorted(missing)}"
        )
