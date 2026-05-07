"""Subprocess adapter for Sprint 2 direct VibeComfy routes.

The worker must not import VibeComfy directly.  This module crosses the Python
3.11 VibeComfy boundary with ``subprocess`` only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from source.core.log import headless_logger
from source.models.model_handlers.qwen_compositor import create_qwen_masked_composite
from source.runtime.vibecomfy_profile import (
    PROCESS_DEFAULT_PROFILE,
    build_memory_profile_cli_args,
)
from source.utils.download_utils import download_image_if_url, download_video_if_url
from source.media.video_contract import (
    VIDEO_EXTENSIONS,
    VideoArtifactContract,
    VideoContractError,
    validate_video_artifact,
)
from source.task_handlers.tasks.template_routing import (
    ResolvedTask,
    RouteSupportState,
    WorkerBackend,
)


_MAX_CAPTURE_CHARS = 4000
_OUTPUT_EXTENSIONS = {
    ".apng",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
}


def handle_vibecomfy_resolved_task(
    resolved: ResolvedTask,
    main_output_dir_base: str | Path,
) -> tuple[bool, str | None]:
    """Run a supported direct VibeComfy route and return the discovered output."""

    validation_error = _validate_supported_resolved_task(resolved)
    if validation_error:
        return False, validation_error

    run_workspace = _prepare_run_workspace(main_output_dir_base, resolved.task_id)
    command = _build_vibecomfy_command(resolved, run_workspace)
    env = _build_subprocess_env(run_workspace)

    headless_logger.debug_block(
        "VIBECOMFY_ROUTE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "support_state": resolved.support_state.value,
            "memory_profile": _memory_profile_for_log(resolved),
        },
        task_id=resolved.task_id,
    )

    try:
        completed = subprocess.run(
            command,
            cwd=run_workspace,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        message = _failure_message(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        _log_failure(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    stdout = _bounded(completed.stdout)
    stderr = _bounded(completed.stderr)
    if completed.returncode != 0:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    output_path = _discover_output_path(stdout=stdout, run_workspace=run_workspace)
    if output_path is None:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    media_metadata = None
    if Path(output_path).suffix.lower() in VIDEO_EXTENSIONS:
        try:
            media_metadata = validate_video_artifact(
                output_path,
                _video_contract_for_resolved_task(resolved),
            )
        except VideoContractError as exc:
            message = _failure_message(
                resolved=resolved,
                exit_code=completed.returncode,
                stderr=f"media contract violation: {exc}",
                stdout=stdout,
            )
            _log_failure(
                resolved=resolved,
                exit_code=completed.returncode,
                stderr=f"media contract violation: {exc}",
                stdout=stdout,
            )
            headless_logger.error(message, task_id=resolved.task_id)
            return False, message

    headless_logger.debug_block(
        "VIBECOMFY_COMPLETE",
        {
            "task_id": resolved.task_id,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": completed.returncode,
            "output_path": output_path,
            "media_metadata": _video_metadata_for_log(media_metadata),
        },
        task_id=resolved.task_id,
    )
    return True, output_path


def _validate_supported_resolved_task(resolved: ResolvedTask) -> str | None:
    if resolved.backend != WorkerBackend.VIBECOMFY:
        return f"VibeComfy adapter received non-VibeComfy backend {resolved.backend.value}"
    if resolved.fail_closed_reason:
        return (
            f"VibeComfy backend fail-closed for task {resolved.task_id} "
            f"({resolved.route_key}): {resolved.fail_closed_reason}"
        )
    if resolved.support_state != RouteSupportState.VIBECOMFY_SUPPORTED:
        return (
            f"VibeComfy route {resolved.route_key!r} is "
            f"{resolved.support_state.value}; adapter will not execute it"
        )
    if not resolved.template_id:
        return f"VibeComfy route {resolved.route_key!r} has no template_id"
    return None


def _build_vibecomfy_command(resolved: ResolvedTask, run_workspace: Path) -> list[str]:
    workflow_ref, ready = _workflow_reference_for_resolved_task(resolved, run_workspace)
    command = [
        _vibecomfy_python(),
        "-m",
        "vibecomfy.cli",
        "run",
        workflow_ref,
        "--runtime",
        "embedded",
    ]
    if ready:
        command.append("--ready")
        prompt = resolved.params.get("prompt")
        if prompt is not None:
            command.extend(["--prompt", str(prompt)])

        seed = resolved.params.get("seed")
        if seed is not None:
            command.extend(["--seed", str(int(seed))])

        steps = resolved.params.get("steps", resolved.params.get("num_inference_steps"))
        if steps is not None:
            command.extend(["--steps", str(int(steps))])

    command.extend(
        build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    )
    return command


def _workflow_reference_for_resolved_task(resolved: ResolvedTask, run_workspace: Path) -> tuple[str, bool]:
    if resolved.route_key == "z_image_turbo":
        return str(_write_z_image_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "z_image_turbo_i2i":
        return str(_write_z_image_img2img_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "qwen_image_2512":
        return str(_write_qwen_image_2512_scratchpad(resolved, run_workspace)), False
    if resolved.route_key in {
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    }:
        return str(_write_qwen_image_edit_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "wan_2_2_t2i":
        return str(_write_wan_2_2_t2i_scratchpad(resolved, run_workspace)), False
    if _is_wan_vace_route(resolved.route_key):
        return str(_write_wan_2_2_vace_scratchpad(resolved, run_workspace)), False
    return str(resolved.template_id), True


def _write_z_image_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(resolved.params.get("resolution") or "1024x1024")
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 8)))
    scratchpad = run_workspace / "z_image_turbo_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "from vibecomfy.patches.resolution import resolution",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/z_image')",
                f"    resolution({width}, {height}).apply(workflow)",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.set_steps({steps})",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_qwen_image_2512_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(resolved.params.get("resolution") or "768x768")
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 4)))
    scratchpad = run_workspace / "qwen_image_2512_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "from vibecomfy.patches.resolution import resolution",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/qwen_image_2512')",
                f"    resolution({width}, {height}).apply(workflow)",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.nodes['238:224'].inputs['value'] = {steps}",
                f"    workflow.nodes['238:225'].inputs['value'] = {steps}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_qwen_image_edit_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_qwen_edit_input(resolved, run_workspace)
    prompt = str(resolved.params.get("prompt") or _default_qwen_edit_prompt(resolved.route_key))
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 4)))
    scratchpad = run_workspace / f"{resolved.route_key}_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('edit/qwen_image_edit')",
                f"    workflow.nodes['78'].inputs['image'] = {json.dumps(input_name)}",
                f"    workflow.nodes['102:76'].inputs['image'] = ['78', 0]",
                f"    workflow.nodes['102:77'].inputs['image'] = ['78', 0]",
                f"    workflow.nodes['102:88'].inputs['pixels'] = ['78', 0]",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.nodes['102:103'].inputs['value'] = {steps}",
                f"    workflow.nodes['102:106'].inputs['value'] = {steps}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_z_image_img2img_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_image_input(
        resolved,
        run_workspace,
        "image",
        "image_url",
        "image_guide",
        fallback_filename=f"z_image_img2img_{resolved.task_id}.png",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or "1024x1024"
    )
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 8)))
    denoise = float(resolved.params.get("denoising_strength", resolved.params.get("denoise", 0.7)))
    scratchpad = run_workspace / "z_image_img2img_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/z_image_img2img')",
                f"    input_name = {json.dumps(input_name)}",
                "    for node in workflow.nodes.values():",
                "        if node.class_type == 'LoadImage':",
                "            node.inputs['image'] = input_name",
                "        elif node.class_type == 'ImageScale':",
                f"            node.inputs['width'] = {width}",
                f"            node.inputs['height'] = {height}",
                "        elif node.class_type == 'KSampler':",
                f"            node.inputs['seed'] = {seed}",
                f"            node.inputs['steps'] = {steps}",
                f"            node.inputs['denoise'] = {denoise}",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_wan_2_2_t2i_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or "832x480"
    )
    prompt = str(resolved.params.get("prompt") or "")
    negative = str(resolved.params.get("negative_prompt") or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted")
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    cfg_1 = float(resolved.params.get("guidance_scale", 3))
    cfg_2 = float(resolved.params.get("guidance2_scale", 1))
    shift = float(resolved.params.get("flow_shift", 5))
    scratchpad = run_workspace / "wan_2_2_t2i_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_t2i')",
                f"    workflow.nodes['78'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['78'].inputs['widget_1'] = {height}",
                "    workflow.nodes['78'].inputs['widget_2'] = 1",
                f"    workflow.nodes['78'].inputs['width'] = {width}",
                f"    workflow.nodes['78'].inputs['height'] = {height}",
                "    workflow.nodes['78'].inputs['num_frames'] = 1",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_1'] = {cfg_1}",
                f"    workflow.nodes['27'].inputs['widget_2'] = {shift}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['87'].inputs['steps'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_1'] = {cfg_2}",
                f"    workflow.nodes['87'].inputs['widget_2'] = {shift}",
                f"    workflow.nodes['87'].inputs['widget_3'] = {seed}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_wan_2_2_vace_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    start_name = _materialize_image_input(
        resolved,
        run_workspace,
        "start_image_url",
        "start_image",
        "image",
        "image_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=0,
        fallback_filename=f"vace_start_{resolved.task_id}.png",
    )
    end_name = _materialize_image_input(
        resolved,
        run_workspace,
        "end_image_url",
        "end_image",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=1,
        fallback_filename=f"vace_end_{resolved.task_id}.png",
    )
    control_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "video_source",
        "video_guide",
        "structure_video_path",
        "source_video_url",
        "vid2vid_source_video_path",
        nested_keys=("travel_guidance", "individual_segment_params", "segment_params", "orchestrator_details"),
        fallback_filename=f"vace_control_{resolved.task_id}.mp4",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "parsed_resolution_wh")
        or "832x480"
    )
    frames = int(
        resolved.params.get("num_frames")
        or resolved.params.get("video_length")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "num_frames", "video_length")
        or 81
    )
    fps = int(float(resolved.params.get("fps") or resolved.params.get("fps_helpers") or 16))
    prompt = str(
        resolved.params.get("prompt")
        or resolved.params.get("base_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "prompt", "base_prompt")
        or ""
    )
    negative = str(
        resolved.params.get("negative_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "negative_prompt")
        or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
    )
    seed = int(
        resolved.params.get("seed")
        or resolved.params.get("seed_to_use")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "seed", "seed_to_use", "seed_base")
        or -1
    )
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    scratchpad = run_workspace / "wan_2_2_vace_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_vace_cocktail')",
                f"    workflow.nodes['64'].inputs['image'] = {json.dumps(start_name)}",
                f"    workflow.nodes['112'].inputs['image'] = {json.dumps(end_name)}",
                f"    control_name = {json.dumps(control_name)}",
                "    if control_name:",
                "        workflow.nodes['199'].inputs['video'] = control_name",
                f"        workflow.nodes['199'].inputs['force_rate'] = {fps}",
                f"        workflow.nodes['199'].inputs['custom_width'] = {width}",
                f"        workflow.nodes['199'].inputs['custom_height'] = {height}",
                f"        workflow.nodes['199'].inputs['frame_load_cap'] = {frames}",
                "    else:",
                "        workflow.disconnect('111.control_images')",
                f"    workflow.nodes['111'].inputs['widget_0'] = {frames}",
                f"    workflow.nodes['56'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['56'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['56'].inputs['widget_2'] = {frames}",
                f"    workflow.nodes['56'].inputs['width'] = {width}",
                f"    workflow.nodes['56'].inputs['height'] = {height}",
                f"    workflow.nodes['56'].inputs['num_frames'] = {frames}",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['139'].inputs['frame_rate'] = {fps}",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['87'].inputs['steps'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['197'].inputs['steps'] = {steps}",
                f"    workflow.nodes['197'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['197'].inputs['widget_3'] = {seed}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _materialize_qwen_edit_input(resolved: ResolvedTask, run_workspace: Path) -> str:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if resolved.route_key in {"image_inpaint", "annotated_image_edit"}:
        image_source = _first_string_param(resolved.params, "image_guide", "image_url", "image")
        mask_source = _first_string_param(resolved.params, "mask_url", "mask")
        if image_source and mask_source:
            composite = create_qwen_masked_composite(
                image_source,
                mask_source,
                input_dir,
                task_id=resolved.task_id,
            )
            return _copy_to_input_dir(composite, input_dir, f"{resolved.route_key}_{resolved.task_id}.jpg")

    image_source = _first_string_param(
        resolved.params,
        "image_guide",
        "image_url",
        "image",
        "style_reference_image",
        "subject_reference_image",
    )
    if not image_source:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires an input image")
    return _copy_to_input_dir(
        download_image_if_url(image_source, input_dir, resolved.task_id),
        input_dir,
        f"{resolved.route_key}_{resolved.task_id}.png",
    )


def _is_wan_vace_route(route_key: str) -> bool:
    return (
        "model-wan22_vace" in route_key
        or route_key.startswith("join_clips_segment__model-wan22_vace__")
    )


def _materialize_image_input(
    resolved: ResolvedTask,
    run_workspace: Path,
    *keys: str,
    nested_keys: tuple[str, ...] = (),
    list_keys: tuple[str, ...] = (),
    list_index: int = 0,
    fallback_filename: str,
) -> str:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_source = (
        _first_string_param(resolved.params, *keys)
        or _first_nested_string(resolved.params, nested_keys, *keys)
        or _first_list_string(resolved.params, list_keys, list_index)
        or _first_nested_list_string(resolved.params, nested_keys, list_keys, list_index)
    )
    if not image_source:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires image input {keys!r}")
    return _copy_to_input_dir(
        download_image_if_url(image_source, input_dir, resolved.task_id),
        input_dir,
        fallback_filename,
    )


def _materialize_optional_video_input(
    resolved: ResolvedTask,
    run_workspace: Path,
    *keys: str,
    nested_keys: tuple[str, ...] = (),
    fallback_filename: str,
) -> str | None:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    video_source = (
        _first_string_param(resolved.params, *keys)
        or _first_nested_string(resolved.params, nested_keys, *keys)
        or _first_travel_guidance_video(resolved.params)
    )
    if not video_source:
        return None
    return _copy_to_input_dir(
        download_video_if_url(video_source, input_dir, resolved.task_id),
        input_dir,
        fallback_filename,
    )


def _copy_to_input_dir(source: str | Path, input_dir: Path, filename: str) -> str:
    source_path = Path(source)
    suffix = source_path.suffix or Path(filename).suffix or ".png"
    target = input_dir / (Path(filename).stem + suffix)
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return target.name


def _default_qwen_edit_prompt(route_key: str) -> str:
    if route_key == "image_inpaint":
        return "Repair the highlighted green mask area while preserving the original scene."
    if route_key == "annotated_image_edit":
        return "Apply the requested edit indicated by the annotation while preserving the original scene."
    if route_key == "qwen_image_style":
        return "Restyle the subject using the reference image while preserving the subject identity."
    return "Apply the requested image edit while preserving the main subject and scene."


def _first_string_param(params: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_nested_string(params: Mapping[str, Any], nested_keys: tuple[str, ...], *keys: str) -> str | None:
    value = _first_nested_value(params, nested_keys, *keys)
    return value if isinstance(value, str) and value.strip() else None


def _first_nested_value(params: Mapping[str, Any], nested_keys: tuple[str, ...], *keys: str) -> Any:
    for nested_key in nested_keys:
        nested = params.get(nested_key)
        if isinstance(nested, Mapping):
            for key in keys:
                value = nested.get(key)
                if value is not None and value != "":
                    return value
    return None


def _first_list_string(params: Mapping[str, Any], list_keys: tuple[str, ...], index: int) -> str | None:
    for key in list_keys:
        value = params.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) > index:
            item = value[index]
            if isinstance(item, str) and item.strip():
                return item
    return None


def _first_nested_list_string(
    params: Mapping[str, Any],
    nested_keys: tuple[str, ...],
    list_keys: tuple[str, ...],
    index: int,
) -> str | None:
    for nested_key in nested_keys:
        nested = params.get(nested_key)
        if isinstance(nested, Mapping):
            found = _first_list_string(nested, list_keys, index)
            if found:
                return found
    return None


def _first_travel_guidance_video(params: Mapping[str, Any]) -> str | None:
    guidance = params.get("travel_guidance")
    if isinstance(guidance, Mapping):
        videos = guidance.get("videos")
        if isinstance(videos, Sequence) and not isinstance(videos, (str, bytes, bytearray)):
            for entry in videos:
                if isinstance(entry, str) and entry.strip():
                    return entry
                if isinstance(entry, Mapping):
                    for key in ("url", "path", "video_url", "video_path"):
                        value = entry.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
    return None


def _parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value).strip().lower().replace(" ", "").replace("×", "x")
    if "x" not in text:
        raise ValueError(f"invalid VibeComfy resolution {value!r}")
    width_raw, height_raw = text.split("x", 1)
    width, height = int(width_raw), int(height_raw)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid VibeComfy resolution {value!r}")
    return width, height


def _prepare_run_workspace(main_output_dir_base: str | Path, task_id: str) -> Path:
    safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
    run_workspace = Path(main_output_dir_base) / "vibecomfy_runs" / safe_task_id
    run_workspace.mkdir(parents=True, exist_ok=True)
    return run_workspace


def _build_subprocess_env(run_workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VIBECOMFY_WORKER_RUN_DIR"] = str(run_workspace)
    input_dir = run_workspace / "input"
    output_dir = run_workspace / "output"
    temp_dir = run_workspace / "temp"
    for path in (input_dir, output_dir, temp_dir):
        path.mkdir(parents=True, exist_ok=True)
    comfy_config = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "temp_directory": str(temp_dir),
    }
    existing_config = env.get("VIBECOMFY_COMFY_CONFIGURATION")
    if existing_config:
        try:
            parsed = json.loads(existing_config)
            if isinstance(parsed, dict):
                comfy_config = {**parsed, **comfy_config}
        except json.JSONDecodeError:
            pass
    env["VIBECOMFY_COMFY_CONFIGURATION"] = json.dumps(comfy_config)

    vibecomfy_cwd = os.environ.get("VIBECOMFY_CWD")
    if vibecomfy_cwd:
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{vibecomfy_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else vibecomfy_cwd
        )

    return env


def _vibecomfy_python() -> str:
    return os.environ.get("VIBECOMFY_PYTHON") or "python3.11"


def _process_default_memory_profile() -> int | None:
    raw_profile = os.environ.get("VIBECOMFY_MEMORY_PROFILE")
    if raw_profile is None or raw_profile.strip() == "":
        return None
    return int(raw_profile)


def _override_memory_profile(resolved: ResolvedTask) -> int | None:
    raw_profile = resolved.params.get("override_profile")
    if raw_profile is None:
        return PROCESS_DEFAULT_PROFILE
    return int(raw_profile)


def _memory_profile_for_log(resolved: ResolvedTask) -> int | None:
    try:
        args = build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    except ValueError:
        return None
    if "--memory-profile" not in args:
        return None
    return int(args[args.index("--memory-profile") + 1])


def _video_contract_for_resolved_task(resolved: ResolvedTask) -> VideoArtifactContract:
    width, height = _expected_dimensions(resolved.params)
    return VideoArtifactContract(
        expected_frame_count=_int_param(resolved.params, "expected_frame_count", "num_frames", "video_length"),
        expected_fps=_float_param(resolved.params, "expected_fps", "fps", "fps_helpers"),
        expected_duration_seconds=_float_param(resolved.params, "expected_duration_seconds", "duration_seconds"),
        require_audio=_bool_param(resolved.params, "require_audio", "audio_required", "requires_audio"),
        expected_width=width,
        expected_height=height,
        require_thumbnail=_bool_param(resolved.params, "require_thumbnail", "thumbnail_required", "requires_thumbnail"),
        thumbnail_path=_string_param(resolved.params, "thumbnail_path", "thumbnail_storage_path"),
    )


def _expected_dimensions(params: dict[str, Any]) -> tuple[int | None, int | None]:
    width = _int_param(params, "expected_width", "width")
    height = _int_param(params, "expected_height", "height")
    if width is not None or height is not None:
        return width, height
    resolution = params.get("resolution") or params.get("parsed_resolution_wh")
    if resolution is None:
        return None, None
    try:
        return _parse_resolution(resolution)
    except (TypeError, ValueError):
        return None, None


def _int_param(params: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return int(value)
    return None


def _float_param(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return float(value)
    return None


def _bool_param(params: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return False


def _string_param(params: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _video_metadata_for_log(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "content_type": metadata.content_type,
        "frame_count": metadata.frame_count,
        "fps": metadata.fps,
        "duration_seconds": metadata.duration_seconds,
        "has_audio": metadata.has_audio,
        "audio_duration_seconds": metadata.audio_duration_seconds,
        "width": metadata.width,
        "height": metadata.height,
    }


def _discover_output_path(*, stdout: str, run_workspace: Path) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("output: "):
            return _resolve_output_path(line.removeprefix("output: ").strip(), run_workspace)

    metadata_path = _metadata_path_from_stdout(stdout, run_workspace)
    if metadata_path and metadata_path.exists():
        output = _output_from_metadata(metadata_path, run_workspace)
        if output:
            return output

    for output_path in _candidate_output_files(run_workspace):
        return str(output_path)

    return None


def _metadata_path_from_stdout(stdout: str, run_workspace: Path) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("metadata: "):
            raw_path = line.removeprefix("metadata: ").strip()
            return Path(_resolve_output_path(raw_path, run_workspace))
    return None


def _output_from_metadata(metadata_path: Path, run_workspace: Path) -> str | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for output in _flatten_outputs(metadata.get("outputs")):
        return _resolve_output_path(str(output), run_workspace)
    return None


def _flatten_outputs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        outputs: list[str] = []
        for item in value.values():
            outputs.extend(_flatten_outputs(item))
        return outputs
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        outputs = []
        for item in value:
            outputs.extend(_flatten_outputs(item))
        return outputs
    return []


def _resolve_output_path(value: str, run_workspace: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(run_workspace / path)


def _candidate_output_files(run_workspace: Path) -> list[Path]:
    candidates = [
        path
        for path in run_workspace.rglob("*")
        if path.is_file() and path.suffix.lower() in _OUTPUT_EXTENSIONS
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _failure_message(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> str:
    return (
        "VibeComfy task failed "
        f"task_id={resolved.task_id} "
        f"backend={resolved.backend.value} "
        f"template={resolved.template_id} "
        f"profile={_memory_profile_for_log(resolved)} "
        f"exit_code={exit_code} "
        f"stderr={_bounded(stderr)!r} "
        f"stdout={_bounded(stdout)!r}"
    )


def _log_failure(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> None:
    headless_logger.debug_block(
        "VIBECOMFY_FAILURE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": exit_code,
            "stderr": _bounded(stderr),
            "stdout": _bounded(stdout),
        },
        task_id=resolved.task_id,
    )


def _bounded(value: str | None, limit: int = _MAX_CAPTURE_CHARS) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


__all__ = ["handle_vibecomfy_resolved_task"]
