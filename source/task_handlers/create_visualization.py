"""
Handler for creating travel visualizations.

This module handles the 'create_visualization' task type which generates
debug/preview videos showing the generation process with:
- Input images with timeline markers
- Structure and output videos side-by-side
- Progress indicators and segment information
"""

import json
import tempfile
from pathlib import Path

from ..utils import (
    prepare_output_path_with_upload,
    upload_and_get_final_output_location as resolve_final_output_location,
)
from ..media.visualization.comparison import create_travel_visualization
from source.core.log import task_logger

def _handle_create_visualization_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    viz_task_id_str: str) -> tuple[bool, str]:
    """
    Handle a 'create_visualization' task.

    Args:
        task_params_from_db: Task parameters from database
        main_output_dir_base: Base directory for outputs
        viz_task_id_str: Task ID string

    Returns:
        Tuple of (success: bool, output_message: str)
    """
    task_logger.debug(f"[VIZ] Starting visualization task {viz_task_id_str}")

    try:
        # Extract parameters
        params = task_params_from_db.get("params", {})
        if isinstance(params, str):
            params = json.loads(params)

        # Required parameters
        output_video_path = params.get("output_video_path")
        structure_video_path = params.get("structure_video_path")
        input_image_paths = params.get("input_image_paths", [])
        segment_frames = params.get("segment_frames", [])

        # Optional parameters
        guidance_video_path = params.get("guidance_video_path")
        segment_prompts = params.get("segment_prompts")
        layout = params.get("layout", "triple")
        fps = params.get("fps", 16)
        show_guidance = params.get("show_guidance", False)
        structure_video_treatment = params.get("structure_video_treatment", "adjust")
        frame_overlaps = params.get("frame_overlaps")

        # Validate required parameters
        if not output_video_path:
            raise ValueError("Missing required parameter: output_video_path")
        if not structure_video_path:
            raise ValueError("Missing required parameter: structure_video_path")
        if not input_image_paths:
            raise ValueError("Missing required parameter: input_image_paths")
        if not segment_frames:
            raise ValueError("Missing required parameter: segment_frames")

        task_logger.debug(f"[VIZ] Task {viz_task_id_str}: Creating {layout} visualization")
        task_logger.debug(f"[VIZ]   Output video: {output_video_path}")
        task_logger.debug(f"[VIZ]   Structure video: {structure_video_path}")
        task_logger.debug(f"[VIZ]   Input images: {len(input_image_paths)}")
        task_logger.debug(f"[VIZ]   Segments: {len(segment_frames)}")

        # Create temporary output path
        temp_dir = Path(tempfile.mkdtemp(prefix=f"viz_{viz_task_id_str}_"))
        temp_output = temp_dir / f"visualization_{viz_task_id_str}.mp4"

        # Create visualization
        viz_path = create_travel_visualization(
            output_video_path=output_video_path,
            structure_video_path=structure_video_path,
            guidance_video_path=guidance_video_path,
            input_image_paths=input_image_paths,
            segment_frames=segment_frames,
            segment_prompts=segment_prompts,
            viz_output_path=str(temp_output),
            layout=layout,
            fps=fps,
            show_guidance=show_guidance,
            structure_video_treatment=structure_video_treatment,
            frame_overlaps=frame_overlaps
        )

        task_logger.debug(f"[VIZ] Task {viz_task_id_str}: Visualization created at {viz_path}")

        # Prepare final output path with upload
        final_path, initial_db_location = prepare_output_path_with_upload(
            filename=f"{viz_task_id_str}_visualization.mp4",
            task_id=viz_task_id_str,
            main_output_dir_base=main_output_dir_base,
            task_type="create_visualization")

        # Move/upload the visualization
        import shutil
        # Copy temp visualization to final location
        shutil.copy2(viz_path, final_path)

        # Handle upload and get final DB location
        output_location = resolve_final_output_location(
            local_file_path=Path(final_path),
            initial_db_location=initial_db_location)

        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        success_msg = f"Visualization created successfully: {output_location}"
        task_logger.debug(f"[VIZ] Task {viz_task_id_str}: {success_msg}")

        return True, output_location

    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as e:
        error_msg = f"[ERROR Task ID: {viz_task_id_str}] Visualization failed: {e}"
        task_logger.debug(error_msg, exc_info=True)
        return False, error_msg


# Public alias for cross-module use.
handle_create_visualization_task = _handle_create_visualization_task
