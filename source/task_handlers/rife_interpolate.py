"""RIFE interpolation task handler."""

import traceback
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from source.media.video import rife_interpolate_images_to_video
from source.core.log import task_logger
from source.utils.output_paths import (
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
)
from source.utils.resolution_utils import parse_resolution

def handle_rife_interpolate_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, task_queue=None):
    """Handles the 'rife_interpolate_images' task."""
    task_logger.essential("Starting RIFE interpolation task", task_id=task_id)

    input_image_path1_str = task_params_dict.get("input_image_path1")
    input_image_path2_str = task_params_dict.get("input_image_path2")
    output_video_path_str = task_params_dict.get("output_path")
    num_rife_frames = task_params_dict.get("frames")
    resolution_str = task_params_dict.get("resolution")
    custom_output_dir = task_params_dict.get("output_dir")

    required_params = {
        "input_image_path1": input_image_path1_str,
        "input_image_path2": input_image_path2_str,
        "output_path": output_video_path_str,
        "frames": num_rife_frames,
        "resolution": resolution_str
    }
    missing_params = [key for key, value in required_params.items() if value is None]
    if missing_params:
        error_msg = f"Missing required parameters for rife_interpolate_images: {', '.join(missing_params)}"
        task_logger.error(error_msg, task_id=task_id)
        return False, error_msg

    input_image1_path = Path(input_image_path1_str)
    input_image2_path = Path(input_image_path2_str)
    output_video_path = Path(output_video_path_str)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    generation_success = False
    output_location_to_db = None

    final_save_path_for_video, initial_db_location_for_rife = prepare_output_path_with_upload(
        task_id,
        f"{task_id}_interpolated.mp4",
        main_output_dir_base,
        task_type="rife_interpolate_images",
        custom_output_dir=custom_output_dir
    )
    output_video_path = final_save_path_for_video
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    task_logger.debug(f"[Task ID: {task_id}] Checking input image paths.")
    if not input_image1_path.is_file():
        task_logger.error(f"Input image 1 not found: {input_image1_path}", task_id=task_id)
        return False, f"Input image 1 not found: {input_image1_path}"
    if not input_image2_path.is_file():
        task_logger.error(f"Input image 2 not found: {input_image2_path}", task_id=task_id)
        return False, f"Input image 2 not found: {input_image2_path}"
    task_logger.debug(f"[Task ID: {task_id}] Input images found.")

    temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_rife_{task_id}_")

    try:
        pil_image_start = Image.open(input_image1_path).convert("RGB")
        pil_image_end = Image.open(input_image2_path).convert("RGB")

        task_logger.essential("Starting RIFE interpolation via source.media.video.", task_id=task_id)
        task_logger.debug(f"  Input 1: {input_image1_path}")
        task_logger.debug(f"  Input 2: {input_image2_path}")

        rife_success = rife_interpolate_images_to_video(
            image1=pil_image_start,
            image2=pil_image_end,
            num_frames=int(num_rife_frames),
            resolution_wh=parse_resolution(resolution_str),
            output_path=final_save_path_for_video,
            fps=16)

        if rife_success:
            if final_save_path_for_video.exists() and final_save_path_for_video.stat().st_size > 0:
                generation_success = True
                output_location_to_db = upload_and_get_final_output_location(
                    final_save_path_for_video, initial_db_location_for_rife)
                task_logger.essential(f"RIFE video saved to: {final_save_path_for_video.resolve()} (DB: {output_location_to_db})", task_id=task_id)
            else:
                task_logger.error(f"RIFE utility reported success, but output file is missing or empty: {final_save_path_for_video}", task_id=task_id)
                generation_success = False
        else:
            task_logger.error("RIFE interpolation failed.", task_id=task_id)
            generation_success = False

    except (OSError, ValueError, RuntimeError) as e:
        task_logger.error(f"Overall _handle_rife_interpolate_task failed: {e}", task_id=task_id)
        task_logger.debug(f"RIFE traceback: {traceback.format_exc()}", task_id=task_id)
        generation_success = False

    try:
        shutil.rmtree(temp_output_dir)
        task_logger.debug(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except OSError as e_clean:
        task_logger.warning(f"Failed to clean up temporary directory {temp_output_dir}: {e_clean}", task_id=task_id)

    return generation_success, output_location_to_db
