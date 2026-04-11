"""Extract frame task handler."""

import traceback
from pathlib import Path

import cv2
import source.media.video.api as video_api

from source.core.db.task_polling import get_abs_path_from_db_path, get_task_output_location_from_db
from source.core.log import task_logger
from source.utils.orchestrator_utils import report_orchestrator_failure
from source.utils.output_paths import (
    prepare_output_path_with_upload,
    upload_and_get_final_output_location,
)

def handle_extract_frame_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str):
    """Handles the 'extract_frame' task."""
    input_video_task_id = task_params_dict.get("input_video_task_id")
    frame_index = task_params_dict.get("frame_index", 0)  # Default to first frame
    custom_output_dir = task_params_dict.get("output_dir")

    if not input_video_task_id:
        msg = f"Task {task_id}: Missing 'input_video_task_id' in payload."
        report_orchestrator_failure(task_params_dict, msg)
        return False, msg

    try:
        video_path_from_db = get_task_output_location_from_db(input_video_task_id)
        if not video_path_from_db:
            msg = f"Task {task_id}: Could not find output location for dependency task {input_video_task_id}."
            report_orchestrator_failure(task_params_dict, msg)
            return False, msg

        video_abs_path = get_abs_path_from_db_path(video_path_from_db)
        if not video_abs_path:
            msg = f"Task {task_id}: Could not resolve or find video file from DB path '{video_path_from_db}'."
            report_orchestrator_failure(task_params_dict, msg)
            return False, msg

        output_filename = f"{task_id}_frame_{frame_index}.png"
        final_save_path, initial_db_location = prepare_output_path_with_upload(
            task_id,
            output_filename,
            main_output_dir_base,
            task_type="extract_frame",
            custom_output_dir=custom_output_dir
        )

        cap = cv2.VideoCapture(str(video_abs_path))
        if not cap.isOpened():
            msg = f"Task {task_id}: Could not open video file {video_abs_path}"
            report_orchestrator_failure(task_params_dict, msg)
            return False, msg
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        success = video_api.save_frame_from_video(
            input_video_path=video_abs_path,
            frame_index=frame_index,
            output_image_path=final_save_path,
            resolution=(width, height)
        )

        if success:
            final_db_location = upload_and_get_final_output_location(
                final_save_path, initial_db_location)
            return True, final_db_location
        else:
            msg = f"Task {task_id}: save_frame_from_video utility failed for video {video_abs_path}."
            report_orchestrator_failure(task_params_dict, msg)
            return False, msg

    except (OSError, ValueError, RuntimeError) as e:
        error_msg = f"Task {task_id}: Failed during frame extraction: {e}"
        task_logger.error(error_msg, task_id=task_id)
        task_logger.debug(f"Extract frame traceback: {traceback.format_exc()}", task_id=task_id)
        report_orchestrator_failure(task_params_dict, error_msg)
        return False, str(e)
