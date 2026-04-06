import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable

from shared.utils.audio_metadata import save_audio_metadata
from shared.utils.audio_video import save_image_metadata
from shared.utils.video_metadata import save_video_metadata


def _ensure_creation_metadata(configs: Any) -> Any:
    if not isinstance(configs, dict):
        return configs
    saved_configs = configs.copy()
    if "creation_date" in saved_configs and "creation_timestamp" in saved_configs:
        return saved_configs
    recorded_time = time.time()
    if "creation_date" not in saved_configs:
        saved_configs["creation_date"] = datetime.fromtimestamp(recorded_time).isoformat(timespec="seconds")
    if "creation_timestamp" not in saved_configs:
        saved_configs["creation_timestamp"] = int(recorded_time)
    return saved_configs


def record_file_metadata(video_path: str | list[str], configs: Any, is_image: bool, audio_only: bool, gen: dict[str, Any], *, get_processed_queue: Callable[[dict[str, Any]], tuple[list[Any], list[Any], list[Any], list[Any]]], metadata_choice: str = "metadata", embedded_images: Any = None, replace_last_file: bool = False, lock: Any = None, verbose_level: int = 0) -> None:
    file_list, file_settings_list, audio_file_list, audio_file_settings_list = get_processed_queue(gen)
    paths = [video_path] if not isinstance(video_path, list) else video_path
    queue_lock = lock if lock is not None else nullcontext()
    for no, path in enumerate(paths):
        previous_path = None
        saved_configs = _ensure_creation_metadata(configs)
        if configs is not None:
            if metadata_choice == "json":
                with open(os.path.splitext(path)[0] + ".json", "w") as f:
                    json.dump(saved_configs, f, indent=4)
            elif metadata_choice == "metadata":
                if audio_only:
                    save_audio_metadata(path, saved_configs)
                elif is_image:
                    save_image_metadata(path, saved_configs)
                else:
                    save_video_metadata(path, saved_configs, embedded_images)
        if verbose_level > 0:
            if audio_only:
                print(f"New audio file saved to Path: {path}")
            elif is_image:
                print(f"New image saved to Path: {path}")
            else:
                print(f"New video saved to Path: {path}")
        with queue_lock:
            if audio_only:
                audio_file_list.append(path)
                audio_file_settings_list.append(saved_configs)
            else:
                if replace_last_file and not is_image and no == 0 and len(file_list) > 0:
                    previous_path = file_list[-1]
                    file_list[-1] = path
                    file_settings_list[-1] = saved_configs
                else:
                    file_list.append(path)
                    file_settings_list.append(saved_configs)
            gen["last_was_audio"] = audio_only
        if previous_path is not None and previous_path != path:
            if metadata_choice == "json":
                previous_json_path = os.path.splitext(previous_path)[0] + ".json"
                if os.path.isfile(previous_json_path):
                    os.remove(previous_json_path)
            if os.path.isfile(previous_path):
                os.remove(previous_path)
