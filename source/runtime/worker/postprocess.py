"""Worker post-processing helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from source.core.log import headless_logger
from source.task_handlers.tasks.task_types import is_wgp_task
from source.utils.output_paths import prepare_output_path


def move_wgp_output_to_task_type_dir(
    *,
    output_path: str,
    task_type: str,
    task_id: str,
    main_output_dir_base: Path,
) -> str:
    if not is_wgp_task(task_type):
        return output_path
    output_file = Path(output_path)
    if not output_file.exists():
        return output_path
    if output_file.parent.resolve() != main_output_dir_base.resolve():
        return output_path
    new_path, _ = prepare_output_path(
        task_id=task_id,
        filename=output_file.name,
        main_output_dir_base=main_output_dir_base,
        task_type=task_type,
    )
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_file), str(new_path))
    headless_logger.debug(f"Moved WGP output to {new_path}", task_id=task_id)
    return str(new_path)
