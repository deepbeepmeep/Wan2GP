from __future__ import annotations

import shutil
from pathlib import Path


IMPORT_SITE_TARGETS = {
    "download_lora_from_url": [
        "source.models.lora.lora_utils.download_lora_from_url",
    ],
    "download_file": [
        "source.utils.download_file",
        "source.utils.download_utils.download_file",
        "source.task_handlers.tasks.task_registry.download_file",
        "source.task_handlers.travel.segment_processor.download_file",
        "source.task_handlers.travel.predecessor_resolver.download_file",
    ],
    "download_image_if_url": [
        "source.utils.download_image_if_url",
        "source.utils.download_utils.download_image_if_url",
        "source.task_handlers.tasks.task_registry.download_image_if_url",
        "source.models.model_handlers.qwen_handler.download_image_if_url",
        "source.task_handlers.travel.segment_processor.download_image_if_url",
        "source.media.video.travel_guide.download_image_if_url",
        "source.utils.frame_utils.download_image_if_url",
    ],
}


def _ensure_fake_lora() -> str:
    fake_lora = Path("/tmp/preview/fake_lora.safetensors")
    fake_lora.parent.mkdir(parents=True, exist_ok=True)
    if not fake_lora.exists():
        fake_lora.write_bytes(b"preview-lora")
    return str(fake_lora)


def build_network_spoofs(asset_paths: dict[str, str] | None = None) -> dict[str, object]:
    from scripts.preview.assets import ensure_assets

    resolved_assets = asset_paths or ensure_assets(Path(__file__).resolve().parent)
    fake_lora_path = _ensure_fake_lora()

    def download_lora_from_url(url, task_id, model_type=None):
        _ = url, task_id, model_type
        return fake_lora_path

    def download_file(url, dest_folder=None, filename=None):
        suffix = str(url).lower()
        source_path = Path(resolved_assets["image"] if suffix.endswith((".png", ".jpg", ".jpeg", ".webp")) else resolved_assets["video"])
        if dest_folder is not None and filename is not None:
            target_path = Path(dest_folder) / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            return str(target_path.resolve())
        return str(source_path.resolve())

    def download_image_if_url(image_url_or_path, download_target_dir=None, task_id_for_logging="generic_task", debug_mode=False, descriptive_name=None):
        _ = download_target_dir, task_id_for_logging, debug_mode, descriptive_name
        candidate = Path(str(image_url_or_path))
        if candidate.exists():
            return str(candidate.resolve())
        if str(image_url_or_path).startswith("file://"):
            file_path = Path(str(image_url_or_path)[7:])
            if file_path.exists():
                return str(file_path.resolve())
        return resolved_assets["image"]

    return {
        "download_lora_from_url": download_lora_from_url,
        "download_file": download_file,
        "download_image_if_url": download_image_if_url,
        "import_site_targets": IMPORT_SITE_TARGETS,
    }
