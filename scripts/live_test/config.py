"""Configuration and static references for the live-test harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scripts.live_test as live_test_pkg
from scripts.live_test import ORCHESTRATOR_ROOT, WORKER_ROOT, ensure_orchestrator_imports

ensure_orchestrator_imports()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(WORKER_ROOT / ".env")


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class RunpodDefaults:
    gpu_type: str
    worker_image: str
    template_id: str
    volume_mount_path: str
    disk_size_gb: int
    container_disk_gb: int
    min_vcpu_count: int
    min_memory_gb: int
    storage_volumes: tuple[str, ...]
    ram_tiers: tuple[int, ...]


@dataclass(frozen=True)
class LiveTestEnv:
    reigh_live_test_token: str | None
    runpod_api_key: str | None
    supabase_url: str | None
    supabase_service_role_key: str | None


def _load_runpod_defaults() -> RunpodDefaults:
    try:
        from runpod_lifecycle import RunPodConfig

        cfg = RunPodConfig.from_env()
        return RunpodDefaults(
            gpu_type=cfg.gpu_type,
            worker_image=cfg.worker_image,
            template_id=cfg.template_id,
            volume_mount_path=cfg.volume_mount_path,
            disk_size_gb=cfg.disk_size_gb,
            container_disk_gb=cfg.container_disk_gb,
            min_vcpu_count=cfg.min_vcpu_count,
            min_memory_gb=cfg.min_memory_gb,
            storage_volumes=tuple(cfg.storage_volumes) or (cfg.storage_name,) if cfg.storage_name else tuple(cfg.storage_volumes),
            ram_tiers=tuple(cfg.ram_tiers),
        )
    except Exception:
        return RunpodDefaults(
            gpu_type="NVIDIA GeForce RTX 4090",
            worker_image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            template_id="runpod-torch-v240",
            volume_mount_path="/workspace",
            disk_size_gb=50,
            container_disk_gb=50,
            min_vcpu_count=8,
            min_memory_gb=32,
            storage_volumes=("Peter", "EU-NO-1", "EU-CZ-1", "EUR-IS-1"),
            ram_tiers=(72, 60, 48, 32, 16),
        )


ENV = LiveTestEnv(
    reigh_live_test_token=get_env("REIGH_LIVE_TEST_TOKEN"),
    runpod_api_key=get_env("RUNPOD_API_KEY"),
    supabase_url=get_env("SUPABASE_URL"),
    supabase_service_role_key=get_env("SUPABASE_SERVICE_ROLE_KEY"),
)

RUNPOD = _load_runpod_defaults()
RUNPOD_GPU_TYPE = RUNPOD.gpu_type
RUNPOD_WORKER_IMAGE = RUNPOD.worker_image
RUNPOD_TEMPLATE_ID = RUNPOD.template_id
RUNPOD_VOLUME_MOUNT_PATH = RUNPOD.volume_mount_path
LIVE_TEST_CONTAINER_DISK_GB = int(get_env("REIGH_LIVE_TEST_CONTAINER_DISK_GB", "200") or 200)
LIVE_TEST_DISK_SIZE_GB = int(get_env("REIGH_LIVE_TEST_DISK_SIZE_GB", "200") or 200)
RUNPOD_DISK_SIZE_GB = RUNPOD.disk_size_gb
RUNPOD_CONTAINER_DISK_GB = RUNPOD.container_disk_gb
RUNPOD_MIN_VCPU_COUNT = RUNPOD.min_vcpu_count
RUNPOD_MIN_MEMORY_GB = RUNPOD.min_memory_gb
RUNPOD_STORAGE_VOLUMES = RUNPOD.storage_volumes
RUNPOD_RAM_TIERS = RUNPOD.ram_tiers

TIMEOUT_IMAGE_SEC = 900
TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC = 1500
TIMEOUT_TRAVEL_ORCHESTRATOR_SEC = 2400

LTX_MODEL_ID = "ltx2_22B_distilled_1_1"

ANCHOR_IMAGE_A_URL = (
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/"
    "8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg"
)
ANCHOR_IMAGE_B_URL = (
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/"
    "8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-"
    "u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
)

FIXTURES: dict[str, Path] = {
    "qwen_image_basic": WORKER_ROOT
    / "artifacts/worker-matrix/20260316T135006000511Z/qwen_image_basic/prepared_input.json",
    "qwen_image_edit_basic": WORKER_ROOT
    / "artifacts/worker-matrix/20260316T135006000511Z/qwen_image_edit_basic/prepared_input.json",
    "z_image_turbo_i2i_basic": WORKER_ROOT
    / "artifacts/worker-matrix/20260316T135006000511Z/z_image_turbo_i2i_basic/prepared_input.json",
    "qwen_image_style_db_task": WORKER_ROOT
    / "artifacts/worker-matrix/20260316T141738351426Z/qwen_image_style_db_task/prepared_input.json",
    "wan22_i2v_individual_segment": WORKER_ROOT
    / "artifacts/worker-matrix/20260316T140127343314Z/wan22_i2v_individual_segment/prepared_input.json",
}


def load_fixture_json(case_name: str) -> dict[str, Any]:
    fixture_path = FIXTURES[case_name]
    import json

    return json.loads(fixture_path.read_text(encoding="utf-8"))


def __getattr__(name: str):
    if name in {
        "DatabaseClient",
        "RunpodLifecycleMixin",
        "SSHClient",
        "get_pod_ssh_details",
        "terminate_pod",
    }:
        return getattr(live_test_pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ANCHOR_IMAGE_A_URL",
    "ANCHOR_IMAGE_B_URL",
    "DatabaseClient",
    "ENV",
    "FIXTURES",
    "LTX_MODEL_ID",
    "ORCHESTRATOR_ROOT",
    "RunpodLifecycleMixin",
    "RUNPOD",
    "RUNPOD_CONTAINER_DISK_GB",
    "RUNPOD_DISK_SIZE_GB",
    "RUNPOD_GPU_TYPE",
    "RUNPOD_MIN_MEMORY_GB",
    "RUNPOD_MIN_VCPU_COUNT",
    "RUNPOD_RAM_TIERS",
    "RUNPOD_STORAGE_VOLUMES",
    "RUNPOD_TEMPLATE_ID",
    "RUNPOD_VOLUME_MOUNT_PATH",
    "RUNPOD_WORKER_IMAGE",
    "SSHClient",
    "TIMEOUT_IMAGE_SEC",
    "TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC",
    "TIMEOUT_TRAVEL_ORCHESTRATOR_SEC",
    "WORKER_ROOT",
    "get_env",
    "get_pod_ssh_details",
    "load_fixture_json",
    "require_env",
    "terminate_pod",
]
