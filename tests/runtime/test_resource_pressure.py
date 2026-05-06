from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.models.lora.lora_utils import sweep_lora_cache
from source.runtime.worker import resource_pressure


def _write(path: Path, size: int, *, age_seconds: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if age_seconds:
        ts = time.time() - age_seconds
        os.utime(path, (ts, ts))
    return path


def test_lora_cache_sweep_applies_age_count_size_and_orphan_limits(tmp_path):
    cache = tmp_path / "loras"
    old = _write(cache / "old.safetensors", 10, age_seconds=7200)
    newest = _write(cache / "newest.safetensors", 12)
    middle = _write(cache / "middle.safetensors", 14, age_seconds=120)
    orphan = _write(cache / "partial.download", 9, age_seconds=7200)

    result = sweep_lora_cache(
        cache_dirs=[cache],
        max_age_seconds=3600,
        orphan_max_age_seconds=3600,
        max_files=1,
        max_bytes=20,
    )

    assert result.removed_files == 3
    assert result.removed_bytes == 33
    assert not old.exists()
    assert not middle.exists()
    assert not orphan.exists()
    assert newest.exists()
    assert result.remaining_files == 1
    assert result.remaining_bytes == 12


def test_claim_suppression_runs_cleanup_and_writes_quota_state(tmp_path, monkeypatch):
    lora_cache = tmp_path / "loras"
    _write(lora_cache / "old.safetensors", 10, age_seconds=7200)
    artifact_dir = tmp_path / "artifacts"
    _write(artifact_dir / "debug_bundle.tmp", 8, age_seconds=7200)
    disk_path = tmp_path / "disk"
    disk_path.mkdir()

    calls = {"count": 0}

    def fake_disk_usage(_path):
        calls["count"] += 1
        free = 100 if calls["count"] == 1 else 200
        return SimpleNamespace(total=1000, used=800, free=free)

    monkeypatch.setattr(resource_pressure.shutil, "disk_usage", fake_disk_usage)
    monkeypatch.setenv("REIGH_RESOURCE_PRESSURE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("REIGH_DISK_HEALTH_PATHS", str(disk_path))
    monkeypatch.setenv("REIGH_DISK_NEAR_FULL_PCT", "85")
    monkeypatch.setenv("REIGH_DISK_CLAIM_MIN_FREE_MB", "0")
    monkeypatch.setenv("REIGH_LORA_CACHE_DIRS", str(lora_cache))
    monkeypatch.setenv("REIGH_LORA_CACHE_MAX_AGE_SECONDS", "3600")
    monkeypatch.setenv("REIGH_ARTIFACT_CLEANUP_PATHS", str(artifact_dir))
    monkeypatch.setenv("REIGH_ARTIFACT_ORPHAN_MAX_AGE_SECONDS", "3600")

    result = resource_pressure.ensure_resources_for_claim("worker-1")

    assert result.allow_work is True
    assert result.status == "recovered"
    assert result.quota_alert is True
    assert result.cleanup["lora"]["removed_files"] == 1
    assert result.cleanup["artifacts"]["removed_files"] == 1
    assert not (lora_cache / "old.safetensors").exists()
    assert not (artifact_dir / "debug_bundle.tmp").exists()
    state = resource_pressure.read_resource_pressure_state("worker-1")
    assert state["resource_pressure_status"] == "recovered"
    assert state["resource_pressure_quota_alert"] is True


def test_claim_suppression_blocks_when_cleanup_cannot_recover(tmp_path, monkeypatch):
    disk_path = tmp_path / "disk"
    disk_path.mkdir()

    monkeypatch.setattr(
        resource_pressure.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=990, free=10),
    )
    monkeypatch.setenv("REIGH_RESOURCE_PRESSURE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("REIGH_DISK_HEALTH_PATHS", str(disk_path))
    monkeypatch.setenv("REIGH_DISK_NEAR_FULL_PCT", "90")
    monkeypatch.setenv("REIGH_DISK_CLAIM_MIN_FREE_MB", "1")
    monkeypatch.setenv("REIGH_LORA_CACHE_DIRS", str(tmp_path / "missing-loras"))
    monkeypatch.setenv("REIGH_ARTIFACT_CLEANUP_PATHS", str(tmp_path / "missing-artifacts"))

    result = resource_pressure.ensure_resources_for_claim("worker-2")

    assert result.allow_work is False
    assert result.action == "claim_suppressed"
    assert result.quota_alert is True
    assert resource_pressure.read_resource_pressure_state("worker-2")["resource_pressure_action"] == "claim_suppressed"


def test_write_check_blocks_only_after_cleanup_fails(tmp_path, monkeypatch):
    target = tmp_path / "mat" / "input.bin"
    target.parent.mkdir()
    monkeypatch.setattr(
        resource_pressure.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=950, free=50),
    )
    monkeypatch.setenv("REIGH_RESOURCE_PRESSURE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("REIGH_DISK_HEALTH_PATHS", str(target.parent))
    monkeypatch.setenv("REIGH_DISK_NEAR_FULL_PCT", "90")
    monkeypatch.setenv("REIGH_DISK_WRITE_MIN_FREE_MB", "1")
    monkeypatch.setenv("REIGH_DISK_WRITE_RESERVE_MB", "1")
    monkeypatch.setenv("REIGH_LORA_CACHE_DIRS", str(tmp_path / "missing-loras"))
    monkeypatch.setenv("REIGH_ARTIFACT_CLEANUP_PATHS", str(tmp_path / "missing-artifacts"))

    result = resource_pressure.ensure_resources_for_write(
        worker_id="worker-3",
        target_path=target,
        required_bytes=128,
    )

    assert result.allow_work is False
    assert result.action == "write_blocked"
