"""Fixture coverage for checked-in LoRA module manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.models.lora.module_manifest import (
    MANIFEST_DIR,
    LoRAModuleManifestError,
    architecture_from_manifest_filename,
    load_all_module_manifests,
    load_module_manifest,
    parse_module_manifest,
)


REQUIRED_ARCHITECTURES = {
    "qwen_image_20B",
    "qwen_image_2512_20B",
    "qwen_image_edit_20B",
    "qwen_image_edit_plus_20B",
    "qwen_image_edit_plus2_20B",
    "wan_2_2_i2v_lightning_baseline_2_2_2",
    "wan_2_2_vace_lightning_baseline_2_2_2",
    "ltx2_19B",
    "ltx2_distilled",
    "ltx2_22B",
    "ltx2_22B_distilled",
}


def test_all_required_lora_module_manifests_exist_and_parse() -> None:
    manifests = load_all_module_manifests()

    assert REQUIRED_ARCHITECTURES <= set(manifests)
    for architecture in REQUIRED_ARCHITECTURES:
        manifest = manifests[architecture]
        assert manifest.architecture == architecture
        assert manifest.source_path.name == f"module_names_{architecture}.json"
        assert manifest.module_names
        assert len(manifest.module_names) == len(set(manifest.module_names))


def test_every_checked_in_manifest_matches_filename_architecture() -> None:
    for path in sorted(MANIFEST_DIR.glob("module_names_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = parse_module_manifest(data, source_path=path)
        assert manifest.architecture == architecture_from_manifest_filename(path)


def test_load_module_manifest_missing_architecture_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(LoRAModuleManifestError, match="missing LoRA module manifest"):
        load_module_manifest("missing_arch", manifest_dir=tmp_path)


def test_parse_module_manifest_rejects_filename_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "module_names_qwen_image_20B.json"
    data = {
        "schema_version": 1,
        "architecture": "wan_2_2_i2v_lightning_baseline_2_2_2",
        "module_names": ["diffusion_model"],
    }

    with pytest.raises(LoRAModuleManifestError, match="does not match filename"):
        parse_module_manifest(data, source_path=path)
