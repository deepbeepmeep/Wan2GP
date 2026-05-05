from __future__ import annotations

import json
from pathlib import Path

from scripts.dual_run_compare.route_keys import cohort_e_route_key, direct_route_key
from scripts.dual_run_compare.thresholds import DEFAULT_PATH, Thresholds


DUAL_RUN_DIR = DEFAULT_PATH.parent
GOLDEN_DIR = DUAL_RUN_DIR / "golden"
SEED_DIR = DUAL_RUN_DIR / "fixtures" / "golden_seed_payloads"

REQUIRED_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "threshold_version",
    "route_key",
    "canonical_route_key",
    "cohort",
    "route_key_mode",
    "task_type",
    "media_class",
    "calibration_status",
    "source_fixture_status",
    "source_fixture_ref",
    "seed_payload_ref",
    "expected_output_contract",
    "metrics_profile",
    "owner",
}


def _load_manifests() -> dict[str, dict]:
    return {
        path.parent.name: json.loads(path.read_text())
        for path in GOLDEN_DIR.glob("*/manifest.json")
    }


def test_yaml_and_golden_manifests_are_one_to_one() -> None:
    thresholds = Thresholds.load(strict=True)
    manifests = _load_manifests()
    assert set(manifests) == set(thresholds.routes)
    for route_key, manifest in manifests.items():
        route = thresholds.routes[route_key]
        assert manifest["route_key"] == route_key
        assert manifest["canonical_route_key"] == route_key
        assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
        assert manifest["threshold_version"] == thresholds.raw["version"]
        assert manifest["source_fixture_status"] == route["source_fixture_status"]
        assert manifest["source_fixture_ref"] == route["source_fixture_ref"]
        if route["route_key_mode"] == "direct":
            assert direct_route_key(route["task_type"]) == route_key


def test_cohort_e_manifest_directories_are_dimensional() -> None:
    thresholds = Thresholds.load(strict=True)
    for route_key, route in thresholds.routes.items():
        if route["cohort"] != "E":
            continue
        assert "__model-" in route_key
        assert "__guidance-" in route_key
        assert "__continuity-" in route_key
        assert "__profile-" in route_key
        manifest = json.loads((GOLDEN_DIR / route_key / "manifest.json").read_text())
        for field in ("model_family", "guidance_kind", "continuity_case", "profile"):
            assert manifest[field] == route[field]
        assert (
            cohort_e_route_key(
                task_type=route["task_type"],
                model_family=route["model_family"],
                guidance_kind=route["guidance_kind"],
                continuity_case=route["continuity_case"],
                profile=route["profile"],
            )
            == route_key
        )


def test_i2v_travel_routes_are_not_vace_keyed() -> None:
    thresholds = Thresholds.load(strict=True)
    for route_key, route in thresholds.routes.items():
        if route.get("model_name") == "wan_2_2_i2v_lightning_baseline_2_2_2":
            assert route["model_family"] == "wan22_i2v"
            assert "vace" not in route_key


def test_seed_fixture_required_routes_are_explicit_and_local() -> None:
    thresholds = Thresholds.load(strict=True)
    for route_key, route in thresholds.routes.items():
        manifest = json.loads((GOLDEN_DIR / route_key / "manifest.json").read_text())
        if route["source_fixture_status"] == "seed_fixture_required":
            assert manifest["source_fixture_status"] == "seed_fixture_required"
            seed_ref = manifest["seed_payload_ref"]
            assert seed_ref.startswith("fixtures/golden_seed_payloads/")
            assert (DUAL_RUN_DIR / seed_ref).is_file()
        else:
            assert manifest["source_fixture_status"] == "worker_matrix_case"
            assert manifest["seed_payload_ref"] is None
