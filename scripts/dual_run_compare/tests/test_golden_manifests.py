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

SPRINT5_DIRECT_WGP_ONLY_ROUTES = {
    "qwen_image",
    "qwen_image_2512",
    "qwen_image_edit",
    "qwen_image_style",
    "image_inpaint",
    "annotated_image_edit",
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


def test_sprint5_direct_routes_have_independent_wgp_only_policy_evidence() -> None:
    thresholds = Thresholds.load(strict=True)
    manifests = _load_manifests()
    registry = json.loads((DUAL_RUN_DIR / "fixtures" / "non_rayworker" / "registry_snapshot.json").read_text())

    assert SPRINT5_DIRECT_WGP_ONLY_ROUTES <= set(thresholds.routes)
    assert SPRINT5_DIRECT_WGP_ONLY_ROUTES <= set(manifests)
    assert SPRINT5_DIRECT_WGP_ONLY_ROUTES <= set(registry["routes"])

    for route_key in sorted(SPRINT5_DIRECT_WGP_ONLY_ROUTES):
        route = thresholds.routes[route_key]
        manifest = manifests[route_key]
        registry_route = registry["routes"][route_key]

        assert route["calibration_status"] == "wgp_only"
        assert manifest["calibration_status"] == "wgp_only"
        assert manifest["route_policy"] == "wgp_only"
        assert manifest["policy_evidence_ref"] == "fixtures/non_rayworker/registry_snapshot.json"
        assert manifest["task_type"] == route["task_type"] == registry_route["task_type"]
        assert registry_route["report_status_policy"] == "wgp_only"
        assert registry_route["landed_status"] == "sprint5_wgp_only"

    assert manifests["qwen_image"]["task_type"] == "qwen_image"
    assert manifests["qwen_image_2512"]["task_type"] == "qwen_image_2512"
    assert manifests["qwen_image"]["seed_payload_ref"] != manifests["qwen_image_2512"]["seed_payload_ref"]
    assert thresholds.routes["qwen_image"]["source_fixture_ref"] != thresholds.routes["qwen_image_2512"]["source_fixture_ref"]


def test_sprint5_wgp_only_direct_routes_do_not_require_live_parity_observations() -> None:
    thresholds = Thresholds.load(strict=True)
    manifests = _load_manifests()
    registry = json.loads((DUAL_RUN_DIR / "fixtures" / "non_rayworker" / "registry_snapshot.json").read_text())

    for route_key in sorted(SPRINT5_DIRECT_WGP_ONLY_ROUTES):
        assert thresholds.routes[route_key]["calibration_status"] == "wgp_only"
        assert manifests[route_key]["route_policy"] == "wgp_only"
        assert registry["routes"][route_key]["report_status_policy"] == "wgp_only"
        assert registry["routes"][route_key]["report_status_policy"] != "red_or_green_required"
