from __future__ import annotations

from datetime import datetime, timezone

from scripts.dual_run_compare.queue_contract import build_queue_contract_report
from scripts.dual_run_compare.runtime_metrics import (
    RUNTIME_METADATA_KEYS,
    RUNTIME_METRIC_KEYS,
    build_runtime_metrics_report,
    normalize_runtime_observations,
)


NOW = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


def _live_runtime_observation(route_key: str = "video_enhance") -> dict:
    return {
        "environment": "staging",
        "observed_at": "2026-05-06T11:30:00+00:00",
        "task_id": "task-runtime-123",
        "task_type": route_key,
        "route_key": route_key,
        "worker_backend": "api_orchestrator",
        "runtime": {
            "backend": "api_orchestrator",
            "pool": "api-owned",
            "worker_id": "api-worker-1",
            "metrics": {
                "latency": {"p50_ms": 1250, "p95_ms": 1800},
                "vram": {"peak_mb": 9216},
                "oom": False,
                "error_class": None,
            },
        },
        "selector_namespace": "canary",
        "selector_version": "2026-05-06",
        "status": "completed",
        "completion_evidence": {
            "handler": "complete_task/generation-handlers.ts",
            "status": "completed",
        },
        "billing_evidence": {
            "handler": "complete_task/billing.ts",
            "status": "charged",
        },
        "source_ref": {"kind": "system_logs", "id": "log-runtime-123"},
        "redaction": {"status": "redacted", "secret_scan": "passed"},
    }


def test_queue_contract_matches_current_task_handlers() -> None:
    report = build_queue_contract_report()

    assert report["status"] == "green"
    assert report["active_api_owned_route_count"] == 14
    assert report["routes"]["video_enhance"]["handler"] == "handle_video_enhance"
    assert report["routes"]["qwen_image"]["handler"] == "handle_qwen_image"
    assert report["routes"]["banodoco_render_timeline"]["status"] == "fallback"
    for route_key, route in report["routes"].items():
        if route["route_classification"] == "active_api_owned":
            assert route["status"] == "green", route_key
            assert route["runtime_owner"] == "api_orchestrator"
            assert {"task_handler_registry", "queue_contract_policy", "queue_payload_shape"} == {
                section["key"] for section in route["sections"]
            }


def test_runtime_metrics_record_distinct_missing_sections_by_landed_status() -> None:
    report = build_runtime_metrics_report()

    assert report["runtime_metric_keys"] == list(RUNTIME_METRIC_KEYS)
    assert report["runtime_metadata_keys"] == list(RUNTIME_METADATA_KEYS)
    assert report["routes"]["video_enhance"]["status"] == "red"
    assert report["routes"]["qwen_image"]["status"] == "wgp_only"
    assert report["routes"]["z_image_turbo"]["status"] == "red"
    assert report["routes"]["qwen_image_2512"]["status"] == "wgp_only"
    assert report["routes"]["qwen_image_edit"]["status"] == "wgp_only"
    assert report["routes"]["qwen_image_style"]["status"] == "wgp_only"
    assert report["routes"]["image_inpaint"]["status"] == "wgp_only"
    assert report["routes"]["annotated_image_edit"]["status"] == "wgp_only"
    assert report["routes"]["wan_2_2_t2i"]["status"] == "wgp_only"
    assert report["routes"]["banodoco_timeline_generate"]["status"] == "fallback"

    video_sections = {section["key"]: section for section in report["routes"]["video_enhance"]["sections"]}
    qwen_sections = {section["key"]: section for section in report["routes"]["qwen_image"]["sections"]}
    wan_sections = {section["key"]: section for section in report["routes"]["wan_2_2_t2i"]["sections"]}
    assert set(video_sections) == set(RUNTIME_METRIC_KEYS)
    assert set(qwen_sections) == set(RUNTIME_METRIC_KEYS)
    assert set(wan_sections) == set(RUNTIME_METRIC_KEYS)
    for metric_key in RUNTIME_METRIC_KEYS:
        assert video_sections[metric_key]["status"] == "missing_evidence"
        assert qwen_sections[metric_key]["status"] == "wgp_only"
        assert wan_sections[metric_key]["status"] == "wgp_only"


def test_runtime_metric_observations_are_recorded_separately() -> None:
    report = build_runtime_metrics_report(
        runtime_observations=[_live_runtime_observation()],
        now=NOW,
    )

    route = report["routes"]["video_enhance"]
    assert route["status"] == "green"
    assert route["latency"] == {"p50_ms": 1250, "p95_ms": 1800}
    assert route["vram"] == {"peak_mb": 9216}
    assert route["oom"] is False
    assert route["error_class"] is None
    assert route["runtime_observation"] == {
        "route_key": "video_enhance",
        "selector_namespace": "canary",
        "selector_version": "2026-05-06",
        "backend": "api_orchestrator",
        "pool": "api-owned",
        "worker_id": "api-worker-1",
        "task_id": "task-runtime-123",
        "source_ref": {"kind": "system_logs", "id": "log-runtime-123"},
    }
    assert {section["status"] for section in route["sections"]} == {"green"}


def test_runtime_metric_observations_reject_stale_or_incomplete_live_evidence() -> None:
    stale = _live_runtime_observation()
    stale["observed_at"] = "2026-05-05T11:59:59+00:00"
    missing_billing = _live_runtime_observation("flux_klein_edit")
    missing_billing["billing_evidence"] = {}

    report = build_runtime_metrics_report(runtime_observations=[stale, missing_billing], now=NOW)

    video = report["routes"]["video_enhance"]
    flux = report["routes"]["flux_klein_edit"]
    assert video["status"] == "red"
    assert flux["status"] == "red"
    assert "stale" in video["runtime_observation_validation_errors"][0]
    assert "billing_evidence" in flux["runtime_observation_validation_errors"][0]


def test_runtime_metric_normalizer_keeps_legacy_mapping_for_debug_but_marks_it_invalid() -> None:
    normalized = normalize_runtime_observations(
        {
            "video_enhance": {
                "latency": {"p95_ms": 1800},
                "vram": {"peak_mb": 9216},
                "oom": False,
                "error_class": None,
            }
        }
    )

    assert normalized["video_enhance"]["latency"] == {"p95_ms": 1800}
    assert normalized["video_enhance"]["route_key"] == "video_enhance"
    assert normalized["video_enhance"]["validation_errors"] == [
        "standardized recent live runtime observation is required"
    ]

    report = build_runtime_metrics_report(runtime_observations=normalized)
    assert report["routes"]["video_enhance"]["status"] == "red"
