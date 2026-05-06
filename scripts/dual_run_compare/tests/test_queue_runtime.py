from __future__ import annotations

from scripts.dual_run_compare.queue_contract import build_queue_contract_report
from scripts.dual_run_compare.runtime_metrics import RUNTIME_METRIC_KEYS, build_runtime_metrics_report


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
        runtime_observations={
            "video_enhance": {
                "latency": {"p50_ms": 1250, "p95_ms": 1800},
                "vram": {"peak_mb": 9216},
                "oom": False,
                "error_class": None,
            }
        }
    )

    route = report["routes"]["video_enhance"]
    assert route["status"] == "green"
    assert route["latency"] == {"p50_ms": 1250, "p95_ms": 1800}
    assert route["vram"] == {"peak_mb": 9216}
    assert route["oom"] is False
    assert route["error_class"] is None
    assert {section["status"] for section in route["sections"]} == {"green"}
