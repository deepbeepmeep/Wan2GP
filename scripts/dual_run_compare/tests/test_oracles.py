from __future__ import annotations

from copy import deepcopy

from scripts.dual_run_compare.oracles import (
    GREEN,
    RED,
    FULL_CANARY_PRODUCT_EFFECT_SECTIONS,
    FULL_BILLING_IDEMPOTENCY_SECTIONS,
    REQUIRED_FULL_CANARY_SECTIONS,
    active_api_routes,
    default_billing_idempotency_observations,
    default_product_effect_observations,
    load_registry_snapshot,
    validate_billing_idempotency_oracles,
    validate_product_effect_oracles,
    validate_shared_path_oracles,
)


FULL_CANARY_ROUTES = {
    "video_enhance",
    "image-upscale",
    "animate_character",
    "flux_klein_edit",
}


def test_shared_path_oracles_cover_every_active_api_route() -> None:
    snapshot = load_registry_snapshot()
    report = validate_shared_path_oracles(snapshot)

    assert report["status"] == GREEN
    assert report["active_api_owned_route_count"] == 14
    assert set(active_api_routes(snapshot)) <= set(report["routes"])
    for route_key, route in active_api_routes(snapshot).items():
        route_report = report["routes"][route_key]
        assert route_report["status"] == GREEN
        section_keys = {section["key"] for section in route_report["sections"]}
        assert {"billing", "completion", "queue_contract", "shadow_side_effects"} <= section_keys
        assert route_report["completion_handler"] == "complete_task/generation-handlers.ts"
        assert route_report["billing_path"] == "complete_task/billing.ts"
        assert route_report["idempotency_policy"] == "complete_task_spend_ledger_idempotent_required"


def test_shared_completion_or_billing_regression_fails_non_canary_route() -> None:
    snapshot = load_registry_snapshot()
    mutated = deepcopy(snapshot)
    assert mutated["routes"]["qwen_image"]["canary_depth"] != "full_canary"
    mutated["routes"]["qwen_image"]["shared_path_policy"].pop("billing")

    report = validate_shared_path_oracles(mutated)

    assert report["status"] == RED
    assert report["routes"]["qwen_image"]["status"] == RED
    billing_section = next(section for section in report["routes"]["qwen_image"]["sections"] if section["key"] == "billing")
    assert billing_section["status"] == "missing_evidence"


def test_full_canaries_keep_product_billing_completion_depth() -> None:
    report = validate_shared_path_oracles()

    assert {
        route_key
        for route_key, route_report in report["routes"].items()
        if route_report.get("canary_depth") == "full_canary"
    } == FULL_CANARY_ROUTES
    for route_key in FULL_CANARY_ROUTES:
        sections = {section["key"]: section for section in report["routes"][route_key]["sections"]}
        assert set(sections["required_full_canary_sections"]["sections"]) == REQUIRED_FULL_CANARY_SECTIONS
        assert sections["payload_shape"]["status"] == GREEN
        assert sections["output_shape"]["status"] == GREEN
        assert sections["completion_handler"]["status"] == GREEN
        assert sections["billing_path"]["status"] == GREEN


def test_product_effect_oracles_cover_full_canaries_without_live_credentials() -> None:
    report = validate_product_effect_oracles()

    assert report["status"] == GREEN
    assert report["active_api_owned_route_count"] == 14
    for route_key in FULL_CANARY_ROUTES:
        route_report = report["routes"][route_key]
        assert route_report["status"] == GREEN
        assert set(route_report["required_product_effect_sections"]) == FULL_CANARY_PRODUCT_EFFECT_SECTIONS
        sections = {section["key"]: section for section in route_report["sections"]}
        assert set(sections) == FULL_CANARY_PRODUCT_EFFECT_SECTIONS
        assert all(section["status"] == GREEN for section in sections.values())
        assert sections["generation_created"]["observed"] is True
        assert sections["shadow_visible_record_absence"]["shadow_visible_record_count"] == 0
        assert sections["completion_handler"]["observed"] == "complete_task/generation-handlers.ts"


def test_product_effect_oracle_rejects_shadow_visible_record_for_full_canary() -> None:
    observations = default_product_effect_observations()
    observations["video_enhance"]["shadow_visible_record_count"] = 1

    report = validate_product_effect_oracles(observations=observations)

    assert report["status"] == RED
    assert report["routes"]["video_enhance"]["status"] == RED
    section = next(
        section
        for section in report["routes"]["video_enhance"]["sections"]
        if section["key"] == "shadow_visible_record_absence"
    )
    assert section["status"] == "red"
    assert section["shadow_visible_record_count"] == 1


def test_non_canary_product_effect_checks_keep_lightweight_completion_contract() -> None:
    report = validate_product_effect_oracles()

    for route_key, route_report in report["routes"].items():
        if route_key in FULL_CANARY_ROUTES or route_report["route_classification"] != "active_api_owned":
            continue
        assert route_report["status"] == GREEN
        assert route_report["required_product_effect_sections"] == ["completion_contract"]
        assert route_report["sections"] == [
            {
                "key": "completion_contract",
                "status": GREEN,
                "required": True,
                "completion_handler": "complete_task/generation-handlers.ts",
                "policy": "required_lightweight",
            }
        ]


def test_billing_idempotency_oracles_cover_active_routes_without_live_credentials() -> None:
    report = validate_billing_idempotency_oracles()

    assert report["status"] == GREEN
    assert report["active_api_owned_route_count"] == 14
    assert set(report["pending_refund_routes"]) == set(active_api_routes(load_registry_snapshot()))
    for route_key, route_report in report["routes"].items():
        if route_report["route_classification"] != "active_api_owned":
            continue
        sections = {section["key"]: section for section in route_report["sections"]}
        assert sections["lightweight_billing_policy"]["status"] == GREEN
        assert sections["spend_ledger_idempotency"]["status"] == GREEN
        assert sections["sub_task_billing_skip"]["status"] == GREEN
        assert sections["cancellation_cost_behavior"]["status"] == GREEN
        assert sections["refund_path_discovery"]["status"] == "pending_not_implemented"
        assert route_report["status"] == GREEN
        assert route_report["refund_status"] == "pending_not_implemented"
        if route_key == "video_enhance":
            assert set(route_report["required_billing_idempotency_sections"]) == FULL_BILLING_IDEMPOTENCY_SECTIONS
            assert sections["video_enhance_compound_cost"]["status"] == GREEN
            assert sections["video_enhance_compound_cost"]["observed_cost"] == 0.037883


def test_billing_oracle_rejects_missing_spend_ledger_idempotency() -> None:
    observations = default_billing_idempotency_observations()
    observations["spend_ledger_idempotency"]["skips_duplicate_spend"] = False

    report = validate_billing_idempotency_oracles(observations=observations)

    assert report["status"] == RED
    assert report["routes"]["qwen_image"]["status"] == RED
    section = next(
        section
        for section in report["routes"]["qwen_image"]["sections"]
        if section["key"] == "spend_ledger_idempotency"
    )
    assert section["status"] == "red"
    assert section["skips_duplicate_spend"] is False


def test_refund_is_never_green_without_executable_refund_ledger_path() -> None:
    observations = default_billing_idempotency_observations()
    observations["refund_path_discovery"] = {
        "ledger_table": "credits_ledger",
        "ledger_type": "refund",
        "enum_allows_refund": True,
        "executable_refund_ledger_path": False,
        "source": "update-task-status/cancellationBilling.ts",
    }

    report = validate_billing_idempotency_oracles(observations=observations)
    refund_section = next(
        section
        for section in report["routes"]["video_enhance"]["sections"]
        if section["key"] == "refund_path_discovery"
    )

    assert report["status"] == GREEN
    assert refund_section["status"] == "pending_not_implemented"
    assert refund_section["source"] == "update-task-status/cancellationBilling.ts"
