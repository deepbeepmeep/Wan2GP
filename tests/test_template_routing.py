from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "task_handlers"
    / "tasks"
    / "template_routing.py"
)


@pytest.fixture()
def routing():
    spec = importlib.util.spec_from_file_location("template_routing_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_supported_direct_routes_resolve_to_vibecomfy_templates(routing) -> None:
    z_image = routing.resolve_task_route(
        task_id="task-z",
        task_type="z_image_turbo",
        params={"prompt": "a small bright studio", "resolution": "1024x1024"},
        backend="vibecomfy",
    )
    assert z_image.route_key == "z_image_turbo"
    assert z_image.support_state == routing.RouteSupportState.VIBECOMFY_SUPPORTED
    assert z_image.template_id == "image/z_image"
    assert z_image.should_use_vibecomfy is True


@pytest.mark.parametrize(
    "task_type",
    [
        "qwen_image_2512",
        "qwen_image",
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    ],
)
def test_sprint_qwen_direct_routes_are_explicit_wgp_only(routing, task_type: str) -> None:
    resolved = routing.resolve_task_route(
        task_id="task-unsupported",
        task_type=task_type,
        params={"prompt": "edit this"},
        backend="vibecomfy",
    )

    assert resolved.route_key == task_type
    assert resolved.support_state == routing.RouteSupportState.WGP_ONLY
    assert resolved.template_id is None
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "wgp_only" in resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_routing_telemetry_fields_are_compact_and_stable(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="telemetry-task",
        task_type="z_image_turbo",
        params={
            "prompt": "telemetry",
            "resolution": "1024x1024",
            "override_profile": "3",
        },
        backend="vibecomfy",
    )

    assert routing.routing_telemetry_fields(resolved) == {
        "task_id": "telemetry-task",
        "task_type": "z_image_turbo",
        "route_key": "z_image_turbo",
        "backend": "vibecomfy",
        "template_id": "image/z_image",
        "support_state": "vibecomfy_supported",
        "memory_profile": "3",
    }


@pytest.mark.parametrize("task_type", ["travel_segment", "individual_travel_segment"])
def test_travel_children_derive_route_and_fail_closed_under_vibecomfy(
    routing, task_type: str
) -> None:
    resolved = routing.resolve_task_route(
        task_id=f"{task_type}-1",
        task_type=task_type,
        params={
            "model_name": "ltx2_distilled_19B",
            "travel_guidance": {"kind": "ltx_anchor"},
            "video_source": "/tmp/prefix.mp4",
            "override_profile": "3",
        },
        backend="vibecomfy",
    )

    assert (
        resolved.route_key
        == f"{task_type}__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-3"
    )
    assert resolved.support_state == routing.RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.template_id is None
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_template_independent_child_route_avoids_wan_vace_cocktail_keys(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="ltx-child",
        task_type="individual_travel_segment",
        params={
            "model_name": "ltx2_19B",
            "travel_mode": "ltx",
            "guidance_kind": "ltx_anchor",
            "individual_segment_params": {"base_prompt": "gentle camera drift"},
        },
        backend="vibecomfy",
    )

    combined_route_bits = " ".join(
        str(part)
        for part in (
            resolved.route_key,
            resolved.template_id,
            resolved.fail_closed_reason,
            resolved.params.get("model"),
        )
    ).lower()
    assert (
        resolved.route_key
        == "individual_travel_segment__model-ltx2__guidance-ltx_anchor__continuity-first_last__profile-default"
    )
    assert "vace" not in combined_route_bits
    assert "cocktail" not in combined_route_bits
    assert "wan_2_2" not in combined_route_bits


def test_travel_route_key_distinguishes_wan_vace_payload_context(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="wan-vace-child",
        task_type="travel_segment",
        params={
            "model_name": "wan_2_2_vace_14B",
            "travel_guidance": {"kind": "vace"},
            "video_source": "/tmp/svi-prefix.mp4",
            "wgp_profile": "default",
        },
        backend="vibecomfy",
    )

    assert (
        resolved.route_key
        == "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default"
    )
    assert resolved.support_state == routing.RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_wan_2_2_t2i_remains_wgp_only_even_when_vibecomfy_is_explicit(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="wan-image",
        task_type="wan_2_2_t2i",
        params={"prompt": "single frame", "resolution": "1024x1024"},
        backend="vibecomfy",
    )

    assert resolved.route_key == "wan_2_2_t2i"
    assert resolved.support_state == routing.RouteSupportState.WGP_ONLY
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "wgp_only" in resolved.fail_closed_reason


def test_unset_backend_and_wgp_backend_preserve_wgp_defaults(routing, monkeypatch) -> None:
    monkeypatch.delenv("REIGH_BACKEND", raising=False)
    unset_backend = routing.resolve_task_route(
        task_id="task-default",
        task_type="z_image_turbo",
        params={"resolution": "896x496"},
    )
    assert unset_backend.backend == routing.WorkerBackend.WGP
    assert unset_backend.fail_closed_reason is None
    assert unset_backend.should_use_vibecomfy is False

    explicit_wgp = routing.resolve_task_route(
        task_id="task-wgp",
        task_type="z_image_turbo",
        params={"resolution": "896x496"},
        backend="wgp",
    )
    assert explicit_wgp.backend == routing.WorkerBackend.WGP
    assert explicit_wgp.fail_closed_reason is None


def test_reigh_backend_vibecomfy_is_parsed_strictly(routing, monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    resolved = routing.resolve_task_route(
        task_id="task-env",
        task_type="z_image_turbo",
        params={"prompt": "env-selected"},
    )
    assert resolved.backend == routing.WorkerBackend.VIBECOMFY
    assert resolved.should_use_vibecomfy is True

    with pytest.raises(ValueError, match="Unsupported REIGH_BACKEND"):
        routing.parse_worker_backend("comfy")


def test_z_image_non_default_resolution_remains_vibecomfy_supported(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="z-image-non-default",
        task_type="z_image_turbo",
        params={"prompt": "non default", "resolution": "896x496"},
        backend="vibecomfy",
    )

    assert resolved.should_use_vibecomfy is True
    assert resolved.fail_closed_reason is None


def test_module_has_no_wgp_heavy_or_vibecomfy_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            imported_roots.add(node.module.split(".", 1)[0])

    assert "vibecomfy" not in imported_roots
    assert "source" not in imported_roots
    assert not any("wgp" in module.lower() for module in imported_modules)
