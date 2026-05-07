from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest


def _module(name: str, **attrs):
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _noop(*_args, **_kwargs):
    return None


def _import_task_registry(monkeypatch):
    def _fake_handler(*_args, **_kwargs):
        return False, "stubbed"

    class _GenerationTask:
        def __init__(self, id, model, prompt, parameters):
            self.id = id
            self.model = model
            self.prompt = prompt
            self.parameters = parameters

    stubs = {
        "source.task_handlers.tasks.task_conversion": _module(
            "source.task_handlers.tasks.task_conversion",
            db_task_to_generation_task=lambda *_args, **_kwargs: SimpleNamespace(
                id="stub", parameters={}
            ),
        ),
        "source.task_handlers.extract_frame": _module(
            "source.task_handlers.extract_frame",
            handle_extract_frame_task=_fake_handler,
        ),
        "source.task_handlers.rife_interpolate": _module(
            "source.task_handlers.rife_interpolate",
            handle_rife_interpolate_task=_fake_handler,
        ),
        "source.models.comfy.comfy_handler": _module(
            "source.models.comfy.comfy_handler",
            handle_comfy_task=_fake_handler,
        ),
        "source.task_handlers.travel.orchestrator": _module(
            "source.task_handlers.travel.orchestrator",
            handle_travel_orchestrator_task=_fake_handler,
        ),
        "source.task_handlers.travel.stitch": _module(
            "source.task_handlers.travel.stitch",
            handle_travel_stitch_task=_fake_handler,
        ),
        "source.task_handlers.magic_edit": _module(
            "source.task_handlers.magic_edit",
            handle_magic_edit_task=_fake_handler,
        ),
        "source.task_handlers.join.generation": _module(
            "source.task_handlers.join.generation",
            handle_join_clips_task=_fake_handler,
        ),
        "source.task_handlers.join.final_stitch": _module(
            "source.task_handlers.join.final_stitch",
            handle_join_final_stitch=_fake_handler,
        ),
        "source.task_handlers.join.orchestrator": _module(
            "source.task_handlers.join.orchestrator",
            handle_join_clips_orchestrator_task=_fake_handler,
        ),
        "source.task_handlers.edit_video_orchestrator": _module(
            "source.task_handlers.edit_video_orchestrator",
            handle_edit_video_orchestrator_task=_fake_handler,
        ),
        "source.task_handlers.inpaint_frames": _module(
            "source.task_handlers.inpaint_frames",
            handle_inpaint_frames_task=_fake_handler,
        ),
        "source.task_handlers.create_visualization": _module(
            "source.task_handlers.create_visualization",
            handle_create_visualization_task=_fake_handler,
        ),
        "source.task_handlers.travel.segment_processor": _module(
            "source.task_handlers.travel.segment_processor",
            TravelSegmentProcessor=object,
            TravelSegmentContext=object,
        ),
        "source.task_handlers.travel.predecessor_resolver": _module(
            "source.task_handlers.travel.predecessor_resolver",
            download_predecessor_video=_noop,
            extract_prefix_video=_noop,
            resolve_generation_id=_noop,
            resolve_segment_predecessor=_noop,
        ),
        "source.media.video": _module(
            "source.media.video",
            extract_last_frame_as_image=_noop,
        ),
        "source.core.db.task_polling": _module(
            "source.core.db.task_polling",
            get_task_params=lambda *_args, **_kwargs: {},
        ),
        "source.utils.download_utils": _module(
            "source.utils.download_utils",
            download_file=_noop,
            download_image_if_url=lambda value, *_args, **_kwargs: value,
        ),
        "source.task_handlers.queue.task_queue": _module(
            "source.task_handlers.queue.task_queue",
            GenerationTask=_GenerationTask,
        ),
        "source.task_handlers.travel.chaining": _module(
            "source.task_handlers.travel.chaining",
            handle_travel_chaining_after_wgp=_fake_handler,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("source.task_handlers.tasks.task_registry", None)
    import source.task_handlers.tasks as tasks_pkg

    if hasattr(tasks_pkg, "task_registry"):
        delattr(tasks_pkg, "task_registry")

    from source.task_handlers.tasks import task_registry

    return task_registry


class _Queue:
    def __init__(self, *, fail_on_submit: bool = False):
        self.fail_on_submit = fail_on_submit
        self.submitted = []

    def submit_task(self, task):
        if self.fail_on_submit:
            raise AssertionError("WGP queue submit should not be called")
        self.submitted.append(task)
        return task.id

    def get_task_status(self, task_id):
        return SimpleNamespace(status="completed", result_path=f"/tmp/{task_id}.png")


class _LogCapture:
    def __init__(self):
        self.debug_blocks = []
        self.errors = []

    def debug_block(self, title, payload, **kwargs):
        self.debug_blocks.append((title, payload, kwargs))

    def error(self, message, **kwargs):
        self.errors.append((message, kwargs))


def _context(queue: _Queue, tmp_path):
    return {
        "task_id": "task-1",
        "task_params_dict": {"prompt": "a quiet studio"},
        "task_queue": queue,
        "wan2gp_path": "/tmp/wan",
        "debug_mode": False,
        "colour_match_videos": False,
        "mask_active_frames": False,
        "main_output_dir_base": tmp_path,
    }


def _travel_ctx(tmp_path):
    return SimpleNamespace(
        individual_params={},
        segment_params={"override_profile": "3"},
        orchestrator_details={"travel_guidance": {"kind": "ltx_anchor"}},
        orchestrator_task_id_ref="orchestrator-1",
        orchestrator_run_id="run-1",
        segment_idx=0,
    )


def test_wgp_default_direct_route_preserves_builder_and_queue(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.delenv("REIGH_BACKEND", raising=False)
    built = []

    def _builder(params, task_id, task_type, wan2gp_path, debug_mode):
        built.append((params, task_id, task_type, wan2gp_path, debug_mode))
        return SimpleNamespace(id=task_id, parameters={})

    monkeypatch.setattr(task_registry, "db_task_to_generation_task", _builder)
    queue = _Queue()

    ok, result = task_registry.TaskRegistry._handle_direct_queue_task(
        "z_image_turbo",
        _context(queue, tmp_path),
    )

    assert ok is True
    assert result == "/tmp/task-1.png"
    assert built and built[0][2] == "z_image_turbo"
    assert len(queue.submitted) == 1


@pytest.mark.parametrize(
    "task_type",
    [
        "z_image_turbo",
        "z_image_turbo_i2i",
        "wan_2_2_t2i",
        "qwen_image_2512",
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    ],
)
def test_vibecomfy_supported_direct_routes_bypass_wgp_queue(
    monkeypatch, tmp_path, task_type: str
):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")

    def _builder(*_args, **_kwargs):
        raise AssertionError("WGP builder should not run")

    def _adapter(resolved, main_output_dir_base):
        assert resolved.route_key == task_type
        assert resolved.should_use_vibecomfy is True
        assert main_output_dir_base == tmp_path
        return True, f"{task_type}.png"

    monkeypatch.setattr(task_registry, "db_task_to_generation_task", _builder)
    monkeypatch.setattr(
        "source.task_handlers.tasks.task_execution._load_vibecomfy_handler",
        lambda: _adapter,
    )
    context = _context(_Queue(fail_on_submit=True), tmp_path)
    context["task_params_dict"] = {"prompt": "direct", "resolution": "1024x1024"}
    ok, result = task_registry.TaskRegistry._handle_direct_queue_task(task_type, context)

    assert ok is True
    assert result == f"{task_type}.png"


def test_vibecomfy_direct_selection_emits_routing_card(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    logger = _LogCapture()

    def _builder(*_args, **_kwargs):
        raise AssertionError("WGP builder should not run")

    def _adapter(_resolved, _main_output_dir_base):
        return True, "selected.png"

    monkeypatch.setattr(task_registry, "db_task_to_generation_task", _builder)
    monkeypatch.setattr(
        "source.task_handlers.tasks.task_execution._load_vibecomfy_handler",
        lambda: _adapter,
    )
    monkeypatch.setattr(
        "source.task_handlers.tasks.task_execution.headless_logger",
        logger,
    )
    context = _context(_Queue(fail_on_submit=True), tmp_path)
    context["task_params_dict"] = {
        "prompt": "direct telemetry",
        "resolution": "1024x1024",
        "override_profile": "3",
    }

    ok, result = task_registry.TaskRegistry._handle_direct_queue_task(
        "z_image_turbo",
        context,
    )

    assert ok is True
    assert result == "selected.png"
    routing_cards = [
        payload for title, payload, _kwargs in logger.debug_blocks
        if title == "VIBECOMFY_ROUTING"
    ]
    assert routing_cards
    card = routing_cards[0]
    assert card["task_id"] == "task-1"
    assert card["task_type"] == "z_image_turbo"
    assert card["route_key"] == "z_image_turbo"
    assert card["backend"] == "vibecomfy"
    assert card["template_id"] == "image/z_image"
    assert card["support_state"] == "vibecomfy_supported"
    assert card["memory_profile"] == "3"
    assert card["decision"] == "vibecomfy_adapter"


def test_vibecomfy_wgp_only_direct_route_fails_closed_without_wgp_fallback(
    monkeypatch, tmp_path
):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")

    def _builder(*_args, **_kwargs):
        raise AssertionError("WGP builder should not run")

    monkeypatch.setattr(task_registry, "db_task_to_generation_task", _builder)

    ok, message = task_registry.TaskRegistry._handle_direct_queue_task(
        "qwen_image",
        _context(_Queue(fail_on_submit=True), tmp_path),
    )

    assert ok is False
    assert message
    assert "fail-closed" in message
    assert "wgp_only" in message


@pytest.mark.parametrize(
    "task_type",
    [
        "qwen_image",
    ],
)
def test_sprint_qwen_routes_fail_closed_without_wgp_fallback(monkeypatch, tmp_path, task_type: str):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")

    def _builder(*_args, **_kwargs):
        raise AssertionError("WGP builder should not run")

    monkeypatch.setattr(task_registry, "db_task_to_generation_task", _builder)
    context = _context(_Queue(fail_on_submit=True), tmp_path)
    context["task_params_dict"] = {
        "prompt": "direct",
        "resolution": "768x768",
        "image": "https://example.com/input.png",
        "image_url": "https://example.com/input.png",
        "mask_url": "https://example.com/mask.png",
    }

    ok, message = task_registry.TaskRegistry._handle_direct_queue_task(task_type, context)

    assert ok is False
    assert message
    assert "fail-closed" in message
    assert task_type in message
    assert "wgp_only" in message


@pytest.mark.parametrize(
    ("is_standalone", "expected_route"),
    [
        (
            False,
            "travel_segment__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-3",
        ),
        (
            True,
            "individual_travel_segment__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-3",
        ),
    ],
)
def test_travel_child_selector_fails_closed_before_queue_submit(
    monkeypatch, tmp_path, is_standalone: bool, expected_route: str
):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")

    monkeypatch.setattr(task_registry, "log_ram_usage", _noop)
    monkeypatch.setattr(task_registry, "_resolve_segment_context", lambda *_args: _travel_ctx(tmp_path))
    monkeypatch.setattr(
        task_registry,
        "_resolve_generation_inputs",
        lambda *_args: SimpleNamespace(
            model_name="ltx2_19B",
            prompt_for_wgp="gentle camera drift",
            generation_policy=object(),
            segment_processing_dir=tmp_path,
        ),
    )
    monkeypatch.setattr(
        task_registry,
        "_resolve_image_references",
        lambda *_args: SimpleNamespace(
            active_svi_continuation=False,
            prefix_video_for_source=None,
        ),
    )
    monkeypatch.setattr(task_registry, "_process_structure_guidance", lambda *_args: object())
    monkeypatch.setattr(
        task_registry,
        "_build_generation_params",
        lambda *_args: {
            "model_name": "ltx2_distilled_19B",
            "prompt": "gentle camera drift",
            "video_source": str(tmp_path / "prefix.mp4"),
        },
    )
    monkeypatch.setattr(task_registry, "_apply_video_source_continuation", _noop)
    monkeypatch.setattr(task_registry, "_apply_uni3c_config", _noop)
    monkeypatch.setattr(
        task_registry,
        "execute_resolved_direct_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("travel child should not enter direct executor")
        ),
    )
    queue = _Queue(fail_on_submit=True)

    ok, message = task_registry._handle_travel_segment_via_queue_impl(
        task_params_dict={"prompt": "travel"},
        main_output_dir_base=tmp_path,
        task_id="travel-child-1",
        colour_match_videos=False,
        mask_active_frames=False,
        task_queue=queue,
        is_standalone=is_standalone,
    )

    assert ok is False
    assert message
    assert "fail-closed" in message
    assert expected_route in message
    assert "vibecomfy_unsupported" in message
    assert "will not fall back to WGP" in message
    assert queue.submitted == []
    route_bits = message.lower()
    assert "vace" not in route_bits
    assert "cocktail" not in route_bits
    assert "wan_2_2" not in route_bits


def test_travel_fail_closed_emits_routing_card(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    logger = _LogCapture()

    monkeypatch.setattr(task_registry, "headless_logger", logger)
    monkeypatch.setattr(task_registry, "log_ram_usage", _noop)
    monkeypatch.setattr(task_registry, "_resolve_segment_context", lambda *_args: _travel_ctx(tmp_path))
    monkeypatch.setattr(
        task_registry,
        "_resolve_generation_inputs",
        lambda *_args: SimpleNamespace(
            model_name="ltx2_19B",
            prompt_for_wgp="gentle camera drift",
            generation_policy=object(),
            segment_processing_dir=tmp_path,
        ),
    )
    monkeypatch.setattr(
        task_registry,
        "_resolve_image_references",
        lambda *_args: SimpleNamespace(
            active_svi_continuation=False,
            prefix_video_for_source=None,
        ),
    )
    monkeypatch.setattr(task_registry, "_process_structure_guidance", lambda *_args: object())
    monkeypatch.setattr(
        task_registry,
        "_build_generation_params",
        lambda *_args: {
            "model_name": "ltx2_distilled_19B",
            "prompt": "gentle camera drift",
            "video_source": str(tmp_path / "prefix.mp4"),
        },
    )
    monkeypatch.setattr(task_registry, "_apply_video_source_continuation", _noop)
    monkeypatch.setattr(task_registry, "_apply_uni3c_config", _noop)

    ok, message = task_registry._handle_travel_segment_via_queue_impl(
        task_params_dict={"prompt": "travel"},
        main_output_dir_base=tmp_path,
        task_id="travel-child-telemetry",
        colour_match_videos=False,
        mask_active_frames=False,
        task_queue=_Queue(fail_on_submit=True),
        is_standalone=True,
    )

    assert ok is False
    assert message
    routing_cards = [
        payload for title, payload, _kwargs in logger.debug_blocks
        if title == "VIBECOMFY_ROUTING"
    ]
    assert routing_cards
    card = routing_cards[0]
    assert card["task_id"] == "travel-child-telemetry"
    assert card["task_type"] == "individual_travel_segment"
    assert (
        card["route_key"]
        == "individual_travel_segment__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-3"
    )
    assert card["backend"] == "vibecomfy"
    assert card["template_id"] is None
    assert card["support_state"] == "vibecomfy_unsupported"
    assert card["memory_profile"] == "3"
    assert card["decision"] == "fail_closed"
    assert "will not fall back to WGP" in card["fail_closed_reason"]


def test_wgp_travel_child_still_submits_and_waits(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.delenv("REIGH_BACKEND", raising=False)

    monkeypatch.setattr(task_registry, "log_ram_usage", _noop)
    monkeypatch.setattr(task_registry, "_resolve_segment_context", lambda *_args: _travel_ctx(tmp_path))
    monkeypatch.setattr(
        task_registry,
        "_resolve_generation_inputs",
        lambda *_args: SimpleNamespace(
            model_name="ltx2_19B",
            prompt_for_wgp="gentle camera drift",
            generation_policy=object(),
            segment_processing_dir=tmp_path,
        ),
    )
    monkeypatch.setattr(
        task_registry,
        "_resolve_image_references",
        lambda *_args: SimpleNamespace(
            active_svi_continuation=False,
            prefix_video_for_source=None,
        ),
    )
    monkeypatch.setattr(task_registry, "_process_structure_guidance", lambda *_args: object())
    monkeypatch.setattr(
        task_registry,
        "_build_generation_params",
        lambda *_args: {"model": "ltx2_19B", "prompt": "gentle camera drift"},
    )
    monkeypatch.setattr(task_registry, "_apply_video_source_continuation", _noop)
    monkeypatch.setattr(task_registry, "_apply_uni3c_config", _noop)
    queue = _Queue()

    ok, result = task_registry._handle_travel_segment_via_queue_impl(
        task_params_dict={"prompt": "travel"},
        main_output_dir_base=tmp_path,
        task_id="travel-child-wgp",
        colour_match_videos=False,
        mask_active_frames=False,
        task_queue=queue,
        is_standalone=True,
    )

    assert ok is True
    assert result == "/tmp/travel-child-wgp.png"
    assert len(queue.submitted) == 1
    assert queue.submitted[0].parameters["_source_task_type"] == "travel_segment"


def test_travel_child_queue_payload_prefers_travel_guidance_over_structure_fields(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.delenv("REIGH_BACKEND", raising=False)
    ctx = SimpleNamespace(
        individual_params={
            "travel_guidance": {"kind": "vace", "mode": "flow"},
            "structure_guidance": {"type": "canny"},
            "structure_videos": [{"url": "https://example.test/guide.mp4"}],
            "chain_segments": True,
            "continuation_config": {"type": "video_source"},
            "continuity_case": "video_source",
        },
        segment_params={},
        orchestrator_details={},
        orchestrator_task_id_ref="orchestrator-1",
        orchestrator_run_id="run-1",
        segment_idx=0,
    )

    monkeypatch.setattr(task_registry, "log_ram_usage", _noop)
    monkeypatch.setattr(task_registry, "_resolve_segment_context", lambda *_args: ctx)
    monkeypatch.setattr(
        task_registry,
        "_resolve_generation_inputs",
        lambda *_args: SimpleNamespace(
            model_name="wan_2_2_vace_lightning_baseline_2_2_2",
            prompt_for_wgp="guided bridge",
            generation_policy=object(),
            segment_processing_dir=tmp_path,
        ),
    )
    monkeypatch.setattr(
        task_registry,
        "_resolve_image_references",
        lambda *_args: SimpleNamespace(active_svi_continuation=False, prefix_video_for_source=None),
    )
    monkeypatch.setattr(task_registry, "_process_structure_guidance", lambda *_args: object())
    monkeypatch.setattr(
        task_registry,
        "_build_generation_params",
        lambda *_args: {"model_name": "wan_2_2_vace_lightning_baseline_2_2_2", "prompt": "guided bridge"},
    )
    monkeypatch.setattr(task_registry, "_apply_video_source_continuation", _noop)
    monkeypatch.setattr(task_registry, "_apply_uni3c_config", _noop)
    queue = _Queue()

    ok, result = task_registry._handle_travel_segment_via_queue_impl(
        task_params_dict={"prompt": "travel"},
        main_output_dir_base=tmp_path,
        task_id="travel-child-guidance",
        colour_match_videos=False,
        mask_active_frames=False,
        task_queue=queue,
        is_standalone=True,
    )

    assert ok is True
    assert result == "/tmp/travel-child-guidance.png"
    parameters = queue.submitted[0].parameters
    assert parameters["travel_guidance"] == {"kind": "vace", "mode": "flow"}
    assert "structure_guidance" not in parameters
    assert "structure_videos" not in parameters
    assert parameters["chain_segments"] is True
    assert parameters["continuation_config"] == {"type": "video_source"}
    assert parameters["continuity_case"] == "video_source"


def test_travel_child_queue_payload_preserves_legacy_structure_contract_without_travel_guidance(monkeypatch, tmp_path):
    task_registry = _import_task_registry(monkeypatch)
    monkeypatch.delenv("REIGH_BACKEND", raising=False)
    ctx = SimpleNamespace(
        individual_params={},
        segment_params={
            "structure_guidance": {"type": "depth", "strength": 0.8},
            "structure_videos": [{"url": "https://example.test/depth.mp4"}],
            "chain_segments": False,
            "independent_segments": True,
        },
        orchestrator_details={},
        orchestrator_task_id_ref="orchestrator-1",
        orchestrator_run_id="run-1",
        segment_idx=0,
    )

    monkeypatch.setattr(task_registry, "log_ram_usage", _noop)
    monkeypatch.setattr(task_registry, "_resolve_segment_context", lambda *_args: ctx)
    monkeypatch.setattr(
        task_registry,
        "_resolve_generation_inputs",
        lambda *_args: SimpleNamespace(
            model_name="wan_2_2_vace_lightning_baseline_2_2_2",
            prompt_for_wgp="legacy guided bridge",
            generation_policy=object(),
            segment_processing_dir=tmp_path,
        ),
    )
    monkeypatch.setattr(
        task_registry,
        "_resolve_image_references",
        lambda *_args: SimpleNamespace(active_svi_continuation=False, prefix_video_for_source=None),
    )
    monkeypatch.setattr(task_registry, "_process_structure_guidance", lambda *_args: object())
    monkeypatch.setattr(
        task_registry,
        "_build_generation_params",
        lambda *_args: {"model_name": "wan_2_2_vace_lightning_baseline_2_2_2", "prompt": "legacy guided bridge"},
    )
    monkeypatch.setattr(task_registry, "_apply_video_source_continuation", _noop)
    monkeypatch.setattr(task_registry, "_apply_uni3c_config", _noop)
    queue = _Queue()

    ok, _result = task_registry._handle_travel_segment_via_queue_impl(
        task_params_dict={"prompt": "travel"},
        main_output_dir_base=tmp_path,
        task_id="travel-child-legacy-structure",
        colour_match_videos=False,
        mask_active_frames=False,
        task_queue=queue,
        is_standalone=True,
    )

    assert ok is True
    parameters = queue.submitted[0].parameters
    assert "travel_guidance" not in parameters
    assert parameters["structure_guidance"] == {"type": "depth", "strength": 0.8}
    assert parameters["structure_videos"] == [{"url": "https://example.test/depth.mp4"}]
    assert parameters["chain_segments"] is False
    assert parameters["independent_segments"] is True
