"""Targeted coverage tests for remaining high-impact open modules."""

from __future__ import annotations

import sys
import types
import uuid

import numpy as np
from PIL import Image

import source.core.db.lifecycle.task_completion as task_completion
import source.core.db.dependencies.task_dependencies_children as deps_children
import source.core.db.dependencies.task_dependencies_queries as deps_queries
import source.core.db.lifecycle.task_status as task_status
import source.media.vlm.image_prep as image_prep
import source.media.visualization.timeline as timeline
import source.models.comfy.comfy_handler as comfy_handler
import source.models.wgp.model_ops as model_ops
import source.task_handlers.create_visualization as create_visualization


class _Result:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _EdgeResp:
    def __init__(self, status_code: int, text: str = "", payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
    def json(self):
        return self._payload


def test_add_task_to_db_queues_and_verifies(monkeypatch):
    class _VerifyClient:
        def table(self, *_args, **_kwargs):
            return self
        def select(self, *_args, **_kwargs):
            return self
        def eq(self, *_args, **_kwargs):
            return self
        def single(self):
            return self
        def execute(self):
            return _Result(
                data={
                    "status": task_completion.STATUS_QUEUED,
                    "created_at": "2026-02-26T00:00:00Z",
                    "project_id": "proj-1",
                    "task_type": "travel_segment",
                }
            )

    monkeypatch.setattr(task_completion._cfg, "SUPABASE_EDGE_CREATE_TASK_URL", "https://edge.example/create-task", raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_ACCESS_TOKEN", "token-abc", raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_CLIENT", _VerifyClient(), raising=False)
    monkeypatch.setattr(task_completion._cfg, "PG_TABLE_NAME", "tasks", raising=False)
    monkeypatch.setattr(uuid, "uuid4", lambda: "task-fixed-id")
    called = {}
    def _fake_call(**kwargs):
        called.update(kwargs)
        return _EdgeResp(200, payload={"ok": True}), None
    monkeypatch.setattr(task_completion, "_call_edge_function_with_retry", _fake_call)
    task_id = task_completion.add_task_to_db(
        task_payload={"prompt": "hello", "project_id": "proj-1"},
        task_type_str="travel_segment",
    )
    assert task_id == "task-fixed-id"
    assert called["function_name"] == "create-task"
    assert called["payload"]["family"] == "travel_segment"
    assert called["payload"]["input"]["task_id"] == "task-fixed-id"


def test_add_task_to_db_raises_when_edge_url_missing(monkeypatch):
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_EDGE_CREATE_TASK_URL", None, raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.delenv("SUPABASE_EDGE_CREATE_TASK_URL", raising=False)
    try:
        task_completion.add_task_to_db({"prompt": "x"}, "travel_segment")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "create-task" in str(exc)


def test_task_dependencies_get_task_dependency_returns_none_without_edge(monkeypatch):
    """get_task_dependency returns None when no edge function URL is available (no direct DB fallback)."""
    monkeypatch.setattr(deps_queries._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(deps_queries._cfg, "SUPABASE_ACCESS_TOKEN", None, raising=False)
    monkeypatch.delenv("SUPABASE_EDGE_GET_TASK_OUTPUT_URL", raising=False)
    assert deps_queries.get_task_dependency("task-1") is None


def test_cancel_orchestrator_children_cancels_only_non_terminal():
    child_map = {
        "segments": [
            {"id": "c1", "status": "Queued"},
            {"id": "c2", "status": "Complete"},
        ],
        "stitch": [{"id": "c3", "status": "In Progress"}],
        "join_clips_segment": [],
        "join_clips_orchestrator": [],
        "join_final_stitch": [],
    }
    cancelled = []
    count = deps_children.cancel_orchestrator_children(
        "orch-1",
        reason="cancelled",
        child_fetcher=lambda _task_id: child_map,
        status_updater=lambda task_id, status, output_location=None: cancelled.append(
            (task_id, status, output_location)
        ),
    )
    assert count == 2
    assert [c[0] for c in cancelled] == ["c1", "c3"]


def test_get_predecessor_output_falls_back_to_direct_calls(monkeypatch):
    monkeypatch.setenv("SUPABASE_ALLOW_DIRECT_QUERY_FALLBACK", "1")
    monkeypatch.setattr(deps_queries._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(deps_queries._cfg, "SUPABASE_ACCESS_TOKEN", None, raising=False)
    predecessor, output = deps_queries.get_predecessor_output_via_edge_function(
        "task-2",
        dependency_lookup=lambda _task_id: "dep-2",
        output_lookup=lambda _task_id: "s3://out.mp4",
    )
    assert predecessor == "dep-2"
    assert output == "s3://out.mp4"


def test_task_status_requeue_returns_false_when_no_edge_url(monkeypatch):
    """Requeue returns False when no edge function URL is configured (no direct DB fallback)."""
    monkeypatch.setattr(task_status._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.delenv("SUPABASE_EDGE_UPDATE_TASK_URL", raising=False)
    ok = task_status.requeue_task_for_retry(
        task_id_str="task-1",
        error_message="temporary failure",
        current_attempts=1,
        error_category="network",
    )
    assert ok is False


def test_create_framed_vlm_image_combines_images_with_borders():
    start = Image.new("RGB", (20, 10), (10, 20, 30))
    end = Image.new("RGB", (30, 10), (40, 50, 60))
    combined = image_prep.create_framed_vlm_image(start, end, border_width=4)
    assert combined.height == 18
    assert combined.width == (20 + 8) + (30 + 8) + 4


def test_create_labeled_debug_image_returns_canvas_larger_than_inputs():
    start = Image.new("RGB", (32, 24), (0, 0, 0))
    end = Image.new("RGB", (32, 24), (255, 255, 255))
    labeled = image_prep.create_labeled_debug_image(start, end, pair_index=7)
    assert labeled.width > 64
    assert labeled.height > 48


def test_create_visualization_returns_error_on_missing_required_param(tmp_path):
    ok, message = create_visualization.handle_create_visualization_task(
        task_params_from_db={"params": {}},
        main_output_dir_base=tmp_path,
        viz_task_id_str="viz-1",
    )
    assert ok is False
    assert "Missing required parameter" in message


def test_create_visualization_success_flow_with_mocked_io(monkeypatch, tmp_path):
    source_viz = tmp_path / "source_viz.mp4"
    source_viz.write_bytes(b"video-data")
    def _fake_create_travel_visualization(**_kwargs):
        return str(source_viz)
    def _fake_prepare_output_path_with_upload(filename, task_id, main_output_dir_base, task_type):
        final_path = main_output_dir_base / f"{filename}"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return final_path, f"db://{task_id}/{task_type}"
    monkeypatch.setattr(create_visualization, "create_travel_visualization", _fake_create_travel_visualization)
    monkeypatch.setattr(create_visualization, "prepare_output_path_with_upload", _fake_prepare_output_path_with_upload)
    monkeypatch.setattr(
        create_visualization,
        "resolve_final_output_location",
        lambda local_file_path, initial_db_location: f"uploaded://{local_file_path.name}",
    )
    ok, output = create_visualization.handle_create_visualization_task(
        task_params_from_db={
            "params": {
                "output_video_path": "/mock/out.mp4",
                "structure_video_path": "/mock/structure.mp4",
                "input_image_paths": ["/mock/a.jpg", "/mock/b.jpg"],
                "segment_frames": [10, 10],
            }
        },
        main_output_dir_base=tmp_path,
        viz_task_id_str="viz-2",
    )
    assert ok is True
    assert output.startswith("uploaded://")


def test_comfy_handler_requires_workflow(tmp_path):
    ok, message = comfy_handler.handle_comfy_task({}, tmp_path, "task-xyz")
    assert ok is False
    assert "workflow" in message


def test_comfy_shutdown_stops_manager(monkeypatch):
    stopped = {"value": False}
    class _Manager:
        def stop(self):
            stopped["value"] = True
    monkeypatch.setattr(comfy_handler, "_comfy_manager", _Manager(), raising=False)
    comfy_handler.shutdown_comfy()
    assert stopped["value"] is True
    assert comfy_handler._comfy_manager is None


def test_model_ops_load_model_impl_emits_switch_debug_card(monkeypatch):
    logs = []
    rendered_cards = []
    released = []
    notified = []
    perf_ticks = iter([10.0, 12.5])

    class _Logger:
        def essential(self, message, *args, **kwargs):
            logs.append(("essential", message))
        def debug(self, message, *args, **kwargs):
            logs.append(("debug", message))
        def success(self, message, *args, **kwargs):
            logs.append(("success", message))
        def warning(self, message, *args, **kwargs):
            logs.append(("warning", message))
        def error(self, message, *args, **kwargs):
            logs.append(("error", message))
        def debug_anomaly(self, *args, **kwargs):
            logs.append(("debug_anomaly", args[0] if args else ""))

    fake_wgp = types.SimpleNamespace(
        transformer_type="old_model",
        reload_needed=False,
        offloadobj=types.SimpleNamespace(release=lambda: released.append(True)),
        wan_model=object(),
        get_model_def=lambda _key: {"architecture": "wan"},
    )

    def _load_models(model_key):
        fake_wgp.transformer_type = model_key
        return object(), "new-offload"

    fake_wgp.load_models = _load_models

    monkeypatch.setattr(model_ops, "model_logger", _Logger())
    monkeypatch.setattr(model_ops, "is_debug_enabled", lambda: True)
    monkeypatch.setattr(model_ops, "get_wgp_runtime_module_mutable", lambda: fake_wgp)
    monkeypatch.setattr(model_ops, "get_wgp_runtime_module", lambda: types.SimpleNamespace(offloadobj="runtime-offload"))
    monkeypatch.setattr(model_ops, "get_model_recursive_prop", lambda *_args, **_kwargs: ["transformer"])
    monkeypatch.setattr(model_ops, "clear_wgp_loaded_model_state", lambda: None)
    monkeypatch.setattr(model_ops, "set_wgp_loaded_model_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_ops, "set_wgp_reload_needed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_ops, "render_card", lambda _logger, card, task_id=None: rendered_cards.append((card.render(), task_id)))
    monkeypatch.setattr(model_ops.time, "perf_counter", lambda: next(perf_ticks))
    monkeypatch.setitem(
        sys.modules,
        "source.models.wgp.generation_helpers",
        types.SimpleNamespace(notify_worker_model_switch=lambda **kwargs: notified.append(kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )

    orchestrator = types.SimpleNamespace(
        smoke_mode=False,
        current_model="old_model",
        state={},
        offloadobj=None,
        wan_root="/tmp",
        _cached_uni3c_controlnet=None,
        _get_base_model_type=lambda _model_key: "wan",
    )

    switched = model_ops.load_model_impl(orchestrator, "new_model")

    assert switched is True
    assert released == [True]
    assert notified == [{"old_model": "old_model", "new_model": "new_model"}]
    assert ("essential", f"Switching model: {model_ops.model_label('old_model')} → {model_ops.model_label('new_model')}") in logs
    assert ("essential", f"Model loaded: {model_ops.model_label('new_model')}") in logs
    assert rendered_cards and rendered_cards[0][1] is None
    assert "Model Switch" in rendered_cards[0][0]
    assert "offload_reuse" in rendered_cards[0][0]
    assert "duration_s" in rendered_cards[0][0]


def test_timeline_apply_video_treatment_adjust_mode(monkeypatch):
    class _ImageSequenceClip:
        def __init__(self, frames, fps):
            self.frames = frames
            self.fps = fps
            self.duration = len(frames) / fps if fps else 0
    moviepy_mod = types.ModuleType("moviepy")
    moviepy_editor = types.ModuleType("moviepy.editor")
    moviepy_editor.ImageSequenceClip = _ImageSequenceClip
    monkeypatch.setitem(sys.modules, "moviepy", moviepy_mod)
    monkeypatch.setitem(sys.modules, "moviepy.editor", moviepy_editor)
    class _Clip:
        fps = 10
        duration = 1.0
        @staticmethod
        def get_frame(_time):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    out_clip = timeline._apply_video_treatment(
        clip=_Clip(),
        target_duration=1.0,
        target_fps=5,
        treatment="adjust",
        video_name="demo",
    )
    assert len(out_clip.frames) == 5
    assert out_clip.fps == 5
