from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source.models.comfy import vibecomfy_adapter
from source.models.comfy.vibecomfy_adapter import handle_vibecomfy_resolved_task
from source.task_handlers.tasks.template_routing import resolve_task_route


def _resolved(params=None):
    return resolve_task_route(
        task_id="adapter-task",
        task_type="z_image_turbo",
        params={"prompt": "adapter prompt", **(params or {})},
        backend="vibecomfy",
    )


def _resolved_route(task_type: str, params=None):
    return resolve_task_route(
        task_id=f"{task_type}-task",
        task_type=task_type,
        params={"prompt": f"{task_type} prompt", **(params or {})},
        backend="vibecomfy",
    )


def _completed(*, returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class _LogCapture:
    def __init__(self):
        self.debug_blocks = []
        self.errors = []

    def debug_block(self, title, payload, **kwargs):
        self.debug_blocks.append((title, payload, kwargs))

    def error(self, message, **kwargs):
        self.errors.append((message, kwargs))


def test_uses_vibecomfy_python_env_when_set(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setenv("VIBECOMFY_PYTHON", "/opt/vibe/bin/python")

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, result = handle_vibecomfy_resolved_task(_resolved(), tmp_path)

    assert ok is True
    assert result and result.endswith("out.png")
    assert commands[0][0] == "/opt/vibe/bin/python"


def test_defaults_to_python311_and_avoids_unsupported_cli_flags(monkeypatch, tmp_path):
    commands = []
    monkeypatch.delenv("VIBECOMFY_PYTHON", raising=False)

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, _result = handle_vibecomfy_resolved_task(
        _resolved({"resolution": "1024x1024", "seed": 7, "steps": 8}),
        tmp_path,
    )

    command = commands[0]
    assert ok is True
    assert command[:4] == ["python3.11", "-m", "vibecomfy.cli", "run"]
    scratchpad = Path(command[4])
    assert scratchpad.name == "z_image_turbo_scratchpad.py"
    assert scratchpad.exists()
    scratchpad_source = scratchpad.read_text(encoding="utf-8")
    assert "resolution(1024, 1024)" in scratchpad_source
    assert 'workflow.set_prompt("adapter prompt")' in scratchpad_source
    assert "workflow.set_seed(7)" in scratchpad_source
    assert "workflow.set_steps(8)" in scratchpad_source
    assert "--ready" not in command
    assert "--runtime" in command
    assert "--prompt" not in command
    assert "--seed" not in command
    assert "--steps" not in command
    assert "--resolution" not in command
    assert "--output-directory" not in command


def test_memory_profile_args_are_included(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setenv("VIBECOMFY_MEMORY_PROFILE", "1")

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, _result = handle_vibecomfy_resolved_task(_resolved({"override_profile": 3}), tmp_path)

    assert ok is True
    assert "--memory-profile" in commands[0]
    assert commands[0][commands[0].index("--memory-profile") + 1] == "3"


def test_subprocess_failure_returns_bounded_telemetry(monkeypatch, tmp_path):
    long_stderr = "x" * 5000
    logger = _LogCapture()

    def _run(command, cwd, env, text, capture_output, check):
        return _completed(returncode=17, stdout="small stdout", stderr=long_stderr)

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    monkeypatch.setattr(vibecomfy_adapter, "headless_logger", logger)

    ok, message = handle_vibecomfy_resolved_task(_resolved({"override_profile": 3}), tmp_path)

    assert ok is False
    assert message
    assert "backend=vibecomfy" in message
    assert "template=image/z_image" in message
    assert "profile=3" in message
    assert "exit_code=17" in message
    assert "small stdout" in message
    assert len(message) < 4600
    failure_cards = [
        payload for title, payload, _kwargs in logger.debug_blocks
        if title == "VIBECOMFY_FAILURE"
    ]
    assert failure_cards
    failure_card = failure_cards[0]
    assert failure_card["backend"] == "vibecomfy"
    assert failure_card["template_id"] == "image/z_image"
    assert failure_card["memory_profile"] == 3
    assert failure_card["exit_code"] == 17
    assert len(failure_card["stderr"]) == 4000


def test_output_discovery_from_parsed_stdout(monkeypatch, tmp_path):
    logger = _LogCapture()

    def _run(command, cwd, env, text, capture_output, check):
        output = Path(cwd) / "nested" / "parsed.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: nested/parsed.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    monkeypatch.setattr(vibecomfy_adapter, "headless_logger", logger)

    ok, result = handle_vibecomfy_resolved_task(_resolved(), tmp_path)

    assert ok is True
    assert result and result.endswith("nested/parsed.png")
    completion_cards = [
        payload for title, payload, _kwargs in logger.debug_blocks
        if title == "VIBECOMFY_COMPLETE"
    ]
    assert completion_cards
    completion_card = completion_cards[0]
    assert completion_card["backend"] == "vibecomfy"
    assert completion_card["template_id"] == "image/z_image"
    assert completion_card["exit_code"] == 0


def test_output_discovery_from_metadata(monkeypatch, tmp_path):
    def _run(command, cwd, env, text, capture_output, check):
        metadata = Path(cwd) / "out" / "runs" / "run-1" / "metadata.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({"outputs": ["images/final.png"]}), encoding="utf-8")
        image = Path(cwd) / "images" / "final.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_text("fake", encoding="utf-8")
        return _completed(stdout=f"metadata: {metadata.relative_to(cwd)}\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, result = handle_vibecomfy_resolved_task(_resolved(), tmp_path)

    assert ok is True
    assert result and result.endswith("images/final.png")


def test_output_discovery_from_isolated_run_directory(monkeypatch, tmp_path):
    def _run(command, cwd, env, text, capture_output, check):
        assert env["VIBECOMFY_WORKER_RUN_DIR"] == str(cwd)
        output = Path(cwd) / "out" / "runs" / "run-1" / "fallback.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="run_id: run-1\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, result = handle_vibecomfy_resolved_task(_resolved(), tmp_path)

    assert ok is True
    assert result and result.endswith("fallback.png")
    assert "vibecomfy_runs/adapter-task" in result


def test_video_output_is_validated_with_ffprobe_contract(monkeypatch, tmp_path):
    commands = []

    def _run(command, *args, **kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return _completed(stdout=json.dumps({
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1280,
                        "height": 720,
                        "nb_frames": "49",
                        "avg_frame_rate": "24/1",
                        "duration": "2.041667",
                    },
                    {"codec_type": "audio", "duration": "2.041667"},
                ],
                "format": {"duration": "2.041667"},
            }))
        output = Path(kwargs["cwd"]) / "out.mp4"
        output.write_bytes(b"fake")
        return _completed(stdout="output: out.mp4\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")

    ok, result = handle_vibecomfy_resolved_task(
        _resolved({
            "num_frames": 49,
            "fps": 24,
            "resolution": "1280x720",
            "require_audio": True,
            "require_thumbnail": True,
            "thumbnail_path": str(thumbnail),
        }),
        tmp_path,
    )

    assert ok is True
    assert result and result.endswith("out.mp4")
    assert any(command[0] == "ffprobe" for command in commands)


def test_video_output_contract_violation_fails_before_completion(monkeypatch, tmp_path):
    def _run(command, *args, **kwargs):
        if command[0] == "ffprobe":
            return _completed(stdout=json.dumps({
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1280,
                        "height": 720,
                        "nb_frames": "48",
                        "avg_frame_rate": "24/1",
                        "duration": "2.0",
                    }
                ],
                "format": {"duration": "2.0"},
            }))
        output = Path(kwargs["cwd"]) / "out.mp4"
        output.write_bytes(b"fake")
        return _completed(stdout="output: out.mp4\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)

    ok, message = handle_vibecomfy_resolved_task(
        _resolved({"num_frames": 49, "fps": 24}),
        tmp_path,
    )

    assert ok is False
    assert message
    assert "media contract violation" in message
    assert "frame count mismatch" in message


def test_non_default_resolution_is_patched_through_scratchpad(monkeypatch, tmp_path):
    commands = []

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    resolved = resolve_task_route(
        task_id="adapter-task",
        task_type="z_image_turbo",
        params={"prompt": "non default", "resolution": "896x496", "seed": 123, "num_inference_steps": 5},
        backend="vibecomfy",
    )

    ok, result = handle_vibecomfy_resolved_task(resolved, tmp_path)

    assert ok is True
    assert result and result.endswith("out.png")
    scratchpad = Path(commands[0][4])
    scratchpad_source = scratchpad.read_text(encoding="utf-8")
    assert "resolution(896, 496)" in scratchpad_source
    assert 'workflow.set_prompt("non default")' in scratchpad_source
    assert "workflow.set_seed(123)" in scratchpad_source
    assert "workflow.set_steps(5)" in scratchpad_source


def test_qwen_2512_uses_ready_template_scratchpad(monkeypatch, tmp_path):
    commands = []

    def _run(command, cwd, env, text, capture_output, check):
        commands.append((command, env))
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    resolved = _resolved_route(
        "qwen_image_2512",
        {"resolution": "1536x864", "seed": 2512, "num_inference_steps": 4},
    )

    ok, result = handle_vibecomfy_resolved_task(resolved, tmp_path)

    assert ok is True
    assert result and result.endswith("out.png")
    command, env = commands[0]
    scratchpad = Path(command[4])
    source = scratchpad.read_text(encoding="utf-8")
    assert "load_workflow_any('image/qwen_image_2512')" in source
    assert "resolution(1536, 864)" in source
    assert 'workflow.set_prompt("qwen_image_2512 prompt")' in source
    assert "workflow.set_seed(2512)" in source
    assert "workflow.nodes['238:224'].inputs['value'] = 4" in source
    comfy_config = json.loads(env["VIBECOMFY_COMFY_CONFIGURATION"])
    assert comfy_config["input_directory"].endswith("qwen_image_2512-task/input")
    assert comfy_config["output_directory"].endswith("qwen_image_2512-task/output")


def test_qwen_edit_materializes_input_image_and_scratchpad(monkeypatch, tmp_path):
    commands = []
    source_image = tmp_path / "source.png"
    source_image.write_text("fake image", encoding="utf-8")

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    resolved = _resolved_route(
        "qwen_image_edit",
        {"image": str(source_image), "seed": 44, "steps": 5},
    )

    ok, result = handle_vibecomfy_resolved_task(resolved, tmp_path)

    assert ok is True
    assert result and result.endswith("out.png")
    run_dir = tmp_path / "vibecomfy_runs" / "qwen_image_edit-task"
    materialized = list((run_dir / "input").glob("qwen_image_edit_qwen_image_edit-task.*"))
    assert materialized
    source = Path(commands[0][4]).read_text(encoding="utf-8")
    assert "load_workflow_any('edit/qwen_image_edit')" in source
    assert "workflow.nodes['78'].inputs['image']" in source
    assert 'workflow.set_prompt("qwen_image_edit prompt")' in source
    assert "workflow.set_seed(44)" in source
    assert "workflow.nodes['102:103'].inputs['value'] = 5" in source


def test_inpaint_route_uses_mask_composite(monkeypatch, tmp_path):
    commands = []

    def _fake_composite(image_url, mask_url, output_dir, task_id):
        path = Path(output_dir) / "composite.jpg"
        path.write_text(f"{image_url}|{mask_url}|{task_id}", encoding="utf-8")
        return str(path)

    def _run(command, cwd, env, text, capture_output, check):
        commands.append(command)
        output = Path(cwd) / "out.png"
        output.write_text("fake", encoding="utf-8")
        return _completed(stdout="output: out.png\n")

    monkeypatch.setattr(vibecomfy_adapter, "create_qwen_masked_composite", _fake_composite)
    monkeypatch.setattr(vibecomfy_adapter.subprocess, "run", _run)
    resolved = _resolved_route(
        "image_inpaint",
        {"image_url": "https://example.invalid/image.png", "mask_url": "https://example.invalid/mask.png"},
    )

    ok, result = handle_vibecomfy_resolved_task(resolved, tmp_path)

    assert ok is True
    assert result and result.endswith("out.png")
    source = Path(commands[0][4]).read_text(encoding="utf-8")
    assert "load_workflow_any('edit/qwen_image_edit')" in source
    assert "image_inpaint_image_inpaint-task.jpg" in source
