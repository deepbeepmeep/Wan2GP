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
