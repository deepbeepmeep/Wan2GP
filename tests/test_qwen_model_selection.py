"""Regression tests for distinct Qwen image model selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import source.task_handlers.tasks.task_conversion as task_conversion_module
from source.media.prompt_expansion import PromptExpansionMetadata, PromptExpansionResult
from source.models.model_handlers.qwen_handler import QwenHandler
import source.models.model_handlers.qwen_handler as qwen_handler_module
from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
from source.task_handlers.tasks.task_types import TASK_TYPE_TO_MODEL, get_default_model


def _patch_qwen_lora_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_download(self: QwenHandler, _repo_id: str, filename: str, hf_subfolder: str | None = None):
        del hf_subfolder
        target = self.qwen_lora_dir / filename
        target.write_text("fake", encoding="utf-8")
        return target

    monkeypatch.setattr(QwenHandler, "_download_lora_if_missing", _fake_download)


def _patch_qwen_image_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), (24, 64, 96)).save(image_path)

    def _fake_download_image_if_url(*_args, **_kwargs):
        return image_path

    def _fake_composite(self: QwenHandler, _image_url: str, _mask_url: str, output_dir: Path):
        del self
        output_dir.mkdir(parents=True, exist_ok=True)
        composite_path = output_dir / "composite.png"
        Image.new("RGB", (32, 32), (96, 64, 24)).save(composite_path)
        return str(composite_path)

    monkeypatch.setattr(qwen_handler_module, "download_image_if_url", _fake_download_image_if_url)
    monkeypatch.setattr(QwenHandler, "create_qwen_masked_composite", _fake_composite)


def test_qwen_text_to_image_catalog_models_are_distinct() -> None:
    assert TASK_TYPE_TO_MODEL["qwen_image"] == "qwen_image_20B"
    assert TASK_TYPE_TO_MODEL["qwen_image_2512"] == "qwen_image_2512_20B"
    assert get_default_model("qwen_image") != get_default_model("qwen_image_2512")


def test_qwen_image_conversion_uses_non_edit_base_model_and_v1_lora(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_qwen_lora_downloads(monkeypatch)

    task = db_task_to_generation_task(
        {
            "prompt": "a compact sign reading QWEN",
            "resolution": "2000x1000",
            "num_inference_steps": 4,
        },
        task_id="qwen-base",
        task_type="qwen_image",
        wan2gp_path=str(tmp_path),
    )

    assert task.model == "qwen_image_20B"
    assert task.parameters["resolution"] == "1200x600"
    assert task.parameters["guidance_scale"] == 3.5
    assert task.parameters["system_prompt"].startswith("You are a professional image generator")
    assert "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors" in task.parameters["lora_names"]
    assert not any("2512" in name for name in task.parameters["lora_names"])


def test_qwen_image_2512_conversion_uses_distinct_model_prompt_cap_and_lora(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_qwen_lora_downloads(monkeypatch)

    task = db_task_to_generation_task(
        {
            "prompt": "a crisp poster with legible title text",
            "resolution": "2000x1000",
            "num_inference_steps": 4,
        },
        task_id="qwen-2512",
        task_type="qwen_image_2512",
        wan2gp_path=str(tmp_path),
    )

    assert task.model == "qwen_image_2512_20B"
    assert task.parameters["resolution"] == "2000x1000"
    assert task.parameters["guidance_scale"] == 4.0
    assert "text rendering" in task.parameters["system_prompt"]
    assert task.parameters["lora_names"] == ["Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"]


def test_qwen_image_2512_default_file_is_wgp_loadable_without_aliasing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / "Wan2GP" / "defaults" / "qwen_image_2512_20B.json"
    base_default_path = repo_root / "Wan2GP" / "defaults" / "qwen_image_20B.json"
    handler_path = repo_root / "Wan2GP" / "models" / "qwen" / "qwen_handler.py"

    default_data = json.loads(default_path.read_text(encoding="utf-8"))
    base_default_data = json.loads(base_default_path.read_text(encoding="utf-8"))
    supported_source = handler_path.read_text(encoding="utf-8")

    model_def = default_data["model"]
    base_model_def = base_default_data["model"]

    assert model_def["architecture"] == "qwen_image_20B"
    assert "qwen_image_20B" in supported_source
    assert any("qwen_image_2512_20B" in url for url in model_def["URLs"])
    assert not any("qwen_image_2512_20B" in url for url in base_model_def["URLs"])
    assert model_def["URLs"] != base_model_def["URLs"]


def test_qwen_prompt_expansion_runs_before_qwen_handler_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_qwen_lora_downloads(monkeypatch)
    calls = []

    def _fake_expand(prompt: str, **kwargs):
        calls.append((prompt, kwargs))
        return PromptExpansionResult(
            prompt="expanded prompt with richer detail",
            metadata=PromptExpansionMetadata(
                provider="fake-qwen",
                task_type=kwargs["task_type"],
                requested=True,
                applied=True,
                original_prompt=prompt,
                expanded_prompt="expanded prompt with richer detail",
            ),
        )

    monkeypatch.setattr(task_conversion_module, "expand_qwen_prompt", _fake_expand)

    task = db_task_to_generation_task(
        {
            "prompt": "plain prompt",
            "resolution": "1024x1024",
            "qwen_prompt_expansion": True,
        },
        task_id="qwen-expand",
        task_type="qwen_image",
        wan2gp_path=str(tmp_path),
    )

    assert calls and calls[0][0] == "plain prompt"
    assert calls[0][1]["task_type"] == "qwen_image"
    assert task.prompt == "expanded prompt with richer detail"
    assert task.parameters["prompt"] == "expanded prompt with richer detail"
    assert task.parameters["_original_prompt"] == "plain prompt"
    assert task.parameters["_qwen_prompt_expansion"]["original_prompt"] == "plain prompt"
    assert task.parameters["_qwen_prompt_expansion"]["expanded_prompt"] == "expanded prompt with richer detail"
    assert task.parameters["system_prompt"].startswith("You are a professional image generator")
    assert "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors" in task.parameters["lora_names"]


@pytest.mark.parametrize(
    ("variant", "expected_model", "expected_lora"),
    [
        ("qwen-edit", "qwen_image_edit_20B", "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"),
        (
            "qwen-edit-2509",
            "qwen_image_edit_plus_20B",
            "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
        ),
        (
            "qwen-edit-2511",
            "qwen_image_edit_plus2_20B",
            "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        ),
    ],
)
@pytest.mark.parametrize(
    "task_type",
    ["qwen_image_edit", "qwen_image_style", "image_inpaint", "annotated_image_edit"],
)
def test_qwen_edit_routes_select_matching_base_model_and_lightning_lora(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_type: str,
    variant: str,
    expected_model: str,
    expected_lora: str,
) -> None:
    _patch_qwen_lora_downloads(monkeypatch)
    _patch_qwen_image_inputs(monkeypatch, tmp_path)

    params = {
        "prompt": "make the subject cinematic",
        "qwen_edit_model": variant,
        "image": "https://example.com/input.png",
        "image_url": "https://example.com/input.png",
        "mask_url": "https://example.com/mask.png",
        "resolution": "1800x900",
    }
    if task_type == "qwen_image_style":
        params["style_reference_image"] = "https://example.com/style.png"

    task = db_task_to_generation_task(
        params,
        task_id=f"{task_type}-{variant}",
        task_type=task_type,
        wan2gp_path=str(tmp_path),
    )

    assert task.model == expected_model
    assert expected_lora in task.parameters["lora_names"]
