import pytest

from source.core.params import LoRAConfig
from source.core.params.phase_config_parser import parse_phase_config
from source.models.lora.module_manifest import LoRAModuleManifestError
from source.models.lora.sanitizer import (
    normalize_lora_reference,
    sanitize_lora_payload,
    sanitize_lora_values,
)
from source.task_handlers.travel.svi_config import merge_svi_into_generation_params


QWEN_ARCH = "qwen_image_20B"
WAN_ARCH = "wan_2_2_i2v_lightning_baseline_2_2_2"
LTX_ARCH = "ltx2_19B"


def test_url_cleanup_and_duplicate_removal():
    first = "https://huggingface.co/acme/qwen-lora/blob/main/qwen_style.safetensors?download=1"
    duplicate = "https://huggingface.co/acme/qwen-lora/resolve/main/qwen_style.safetensors"

    result = sanitize_lora_values(
        [first, duplicate],
        ["0.8", "1.2"],
        architecture=QWEN_ARCH,
    )

    assert result.manifest is not None
    assert result.loras == [
        "https://huggingface.co/acme/qwen-lora/resolve/main/qwen_style.safetensors"
    ]
    assert result.multipliers == ["0.8"]
    assert [decision.reason for decision in result.decisions] == [None, "duplicate"]


def test_malformed_values_are_quarantined():
    result = sanitize_lora_values(
        ["", 12, "https://huggingface.co/acme/not-a-lora.txt"],
        [1, 2, 3],
        architecture=QWEN_ARCH,
    )

    assert result.loras == []
    assert [decision.reason for decision in result.decisions] == [
        "malformed_empty",
        "malformed_non_string",
        "malformed_extension",
    ]


def test_architecture_mismatch_is_rejected():
    qwen_with_wan = sanitize_lora_values(
        ["https://huggingface.co/acme/wan2.2_i2v_lightx2v_4steps_lora.safetensors"],
        [1.0],
        architecture=QWEN_ARCH,
    )
    wan_with_ltx = sanitize_lora_values(
        ["https://huggingface.co/acme/ltx_ic-lora_style.safetensors"],
        [1.0],
        architecture=WAN_ARCH,
    )
    ltx_with_qwen = sanitize_lora_values(
        ["https://huggingface.co/acme/qwen_image_lightning_lora.safetensors"],
        [1.0],
        architecture=LTX_ARCH,
    )

    assert qwen_with_wan.loras == []
    assert wan_with_ltx.loras == []
    assert ltx_with_qwen.loras == []
    assert qwen_with_wan.decisions[0].reason == "architecture_mismatch:wan_lora_for_qwen_model"
    assert wan_with_ltx.decisions[0].reason == "architecture_mismatch:ltx_lora_for_wan_model"
    assert ltx_with_qwen.decisions[0].reason == "architecture_mismatch:qwen_lora_for_ltx_model"


def test_missing_manifest_fails_for_architecture_scoped_loras():
    with pytest.raises(LoRAModuleManifestError, match="missing LoRA module manifest"):
        sanitize_lora_values(
            ["https://huggingface.co/acme/qwen_style.safetensors"],
            [1.0],
            architecture="qwen_missing_20B",
        )


def test_payload_sanitizer_preserves_phase_multipliers():
    result = sanitize_lora_payload(
        {
            "activated_loras": [
                "https://huggingface.co/acme/qwen_high.safetensors",
                "https://huggingface.co/acme/qwen_low.safetensors",
            ],
            "loras_multipliers": "1;0 0;1",
        },
        architecture=QWEN_ARCH,
    )

    assert result.loras == [
        "https://huggingface.co/acme/qwen_high.safetensors",
        "https://huggingface.co/acme/qwen_low.safetensors",
    ]
    assert result.multipliers == ["1;0", "0;1"]


def test_lora_config_parsing_uses_sanitizer_before_payload_construction():
    params = {
        "activated_loras": [
            "https://huggingface.co/acme/qwen-style/blob/main/qwen_style.safetensors?download=1",
            "https://huggingface.co/acme/qwen-style/resolve/main/qwen_style.safetensors",
            "https://huggingface.co/acme/wan2.2_i2v_lightx2v_4steps_lora.safetensors",
        ],
        "loras_multipliers": "0.7 0.9 1.0",
    }

    config = LoRAConfig.from_params(params, model=QWEN_ARCH, task_id="sanitize")

    assert len(config.entries) == 1
    assert config.entries[0].url == (
        "https://huggingface.co/acme/qwen-style/resolve/main/qwen_style.safetensors"
    )
    assert config.entries[0].multiplier == 0.7


def test_phase_config_parser_uses_manifest_backed_sanitizer():
    phase_config = {
        "num_phases": 2,
        "steps_per_phase": [2, 2],
        "phases": [
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://huggingface.co/acme/wan_style.safetensors", "multiplier": 1.0},
                    {"url": "https://huggingface.co/acme/ltx_ic-lora.safetensors", "multiplier": 0.5},
                ],
            },
            {
                "guidance_scale": 1.0,
                "loras": [
                    {"url": "https://huggingface.co/acme/wan_style.safetensors", "multiplier": 0.0},
                    {"url": "https://huggingface.co/acme/ltx_ic-lora.safetensors", "multiplier": 0.5},
                ],
            },
        ],
    }

    parsed = parse_phase_config(
        phase_config,
        num_inference_steps=4,
        task_id="phase-sanitize",
        model_name=WAN_ARCH,
    )

    assert parsed["lora_names"] == ["https://huggingface.co/acme/wan_style.safetensors"]
    assert parsed["lora_multipliers"] == ["1.0;0.0"]
    assert parsed["additional_loras"] == {"https://huggingface.co/acme/wan_style.safetensors": 1.0}


def test_svi_injection_sanitizes_against_generation_architecture():
    wan_params = {"model_name": WAN_ARCH}
    merge_svi_into_generation_params(wan_params)
    assert len(wan_params["activated_loras"]) == 4
    assert "SVI_Wan2.2" in wan_params["activated_loras"][0]

    qwen_params = {"model_name": QWEN_ARCH}
    merge_svi_into_generation_params(qwen_params)
    assert qwen_params["activated_loras"] == []
    assert qwen_params["loras_multipliers"] == ""


def test_normalize_lora_reference_rejects_non_lora_extensions():
    normalized, filename, reason = normalize_lora_reference("https://huggingface.co/acme/model.ckpt")

    assert normalized is None
    assert filename is None
    assert reason == "malformed_extension"

