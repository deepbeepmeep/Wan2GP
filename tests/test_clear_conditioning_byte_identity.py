from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = ROOT / "Wan2GP"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "clear_conditioning_oracle.pt"

if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from models.ltx2.ltx_core.tools import LatentTools
from models.ltx2.ltx_core.types import LatentState


class _PatchifierStub:
    def __init__(self, num_tokens: int) -> None:
        self._num_tokens = num_tokens

    def get_token_count(self, _target_shape: object) -> int:
        return self._num_tokens


class _ToolsStub:
    clear_conditioning = LatentTools.clear_conditioning

    def __init__(self, num_tokens: int) -> None:
        self.patchifier = _PatchifierStub(num_tokens)
        self.target_shape = object()


def _assert_tensor_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert actual.detach().cpu().numpy().tobytes() == expected.detach().cpu().numpy().tobytes()


def test_clear_conditioning_matches_saved_fixture_byte_for_byte():
    fixture = torch.load(FIXTURE_PATH, map_location="cpu")
    inputs = fixture["inputs"]
    expected_outputs = fixture["outputs"]

    latent_state = LatentState(
        latent=inputs["latent"].clone(),
        clean_latent=inputs["clean_latent"].clone(),
        denoise_mask=inputs["denoise_mask"].clone(),
        positions=inputs["positions"].clone(),
    )

    actual_outputs = _ToolsStub(int(inputs["num_tokens"])).clear_conditioning(latent_state)

    _assert_tensor_bytes_equal(actual_outputs.latent, expected_outputs["latent"])
    _assert_tensor_bytes_equal(actual_outputs.clean_latent, expected_outputs["clean_latent"])
    _assert_tensor_bytes_equal(actual_outputs.denoise_mask, expected_outputs["denoise_mask"])
    _assert_tensor_bytes_equal(actual_outputs.positions, expected_outputs["positions"])
