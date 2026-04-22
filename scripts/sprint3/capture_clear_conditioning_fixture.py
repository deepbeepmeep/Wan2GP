from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WAN_ROOT = ROOT / "Wan2GP"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "clear_conditioning_oracle.pt"

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


def main() -> None:
    torch.manual_seed(20260422)

    token_dim = 7
    num_tokens = 5
    latent_state = LatentState(
        latent=torch.randn(2, token_dim, 4, dtype=torch.float32, device="cpu"),
        clean_latent=torch.randn(2, token_dim, 4, dtype=torch.float32, device="cpu"),
        denoise_mask=torch.randn(2, token_dim, dtype=torch.float32, device="cpu"),
        positions=torch.randint(0, 100, (2, 3, token_dim), dtype=torch.int64, device="cpu"),
    )

    outputs = _ToolsStub(num_tokens).clear_conditioning(latent_state)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inputs": {
                "latent": latent_state.latent.clone(),
                "clean_latent": latent_state.clean_latent.clone(),
                "denoise_mask": latent_state.denoise_mask.clone(),
                "positions": latent_state.positions.clone(),
                "num_tokens": num_tokens,
            },
            "outputs": {
                "latent": outputs.latent.clone(),
                "clean_latent": outputs.clean_latent.clone(),
                "denoise_mask": outputs.denoise_mask.clone(),
                "positions": outputs.positions.clone(),
            },
        },
        FIXTURE_PATH,
    )


if __name__ == "__main__":
    main()
