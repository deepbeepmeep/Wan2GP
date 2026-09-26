import os
from pathlib import Path

import numpy as np
import pytest
import torch
from mmgp import offload

from shared.tinyvae.decoder import REGISTRY, decoder_for, frame_indices, load_decoder
from shared.tinyvae.session import PreviewSession


def test_unavailable_architecture_uses_rgb():
    assert decoder_for("ltx2_22B_msr", {}) is None
    assert decoder_for("qwen_image_21", {}) is None
    assert decoder_for("ltx2_22B", {}) == "taeltx2_3"


def test_scheduler_limits_work_and_honors_abort_and_pass_changes():
    calls = []
    gen = {}
    def decoder(latent, **kwargs):
        calls.append(latent)
        return latent
    events = []
    session = PreviewSession(decoder, lambda *event: events.append(event), gen, False)
    for step in range(40):
        session.capture(step, step, 40, 1)
    assert len(events) <= 7 and calls[-1] == 39
    gen["abort"] = True
    session.capture(99, 39, 40, 1)
    assert calls[-1] == 39
    gen["abort"] = False
    session.capture(100, 0, 4, 2)
    assert calls[-1] == 100


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("TINYVAE_TEST_WEIGHTS"), reason="Requires CUDA and real TinyVAE weights")
def test_temporal_selection_matches_full_decode_and_cancellation_recovers():
    path = Path(os.environ["TINYVAE_TEST_WEIGHTS"]) / REGISTRY["decoders"]["taeltx2_3"]["filename"]
    model = load_decoder("taeltx2_3", str(path))
    manager = offload.profile({"tiny_vae": model}, profile_no=5, quantizeTransformer=False, convertWeightsFloatTo=None, coTenantsMap={"tiny_vae": "*"})
    try:
        latent = torch.randn(128, 3, 4, 4, device="cuda")
        with torch.inference_mode():
            manager.ensure_model_loaded("tiny_vae")
            value = latent.permute(1, 0, 2, 3)[None].to(next(model.parameters()))
            full = model.net.decode_video(value, parallel=False)
            indices = frame_indices(full.shape[1])
            selected = model.net.decode_video(value, parallel=False, output_indices=indices)
            torch.testing.assert_close(selected, full[:, indices], rtol=0, atol=0)
            polls = []
            def abort():
                polls.append(True)
                return len(polls) == 4
            assert model(latent, abort_check=abort) is None
            assert len(polls) == 4
            assert model(latent).size == (800, 200)
    finally:
        manager.release()
