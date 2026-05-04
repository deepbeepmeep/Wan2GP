# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Sam3MultiplexVideoPredictor — user-facing entry point for SAM 3.1 multiplex.

Ported from onevision Sam3Model (webdemo/ta/models/sam3_model.py).
Handles warm-up compilation, bf16 autocast, and session management
via the shared Sam3BasePredictor handle_request/handle_stream_request API.
"""

from contextlib import nullcontext
from typing import Dict, Optional

import torch
from ..logger import get_logger
from ..model.sam3_base_predictor import Sam3BasePredictor

logger = get_logger(__name__)


class Sam3MultiplexVideoPredictor(Sam3BasePredictor):
    """
    User-facing predictor for SAM 3.1 multiplex video tracking.

    Wraps Sam3MultiplexTrackingWithInteractivity with:
    - bf16 autocast
    - Warm-up compilation (when compile=True)
    - Session expiration management
    - handle_request / handle_stream_request dispatch API (from Sam3BasePredictor)
    """

    def __init__(
        self,
        model,
        session_expiration_sec=1200,
        default_output_prob_thresh=0.5,
        async_loading_frames=True,
        warm_up=False,
        manual_model_loading=False,
    ):
        super().__init__()
        self.model = model
        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames
        self.manual_model_loading = manual_model_loading

        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # use bfloat16 inference for Flash Attention kernel
        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        self.bf16_context.__enter__()

        if warm_up:
            self._ensure_model_on_cuda()
            self.model._warm_up_complete = False
            self.model.warm_up_compilation()
            self.model._warm_up_complete = True

    def _ensure_model_on_cuda(self):
        if not torch.cuda.is_available() or self.model is None:
            return
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:
            return
        if first_parameter.device.type != "cuda":
            self.model.to(device=torch.device("cuda"), dtype=torch.bfloat16)

    def load_model_to_gpu(self):
        self._ensure_model_on_cuda()

    def unload_model_from_gpu(self):
        if not torch.cuda.is_available() or self.model is None:
            return
        self._clear_cuda_runtime_caches(self.model)
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def add_prompt(self, *args, **kwargs):
        if not self.manual_model_loading:
            self._ensure_model_on_cuda()
        return super().add_prompt(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        if not self.manual_model_loading:
            self._ensure_model_on_cuda()
        yield from super().propagate_in_video(*args, **kwargs)

    def _extend_expiration_time(self, session):
        """Update last-use time and store session expiration timeout."""
        super()._extend_expiration_time(session)
        if self.session_expiration_sec:
            session["expiration_sec"] = self.session_expiration_sec
