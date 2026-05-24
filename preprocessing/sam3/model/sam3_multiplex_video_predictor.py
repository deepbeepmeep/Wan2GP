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
from ..model.device_utils import accelerator_autocast, empty_accelerator_cache, get_accelerator_device, is_accelerator_device
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
        # use bfloat16 inference on the active accelerator
        self.bf16_context = accelerator_autocast() if is_accelerator_device(get_accelerator_device()) else nullcontext()
        self.bf16_context.__enter__()

        if warm_up:
            self._ensure_model_on_device()
            self.model._warm_up_complete = False
            self.model.warm_up_compilation()
            self.model._warm_up_complete = True

    def _ensure_model_on_device(self):
        device = get_accelerator_device()
        if device.type == "cpu" or self.model is None:
            return
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:
            return
        if first_parameter.device != device:
            self.model.to(device=device, dtype=torch.bfloat16)

    def _ensure_model_on_cuda(self):
        self._ensure_model_on_device()

    def load_model_to_gpu(self):
        self._ensure_model_on_device()

    def unload_model_from_gpu(self):
        if self.model is None:
            return
        self._clear_cuda_runtime_caches(self.model)
        self.model.to("cpu")
        empty_accelerator_cache()

    def add_prompt(self, *args, **kwargs):
        if not self.manual_model_loading:
            self._ensure_model_on_device()
        return super().add_prompt(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        if not self.manual_model_loading:
            self._ensure_model_on_device()
        yield from super().propagate_in_video(*args, **kwargs)

    def _extend_expiration_time(self, session):
        """Update last-use time and store session expiration timeout."""
        super()._extend_expiration_time(session)
        if self.session_expiration_sec:
            session["expiration_sec"] = self.session_expiration_sec
