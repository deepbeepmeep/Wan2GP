from __future__ import annotations

import os
import uuid
from typing import Any, Callable

from .adapters.ltx2 import decode_ltx2_latent
from .loader import load_decoder, unload_decoders
from .registry import PreviewDecoderSpec
from .scheduler import CaptureScheduler
from .types import PreviewContext, PreviewMedia, PreviewOptions
from .worker import PreviewJob, PreviewWorker


class PreviewCoordinator:
    def __init__(self, model_type: str, architecture: str, spec: PreviewDecoderSpec, options: PreviewOptions, publish: Callable[[PreviewMedia], None], warning: Callable[[str], None] | None = None) -> None:
        self.model_type = model_type
        self.architecture = architecture
        self.spec = spec
        self.options = options
        self.generation_id = uuid.uuid4().hex
        self.context_id = uuid.uuid4().hex
        self.sequence = 0
        self.cancelled = False
        self.disabled = False
        self._decoder = None
        self._decode_device: str | None = None
        self._warning = warning
        self._worker = PreviewWorker(self._publish, self._disable)
        self._scheduler: CaptureScheduler | None = None
        self._publish_callback = publish

    def _disable(self, error: Exception) -> None:
        self.disabled = True
        if self._warning is not None:
            self._warning(f"Tiny VAE preview unavailable; continuing with RGB ({error})")

    def _publish(self, media: PreviewMedia) -> None:
        if (
            not self.cancelled
            and not self.disabled
            and media.generation_id == self.generation_id
            and media.context_id == self.context_id
            and media.sequence == self.sequence
        ):
            self._publish_callback(media)

    def capture(self, latent: Any, *, step: int, total_steps: int, pass_no: int | None = None, window_no: int | None = None, fps: float | None = None, duration_seconds: float | None = None, context_id: str | None = None, force_refresh: bool = False) -> bool:
        if self.cancelled or self.disabled:
            return False
        if context_id and context_id != self.context_id:
            self.context_id = context_id
            if self._scheduler is not None:
                self._scheduler.reset(context_id)
        if self._scheduler is None or self._scheduler.total_steps != max(1, int(total_steps or 1)):
            self._scheduler = CaptureScheduler(total_steps, self.options.update_rate, self.options.target_updates)
            self._scheduler.reset(self.context_id)
        if not self._scheduler.should_capture(step, context_id=self.context_id, force_refresh=force_refresh):
            return False
        try:
            import torch

            if os.getenv("WANGP_PREVIEW_TRACE"):
                detached = latent.detach()
                print(
                    "[preview] "
                    f"model={self.model_type} architecture={self.architecture} "
                    f"context={self.context_id} step={step}/{total_steps} "
                    f"pass={pass_no} window={window_no} shape={tuple(detached.shape)} "
                    f"dtype={detached.dtype} device={detached.device} "
                    f"min={float(detached.min())} max={float(detached.max())} mean={float(detached.float().mean())}"
                )
            device = str(self.options.device)
            if device == "auto":
                device = "cuda" if getattr(latent, "is_cuda", False) and torch.cuda.is_available() else "cpu"
            if self._decode_device is not None:
                device = self._decode_device
            if self._decoder is None:
                self._decoder = load_decoder(self.spec.local_path(), self.spec, device=device)
                self._decode_device = device
            parallel = device != "cpu"
            frames, decode_ms, decoded_count = decode_ltx2_latent(self._decoder, latent, spec=self.spec, max_edge=self.options.max_edge, preview_fps=self.options.preview_fps, source_fps=fps, duration_seconds=duration_seconds, parallel=parallel)
            self.sequence += 1
            context = PreviewContext(
                generation_id=self.generation_id,
                context_id=self.context_id,
                sequence=self.sequence,
                model_type=self.model_type,
                architecture=self.architecture,
                decoder_id=self.spec.decoder_id,
                step=step,
                total_steps=total_steps,
                pass_no=pass_no,
                window_no=window_no,
                fps=fps,
                duration_seconds=(decoded_count - 1) / fps if fps and fps > 0 else duration_seconds,
                metadata={"decoded_frame_count": decoded_count},
            )
            return self._worker.try_submit(PreviewJob(tuple(frames), context, self.options, decode_ms))
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                if self._decoder is not None:
                    try:
                        frames, decode_ms, decoded_count = decode_ltx2_latent(self._decoder, latent, spec=self.spec, max_edge=self.options.max_edge, preview_fps=self.options.preview_fps, source_fps=fps, duration_seconds=duration_seconds, parallel=False)
                        self.sequence += 1
                        context = PreviewContext(self.generation_id, self.context_id, self.sequence, self.model_type, self.architecture, self.spec.decoder_id, step, total_steps, pass_no, window_no, fps, (decoded_count - 1) / fps if fps and fps > 0 else duration_seconds, {"decoded_frame_count": decoded_count})
                        return self._worker.try_submit(PreviewJob(tuple(frames), context, self.options, decode_ms))
                    except Exception as retry_error:
                        exc = retry_error
                if device != "cpu":
                    try:
                        unload_decoders()
                        self._decoder = load_decoder(self.spec.local_path(), self.spec, device="cpu")
                        self._decode_device = "cpu"
                        frames, decode_ms, decoded_count = decode_ltx2_latent(self._decoder, latent, spec=self.spec, max_edge=self.options.max_edge, preview_fps=self.options.preview_fps, source_fps=fps, duration_seconds=duration_seconds, parallel=False)
                        self.sequence += 1
                        context = PreviewContext(self.generation_id, self.context_id, self.sequence, self.model_type, self.architecture, self.spec.decoder_id, step, total_steps, pass_no, window_no, fps, (decoded_count - 1) / fps if fps and fps > 0 else duration_seconds, {"decoded_frame_count": decoded_count})
                        return self._worker.try_submit(PreviewJob(tuple(frames), context, self.options, decode_ms))
                    except Exception as cpu_error:
                        exc = cpu_error
            self._disable(exc)
            return False

    def cancel(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            self._worker.invalidate(self.generation_id)

    def close(self) -> None:
        self.cancel()
        self._worker.close(wait=False)
        if self._decoder is not None:
            unload_decoders()
            self._decoder = None
