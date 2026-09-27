"""Bound the number of TinyVAE decodes per denoising pass."""
import torch
from concurrent.futures import ThreadPoolExecutor

from .media import encode_video


class PreviewSession:
    def __init__(self, decoder, send_cmd, gen, image, video=False, duration=None):
        self.decoder, self.send_cmd, self.gen, self.image = decoder, send_cmd, gen, image
        self.video, self.duration = video and not image, duration
        self.context = None
        self.last_step = -1
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Preview Encoder') if self.video else None
        self.future = None
        self.closed = False

    def _encode(self, frames, fps, context):
        cancelled = lambda: self.closed or self.gen.get('abort', False) or context != self.context
        preview = encode_video(frames, fps, cancelled)
        if preview is not None and not cancelled():
            self.send_cmd('preview', preview)

    def capture(self, latent, step, total, pass_no):
        if self.closed or self.gen.get("abort", False):
            return
        context = (pass_no, total)
        if context != self.context or step < self.last_step:
            self.context, self.last_step = context, -1
        # Preview the first completed step, then fit the remaining updates into
        # six intervals, including the final step.
        interval = max(1, (total + 4) // 6)
        if step < 0 or (self.last_step >= 0 and step != total - 1 and step - self.last_step < interval):
            return
        if self.future is not None:
            if not self.future.done() and step != total - 1:
                return
            self.future.result()
        self.last_step = step
        with torch.inference_mode():
            preview = self.decoder(latent, image=self.image, abort_check=lambda: self.gen.get("abort", False), video=self.video, duration=self.duration)
            if preview is not None:
                if self.video:
                    frames, fps = preview
                    self.future = self.executor.submit(self._encode, frames, fps, context)
                else:
                    self.send_cmd("preview", preview)

    def close(self, cancel=False):
        self.closed = cancel or self.gen.get('abort', False)
        try:
            if self.future is not None and not self.closed:
                self.future.result()
        finally:
            self.closed = True
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.future = None
