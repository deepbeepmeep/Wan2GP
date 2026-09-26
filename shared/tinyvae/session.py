"""Bound the number of TinyVAE decodes per denoising pass."""
import torch


class PreviewSession:
    def __init__(self, decoder, send_cmd, gen, image):
        self.decoder, self.send_cmd, self.gen, self.image = decoder, send_cmd, gen, image
        self.context = None
        self.last_step = -1

    def capture(self, latent, step, total, pass_no):
        if self.gen.get("abort", False):
            return
        context = (pass_no, total)
        if context != self.context or step < self.last_step:
            self.context, self.last_step = context, -1
        interval = max(1, (total + 6) // 7)
        if step < 0 or (step != total - 1 and step - self.last_step < interval):
            return
        self.last_step = step
        with torch.inference_mode():
            preview = self.decoder(latent, image=self.image, abort_check=lambda: self.gen.get("abort", False))
            if preview is not None:
                self.send_cmd("preview", preview)
