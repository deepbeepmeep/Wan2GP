"""Memory-conscious FirstBlockCache state for MiniMax H3."""

import torch


class MiniMaxH3FirstBlockCache:
    """Use a compact first-block residual signature to decide whether to reuse the cached block-stack tail."""

    MAX_SIGNATURE_ELEMENTS = 1 << 20

    def __init__(self, cache):
        self.cache = cache
        self.threshold = float(cache.threshold)
        self.start_step = int(cache.start_step)
        self.step = -1
        self.head_signature = None
        self.tail_residual = None
        print(f"[MiniMax H3] First Block Cache enabled (threshold={self.threshold:g})")

    def begin_step(self, step):
        self.step = step

    def should_compute(self, signature):
        compute = self.step < self.start_step or self.head_signature is None or self.tail_residual is None
        if not compute:
            previous = self.head_signature
            difference = (signature - previous).abs().mean(dtype=torch.float32)
            reference = previous.abs().mean(dtype=torch.float32).clamp_min_(1e-8)
            compute = bool((difference / reference).item() > self.threshold)
        if compute:
            self.head_signature = signature
            self.tail_residual = None
        else:
            self.cache.skipped_steps += 1
        return compute

    def capture_head_output(self, target_hidden):
        # The final projection discards the packed prefix, so the tail cache only needs generated audio/video rows.
        return target_hidden.clone()

    def store_tail_residual(self, target_hidden, head_output):
        head_output.neg_().add_(target_hidden)
        self.tail_residual = head_output

    def apply_tail_residual(self, target_hidden):
        target_hidden.add_(self.tail_residual)

    def reset(self):
        self.head_signature = None
        self.tail_residual = None
        self.step = -1


__all__ = ["MiniMaxH3FirstBlockCache"]
