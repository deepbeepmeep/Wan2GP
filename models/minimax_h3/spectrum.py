"""MiniMax H3 adapter for the shared Spectrum feature forecaster."""

from shared.steps_skipping.spectrum import SpectrumConfig, SpectrumFeatureForecaster, SpectrumSegment


# Internal developer switch. True uses isolated capture plus full-anchor offline replay;
# False restores the original degree-4 causal forecaster with its rolling eight anchors.
FULL_ANCHOR_CACHE = True

AUDIO_BLEND_WEIGHT = 0.0
VIDEO_BLEND_WEIGHT = 0.5
LEGACY_DEGREE = 4
LEGACY_BLEND_WEIGHT = 0.5
RES_MULTISTEP_TAIL_ACTUAL_STEPS = 3


class MiniMaxH3Spectrum(SpectrumFeatureForecaster):
    """Apply shared Spectrum replay to H3's contiguous audio and video target features."""

    def __init__(self, cache, sigmas, sample_solver="euler"):
        self.cache = cache
        tail_actual_steps = RES_MULTISTEP_TAIL_ACTUAL_STEPS if FULL_ANCHOR_CACHE and sample_solver == "res_multistep" else 1
        config = SpectrumConfig(full_anchor_cache=FULL_ANCHOR_CACHE, legacy_degree=LEGACY_DEGREE,
                                legacy_blend_weight=LEGACY_BLEND_WEIGHT, tail_actual_steps=tail_actual_steps)
        super().__init__(sigmas, cache.start_step, config)

    def observe(self, feature, target_audio_rows, abort_callback):
        audio_elements = int(target_audio_rows) * feature.shape[-1]
        feature_elements = feature.numel()
        segments = []
        if audio_elements:
            segments.append(SpectrumSegment(0, audio_elements, AUDIO_BLEND_WEIGHT))
        if audio_elements < feature_elements:
            segments.append(SpectrumSegment(audio_elements, feature_elements, VIDEO_BLEND_WEIGHT))
        super().observe(feature, abort_callback, segments)

    def finish_step(self):
        skipped = not self.replaying and self.forecasting
        super().finish_step()
        if skipped:
            self.cache.skipped_steps += 1


__all__ = ["FULL_ANCHOR_CACHE", "MiniMaxH3Spectrum"]
