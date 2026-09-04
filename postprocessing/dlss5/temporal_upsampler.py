from __future__ import annotations

from postprocessing import spatial_upsamplers as spatial_api
from postprocessing import temporal_upsamplers as api
from .runtime import frame_generate, release_flow_model, unavailable_reason
from .spatial_upsampler import DLSS5SpatialUpsampler


class DLSSGTemporalUpsampler(api.SimpleScaleSuffixMixin):
    METHOD = "dlssg"
    MULTIPLIERS = (2.0, 4.0)

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config

    def config(self):
        return DLSS5SpatialUpsampler.normalize_config_section(spatial_api.read_config_section_by_key(self.server_config, "dlss5"))

    def query_temporal_upsampler_def(self):
        reason = unavailable_reason(temporal=True)
        label = "DLSS Frame Generation" + (f" ({reason})" if reason else "")
        return {
            "name": "DLSS Frame Generation",
            "config_key": "dlssg",
            "pos": 10_000,
            "method_pos": {self.METHOD: 10_000},
            "methods": [(label, self.METHOD)],
            "multipliers": {self.METHOD: self.MULTIPLIERS},
            "default_temporal_upsampling": "dlssg2",
            "description": "NVIDIA DLSS Frame Generation with native x2 or streaming two-stage x4 interpolation. Requires Windows 11, HAGS, and GeForce RTX 40 or newer.",
        }

    def validate_upsampling(self, temporal_upsampling, *, source_is_image=False):
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            return f"Unknown DLSS Frame Generation mode: {temporal_upsampling}"
        if source_is_image:
            return "Temporal Upsampling can not be used with an Image"
        reason = unavailable_reason(temporal=True)
        return f"DLSS Frame Generation is unavailable: {reason}. See docs/DLSS5.md." if reason else ""

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, abort_callback=None, progress_callback=None, **kwargs):
        error = self.validate_upsampling(temporal_upsampling)
        if error:
            raise RuntimeError(error)
        scale = int(self.split_value(temporal_upsampling)[1])
        return frame_generate(sample, previous_last_frame, fps, scale, motion_vector=self.config()["motion_vector"], abort_callback=abort_callback, progress_callback=progress_callback)

    def release_vram(self):
        release_flow_model()
