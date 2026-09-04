from __future__ import annotations

from postprocessing import spatial_upsamplers as spatial_api
from postprocessing import temporal_upsamplers as api
from .runtime import dlssg_capabilities, frame_generate, is_rtx_50_series, release_flow_model, unavailable_reason
from .spatial_upsampler import DLSS5SpatialUpsampler


class DLSSGTemporalUpsampler(api.SimpleScaleSuffixMixin):
    METHOD = "dlssg"
    MULTIPLIERS = (2.0, 3.0, 4.0, 5.0, 6.0)

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config

    def config(self):
        return DLSS5SpatialUpsampler.normalize_config_section(spatial_api.read_config_section_by_key(self.server_config, "dlss5"))

    def query_temporal_upsampler_def(self):
        reason = unavailable_reason(temporal=True)
        capabilities = dlssg_capabilities() if not reason else {}
        worker_version = int(capabilities.get("worker_version", 0))
        maximum = int(capabilities.get("multi_frame_count_max", 1))
        hardware_limit = 6 if is_rtx_50_series() else 4
        multipliers = tuple(scale for scale in self.MULTIPLIERS if scale <= min(4, hardware_limit))
        if capabilities:
            multipliers = tuple(scale for scale in self.MULTIPLIERS if scale <= hardware_limit and scale - 1 <= maximum) if worker_version >= 2 else (2.0,)
        if not reason and worker_version < 2:
            reason = "worker v2 required for x3+"
        elif not reason and maximum + 1 < hardware_limit:
            reason = f"GPU/runtime supports up to x{maximum + 1}"
        label = "DLSS Frame Generation" + (f" ({reason})" if reason else "")
        return {
            "name": "DLSS Frame Generation",
            "config_key": "dlssg",
            "pos": 10_000,
            "method_pos": {self.METHOD: 10_000},
            "methods": [(label, self.METHOD)],
            "multipliers": {self.METHOD: multipliers},
            "default_temporal_upsampling": "dlssg*2",
            "description": "NVIDIA DLSS Frame Generation using video motion estimation. It provides x2-x4 interpolation and capability-gated x5/x6 on RTX 50 series. Requires Windows 11, HAGS, and GeForce RTX 40 or newer.",
        }

    def validate_upsampling(self, temporal_upsampling, *, source_is_image=False):
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            return f"Unknown DLSS Frame Generation mode: {temporal_upsampling}"
        if source_is_image:
            return "Temporal Upsampling can not be used with an Image"
        reason = unavailable_reason(temporal=True)
        if reason:
            return f"DLSS Frame Generation is unavailable: {reason}. See docs/DLSS5.md."
        capabilities = dlssg_capabilities()
        worker_version = int(capabilities.get("worker_version", 0))
        maximum = int(capabilities.get("multi_frame_count_max", 1))
        if split[1] > 4 and not is_rtx_50_series():
            return "DLSS Frame Generation x5/x6 requires a GeForce RTX 50-series GPU."
        if split[1] > 2 and worker_version < 2:
            return "DLSS Frame Generation x3-x6 requires worker v2. See docs/DLSS5.md."
        if split[1] - 1 > maximum:
            return f"DLSS Frame Generation on this GPU supports up to x{maximum + 1}."
        return ""

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, abort_callback=None, progress_callback=None, **kwargs):
        error = self.validate_upsampling(temporal_upsampling)
        if error:
            raise RuntimeError(error)
        scale = int(self.split_value(temporal_upsampling)[1])
        return frame_generate(sample, previous_last_frame, fps, scale, motion_vector=self.config()["motion_vector"], abort_callback=abort_callback, progress_callback=progress_callback)

    def release_vram(self):
        release_flow_model()
