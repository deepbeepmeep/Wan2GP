from __future__ import annotations

import math
from typing import Any

import torch

from postprocessing import temporal_upsamplers as temporal_upsampler_api


RIFE_V4_FILENAME = "rife4.26.pkl"
RIFE_V3_FILENAME = "flownet.pkl"


class RifeTemporalUpsampler(temporal_upsampler_api.SimpleScaleSuffixMixin):
    METHOD = "rife"
    VERSION_V3 = "v3"
    VERSION_V4 = "v4"

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_temporal_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "RIFE",
            "config_key": "rife",
            "pos": 10,
            "method_pos": {cls.METHOD: 10},
            "methods": [("RIFE", cls.METHOD)],
            "multipliers": {cls.METHOD: (2.0, 4.0)},
            "default_temporal_upsampling": "rife2",
        }

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"version": cls.VERSION_V4}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        version = str((config or {}).get("version", cls.VERSION_V4) or "").strip().lower()
        return {"version": version if version in (cls.VERSION_V3, cls.VERSION_V4) else cls.VERSION_V4}

    def config(self, server_config: dict[str, Any] | None = None) -> dict[str, Any]:
        return temporal_upsampler_api.read_config_section(self.server_config if server_config is None else server_config, self)

    def query_download_def(self, *, enabled_only: bool = True) -> dict[str, Any]:
        return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": [""], "fileList": [[self._filename_for_version(self.config()["version"])]]}

    def query_download_defs(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        return [self.query_download_def(enabled_only=enabled_only)]

    def download(self, process_files, *, send_cmd=None, status_text: str | None = None, temporal_upsampling: str = ""):
        from shared.utils.download import process_files_def_if_needed

        return process_files_def_if_needed(self.query_download_def(enabled_only=True), send_cmd=send_cmd, status_text=status_text or "Downloading RIFE temporal upsampling model files...")

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            rife_version = gr.Dropdown(
                choices=[("RIFE HDv3 (default)", self.VERSION_V3), ("RIFE v4.26 (latest)", self.VERSION_V4)],
                value=self.normalize_config_section(config)["version"],
                label="RIFE Temporal Upsampling Model",
                interactive=not lock_config,
            )
        return [("version", rife_version)]

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.query_temporal_upsampler_def()["multipliers"][self.METHOD]:
            return f"Unknown temporal upsampling mode: {temporal_upsampling}"
        return "Temporal Upsampling can not be used with an Image" if source_is_image else ""

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, processing_device="cuda", to_uint8_callback=None, **kwargs):
        split = self.split_value(temporal_upsampling)
        if split is None:
            return sample, previous_last_frame, fps
        if split[1] not in self.query_temporal_upsampler_def()["multipliers"][self.METHOD]:
            raise ValueError(f"Unknown temporal upsampling mode: {temporal_upsampling}")
        exp = int(round(math.log2(split[1])))
        if exp <= 0:
            return sample, previous_last_frame, fps
        rife_version = self.config()["version"]
        rife_model_path = self.files_locator.locate_file(self._filename_for_version(rife_version))
        if previous_last_frame is not None and previous_last_frame.dtype != sample.dtype:
            if sample.dtype == torch.uint8:
                if to_uint8_callback is None:
                    raise RuntimeError("RIFE temporal upsampling needs a uint8 conversion callback")
                previous_last_frame = to_uint8_callback(previous_last_frame)
            else:
                previous_last_frame = previous_last_frame.float().div_(127.5).sub_(1.0)
        from postprocessing.rife.inference import temporal_interpolation

        if previous_last_frame is not None:
            sample = torch.cat([previous_last_frame, sample], dim=1)
            previous_last_frame = sample[:, -1:].clone()
            sample = temporal_interpolation(rife_model_path, sample, exp, device=processing_device, rife_version=rife_version)
            sample = sample[:, 1:]
        else:
            sample = temporal_interpolation(rife_model_path, sample, exp, device=processing_device, rife_version=rife_version)
            previous_last_frame = sample[:, -1:].clone()
        return sample, previous_last_frame, fps * 2**exp

    def _filename_for_version(self, version: str) -> str:
        return RIFE_V4_FILENAME if version == self.VERSION_V4 else RIFE_V3_FILENAME
