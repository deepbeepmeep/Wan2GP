from __future__ import annotations

from typing import Any, Callable

import torch

from postprocessing.pid.runtime import (
    PID_TEXT_ENCODER_FILES,
    PID_TEXT_ENCODER_FOLDER,
    PID_TILING_THRESHOLD_DEFAULT,
    get_pid_download_def,
    get_pid_upsampler,
    is_pid_post_upsampling,
    pid_checkpoint_types_for_tiling_threshold,
    pid_backbone_for_upsampling,
    pid_checkpoint_filename,
    pid_post_upsampling_choices,
    pid_vae_filename,
)


class PiDBridge:
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    UPSAMPLING_RATIOS = (4.0,)
    batch_image_inputs = True
    uses_image_profile = True

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_edit_mode_def(cls, include_name: bool = True) -> dict[str, Any]:
        return {
            "name": "PiD",
            "spatial_upsampling_choices": pid_post_upsampling_choices(include_name=include_name),
            "default_spatial_upsampling": "flux_pid4",
        }

    def is_upsampling(self, spatial_upsampling) -> bool:
        return is_pid_post_upsampling(spatial_upsampling)

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if image_mode == 0:
            return "PiD Spatial Upsampling is only available for Images"
        return ""

    def _required_files(self, backbone):
        ckpt_types = pid_checkpoint_types_for_tiling_threshold(self.server_config.get("pid_tiling_threshold", PID_TILING_THRESHOLD_DEFAULT))
        required = [pid_checkpoint_filename(backbone, ckpt_type) for ckpt_type in ckpt_types]
        required.append(pid_vae_filename(backbone))
        required.extend([f"{PID_TEXT_ENCODER_FOLDER}/{filename}" for filename in PID_TEXT_ENCODER_FILES])
        return required

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        backbone = pid_backbone_for_upsampling(spatial_upsampling)
        if all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in self._required_files(backbone)):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        ckpt_types = pid_checkpoint_types_for_tiling_threshold(self.server_config.get("pid_tiling_threshold", PID_TILING_THRESHOLD_DEFAULT))
        for download_def in get_pid_download_def(backbone, ckpt_type=ckpt_types, include_vae=True):
            process_files(**download_def)
        return True

    def _prepare_sample(self, sample, device, dtype):
        frames = sample.transpose(0, 1).contiguous().to(device=device)
        if frames.dtype == torch.uint8:
            frames = frames.to(dtype=dtype).div_(127.5).sub_(1.0)
        else:
            frames = frames.to(dtype=dtype)
        return frames

    def upscale(
        self,
        sample,
        spatial_upsampling,
        *,
        seed=0,
        continue_cache=None,
        return_continue_cache=False,
        vae_tile_size=None,
        process_files: Callable[..., Any],
        vae_config: int,
        init_pipe: Callable[..., int],
        profile,
        still_image=False,
        abort_callback=None,
        progress_callback=None,
    ):
        if not self.is_upsampling(spatial_upsampling):
            raise ValueError(f"Unknown PiD upsampling mode: {spatial_upsampling}")
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        backbone = pid_backbone_for_upsampling(spatial_upsampling)
        persistent = int(self.server_config.get("pid_persistence", self.PERSIST_UNLOAD) or self.PERSIST_UNLOAD) == self.PERSIST_RAM
        session = get_pid_upsampler(
            backbone,
            None,
            init_pipe=init_pipe,
            profile=profile,
            main_offloadobj=None,
            persistent_models=persistent,
            tiling_threshold=self.server_config.get("pid_tiling_threshold", PID_TILING_THRESHOLD_DEFAULT),
            attention_mode=self.server_config.get("attention_mode"),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = self._prepare_sample(sample, device, session.dtype)
        output = session.decode(
            images,
            None,
            prompt="",
            seed=seed,
            vae_encode=True,
            abort_callback=abort_callback,
            progress_callback=progress_callback,
        )
        images = None
        if output is None:
            return None, None
        output = output.to("cpu").transpose(0, 1).contiguous()
        return output, None

    def release_vram(self) -> None:
        from postprocessing.pid.runtime import release_models

        release_models()
