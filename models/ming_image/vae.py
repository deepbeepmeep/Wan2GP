"""Ming Image VAE tiling policy on the shared Qwen Image autoencoder."""

import torch

from models.qwen.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from shared.utils.phase_progress import check_abort


class MingImageVAE(AutoencoderKLQwenImage):
    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision):
        # Match Qwen Image 2.1's memory tiers. Sizes are input/output pixels;
        # the shared autoencoder applies the same tiling to encode and decode.
        if vae_config == 0:
            vae_config = 1 if device_mem_capacity >= 16000 else 2 if device_mem_capacity >= 8000 else 3
        return True, {1: 1024, 2: 512, 3: 256}[vae_config]

    def configure_tiling(self, tile_setting):
        if tile_setting is None or (isinstance(tile_setting, int) and not isinstance(tile_setting, bool)):
            capacity = (
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1048576
                if torch.cuda.is_available() else 0
            )
            enabled, size = self.get_VAE_tile_size(tile_setting or 0, capacity, False)
        elif isinstance(tile_setting, bool):
            enabled, size = tile_setting, 256
        else:
            enabled, size = tile_setting
        self.use_tiling = bool(enabled)
        if enabled:
            self.enable_tiling(
                tile_sample_min_height=size,
                tile_sample_min_width=size,
                tile_sample_stride_height=size * 3 // 4,
                tile_sample_stride_width=size * 3 // 4,
            )

    def encode(self, x, return_dict=True):
        # The shared tiled encoder calls encoder.forward once per tile. Check
        # cancellation there without showing a reference-image progress bar.
        handle = self.encoder.register_forward_pre_hook(lambda *_: check_abort())
        try:
            return super().encode(x, return_dict=return_dict)
        finally:
            handle.remove()
            self.clear_cache()
