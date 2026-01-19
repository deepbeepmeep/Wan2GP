from dataclasses import replace
import itertools
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from ..common.normalization import PixelNorm
from ..transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from .convolution import make_conv_nd
from .enums import LogVarianceType, NormLayerType, PaddingModeType
from .ops import PerChannelStatistics, patchify, unpatchify
from .resnet import ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from .tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
)
from ...types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape, VIDEO_SCALE_FACTORS


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int]] | List[Tuple[str, dict[str, Any]]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                    f"(N means + 1 logvar), got {sample.shape[1]}"
                )

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        temporal_chunk_on_gpu: bool = True,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()

        # Spatiotemporal downscaling between decoded video space and VAE latents.
        # According to the LTXV paper, the standard configuration downsamples
        # video inputs by a factor of 8 in the temporal dimension and 32 in
        # each spatial dimension (height and width). This parameter determines how
        # many video frames and pixels correspond to a single latent cell.
        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8,
            width=32,
            height=32,
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self.temporal_chunk_on_gpu = temporal_chunk_on_gpu
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        self._mask_cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache_limit = 512

        # Compute initial feature_channels by going through blocks in reverse
        # This determines the channel width at the start of the decoder
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    # def forward(self, sample: torch.Tensor, target_shape) -> torch.Tensor:
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            long_side = max(latent.shape[3], latent.shape[4])

            def enable_on_axis(axis_idx: int, factor: int) -> None:
                size = cfg.tile_size_in_pixels // factor
                overlap = cfg.tile_overlap_in_pixels // factor
                axis_length = latent.shape[axis_idx]
                lower_threshold = max(2, overlap + 1)
                tile_size = max(lower_threshold, round(size * axis_length / long_side))
                splitters[axis_idx] = split_in_spatial(tile_size, overlap)
                mappers[axis_idx] = to_mapping_operation(map_spatial_slice, factor)

            enable_on_axis(3, self.video_downscale_factors.height)
            enable_on_axis(4, self.video_downscale_factors.width)

        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = _pixel_frames_to_latent_frames(cfg.tile_size_in_frames, self.video_downscale_factors.time)
            overlap = _pixel_frames_to_latent_frames(cfg.tile_overlap_in_frames, self.video_downscale_factors.time)
            overlap = min(overlap, max(tile_size - 1, 0))
            splitters[2] = split_in_temporal(tile_size, overlap)
            mappers[2] = to_mapping_operation(map_temporal_slice, self.video_downscale_factors.time)

        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices;
        """

        # Calculate full video shape from latent shape to get spatial dimensions
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
        total_frames = full_video_shape.frames
        tiles = self._prepare_tiles(latent, tiling_config)

        temporal_groups = self._group_tiles_by_temporal_slice(tiles)

        blend_device = latent.device
        if blend_device.type == "cuda":
            blend_device = torch.device("cpu")

        chunk_device = blend_device
        if self.temporal_chunk_on_gpu and latent.device.type == "cuda":
            chunk_device = latent.device

        # State for temporal overlap handling
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = self._normalize_temporal_slice(
                temporal_group_tiles[0].out_coords[2],
                total_frames,
            )

            # Calculate the shape of the temporal buffer for this group of tiles.
            # The temporal length depends on whether this is the first tile (starts at 0) or not.
            # - First tile: (frames - 1) * scale + 1
            # - Subsequent tiles: frames * scale
            # This logic is handled by TemporalAxisMapping and reflected in out_coords.
            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=chunk_device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
                temporal_slice=curr_temporal_slice,
                total_frames=total_frames,
            )
            if chunk_device != blend_device:
                buffer = buffer.to(device=blend_device)
                curr_weights = curr_weights.to(device=blend_device)

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                # Check if current temporal slice overlaps with previous temporal slice
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    # The overlap is already masked before it reaches this step. Each tile is accumulated into buffer
                    # with its trapezoidal mask, and curr_weights accumulates the same mask. In the overlap blend we add
                    # the masked values (buffer[...]) and the corresponding weights (curr_weights[...]) into the
                    # previous buffers, then later normalize by weights.
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                        :, :, slice(0, overlap_len), :, :
                    ]

                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                        :, :, temporal_overlap_slice, :, :
                    ]

                # Yield the non-overlapping part of the previous chunk
                previous_weights = previous_weights.clamp_(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                previous_chunk[:, :, :yield_len] /= previous_weights[:, :, :yield_len]
                yield previous_chunk[:, :, :yield_len]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp_(min=1e-8)
            previous_chunk /= previous_weights
            previous_weights = None
            yield previous_chunk

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        """Group tiles by their temporal output slice."""
        if not tiles:
            return []

        groups = []
        current_slice = tiles[0].out_coords[2]
        current_group = []

        for tile in tiles:
            tile_slice = tile.out_coords[2]
            if tile_slice == current_slice:
                current_group.append(tile)
            else:
                groups.append(current_group)
                current_slice = tile_slice
                current_group = [tile]

        # Add the final group
        if current_group:
            groups.append(current_group)

        return groups

    @staticmethod
    def _normalize_temporal_slice(temporal_slice: slice, total_frames: int) -> slice:
        start = 0 if temporal_slice.start is None else temporal_slice.start
        stop = total_frames if temporal_slice.stop is None else temporal_slice.stop
        return slice(start, stop, temporal_slice.step)

    def _prepare_mask_1d(
        self,
        mask: torch.Tensor | None,
        length: int | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        cache_key = (id(mask), device, dtype)
        cached = self._mask_cache.get(cache_key)
        if cached is None or cached[0] is not mask or cached[1].numel() != mask.numel():
            if self._mask_cache_limit > 0 and len(self._mask_cache) >= self._mask_cache_limit:
                self._mask_cache.clear()
            converted = mask.to(device=device, dtype=dtype)
            self._mask_cache[cache_key] = (mask, converted)
        else:
            converted = cached[1]
        if length is not None and converted.numel() > length:
            return converted[:length]
        return converted

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        temporal_slice: slice,
        total_frames: int,
    ) -> torch.Tensor:
        """
        Decode and accumulate all tiles of a temporal group into a local buffer.
        The buffer is local to the group and always starts at time 0; temporal coordinates
        are rebased by subtracting temporal_slice.start.
        """
        weights = torch.zeros(
            (1, 1, buffer.shape[2], buffer.shape[3], buffer.shape[4]),
            device=buffer.device,
            dtype=buffer.dtype,
        )

        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            tile_temporal_slice = self._normalize_temporal_slice(tile.out_coords[2], total_frames)
            temporal_offset = tile_temporal_slice.start - temporal_slice.start
            # Use the tile's output coordinate length, not the decoded tile's length,
            # as the decoder may produce a different number of frames than expected
            expected_temporal_len = tile_temporal_slice.stop - tile_temporal_slice.start
            decoded_temporal_len = decoded_tile.shape[2]

            # Ensure we don't exceed the buffer or decoded tile bounds
            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            # Slice decoded_tile and mask to match the actual length we're writing
            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_t = self._prepare_mask_1d(
                tile.masks_1d[2],
                actual_temporal_len,
                decoded_slice.device,
                decoded_slice.dtype,
            )
            mask_h = self._prepare_mask_1d(
                tile.masks_1d[3],
                decoded_slice.shape[3],
                decoded_slice.device,
                decoded_slice.dtype,
            )
            mask_w = self._prepare_mask_1d(
                tile.masks_1d[4],
                decoded_slice.shape[4],
                decoded_slice.device,
                decoded_slice.dtype,
            )
            if mask_t is not None:
                decoded_slice *= mask_t.view(1, 1, -1, 1, 1)
            if mask_h is not None:
                decoded_slice *= mask_h.view(1, 1, 1, -1, 1)
            if mask_w is not None:
                decoded_slice *= mask_w.view(1, 1, 1, 1, -1)
            if decoded_slice.device != buffer.device or decoded_slice.dtype != buffer.dtype:
                decoded_slice = decoded_slice.to(device=buffer.device, dtype=buffer.dtype)
            decoded_h = decoded_slice.shape[3]
            decoded_w = decoded_slice.shape[4]
            del decoded_tile

            buffer[chunk_coords] += decoded_slice
            del decoded_slice

            if mask_t is not None and (mask_t.device != buffer.device or mask_t.dtype != buffer.dtype):
                mask_t = self._prepare_mask_1d(tile.masks_1d[2], actual_temporal_len, buffer.device, buffer.dtype)
            if mask_h is not None and (mask_h.device != buffer.device or mask_h.dtype != buffer.dtype):
                mask_h = self._prepare_mask_1d(tile.masks_1d[3], decoded_h, buffer.device, buffer.dtype)
            if mask_w is not None and (mask_w.device != buffer.device or mask_w.dtype != buffer.dtype):
                mask_w = self._prepare_mask_1d(tile.masks_1d[4], decoded_w, buffer.device, buffer.dtype)

            weights_slice = weights[chunk_coords]
            if mask_h is not None and mask_w is not None:
                mask_hw = mask_h.view(-1, 1) * mask_w.view(1, -1)
                if mask_t is None:
                    weights_slice += mask_hw.view(1, 1, 1, mask_hw.shape[0], mask_hw.shape[1])
                elif mask_t.numel() == 1:
                    weights_slice += mask_hw.view(1, 1, 1, mask_hw.shape[0], mask_hw.shape[1]) * mask_t
                else:
                    for t_idx in range(actual_temporal_len):
                        weights_slice[:, :, t_idx, :, :] += mask_t[t_idx] * mask_hw
                del mask_hw
            else:
                if mask_t is None:
                    if mask_h is None and mask_w is None:
                        weights_slice += 1
                    elif mask_h is not None:
                        weights_slice += mask_h.view(1, 1, 1, -1, 1)
                    else:
                        weights_slice += mask_w.view(1, 1, 1, 1, -1)
                else:
                    if mask_h is not None:
                        weights_slice += mask_t.view(1, 1, -1, 1, 1) * mask_h.view(1, 1, 1, -1, 1)
                    elif mask_w is not None:
                        weights_slice += mask_t.view(1, 1, -1, 1, 1) * mask_w.view(1, 1, 1, 1, -1)
                    else:
                        weights_slice += mask_t.view(1, 1, -1, 1, 1)
            del mask_t, mask_h, mask_w
        return weights


def decode_video(
    latent: torch.Tensor,
    video_decoder: VideoDecoder,
    tiling_config: TilingConfig | None = None,
) -> Iterator[torch.Tensor]:
    """
    Decode a video latent tensor with the given decoder.
    Args:
        latent: Tensor [c, f, h, w]
        video_decoder: Decoder module.
        tiling_config: Optional tiling settings.
    Yields:
        Decoded chunk [f, h, w, c], uint8 in [0, 255].
    """

    def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames

    if tiling_config is not None:
        for frames in video_decoder.tiled_decode(latent, tiling_config):
            yield convert_to_uint8(frames)
    else:
        decoded_video = video_decoder(latent)
        yield convert_to_uint8(decoded_video)


def _default_intervals(length: int) -> DimensionIntervals:
    return DEFAULT_SPLIT_OPERATION(length)


def _pixel_frames_to_latent_frames(frames: int, scale: int) -> int:
    frames = int(frames)
    if frames <= 0:
        return 0
    scale = max(1, int(scale))
    return (frames - 1 + scale - 1) // scale + 1


def _build_temporal_intervals(
    length: int,
    tiling_config: TilingConfig | None,
    scale: int,
) -> DimensionIntervals:
    if (
        tiling_config is None
        or tiling_config.temporal_config is None
        or length <= 0
    ):
        return _default_intervals(length)
    cfg = tiling_config.temporal_config
    tile_size = max(1, _pixel_frames_to_latent_frames(cfg.tile_size_in_frames, scale))
    overlap = max(0, _pixel_frames_to_latent_frames(cfg.tile_overlap_in_frames, scale))
    overlap = min(overlap, max(tile_size - 1, 0))
    return split_in_temporal(tile_size, overlap)(length)


def _build_spatial_intervals(
    length: int,
    tiling_config: TilingConfig | None,
    factor: int,
    long_side: int,
) -> DimensionIntervals:
    if (
        tiling_config is None
        or tiling_config.spatial_config is None
        or length <= 0
    ):
        return _default_intervals(length)
    cfg = tiling_config.spatial_config
    size = cfg.tile_size_in_pixels // factor
    overlap = cfg.tile_overlap_in_pixels // factor
    lower_threshold = max(2, overlap + 1)
    tile_size = max(lower_threshold, round(size * length / max(1, long_side)))
    return split_in_spatial(tile_size, overlap)(length)


def _make_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
    *,
    left_starts_from_0: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    mask = compute_trapezoidal_mask_1d(length, left_ramp, right_ramp, left_starts_from_0)
    return mask.to(device=device, dtype=dtype)


def encode_video(
    video: torch.Tensor,
    video_encoder: VideoEncoder,
    tiling_config: TilingConfig | None = None,
) -> torch.Tensor:
    """
    Encode a video tensor with the given encoder, optionally using spatial/temporal tiling.
    Args:
        video: Tensor [b, c, f, h, w]
        video_encoder: Encoder module.
        tiling_config: Optional tiling settings.
    Returns:
        Encoded latent tensor [b, c_l, f_l, h_l, w_l].
    """
    if tiling_config is None or (
        tiling_config.spatial_config is None and tiling_config.temporal_config is None
    ):
        return video_encoder(video)

    b, _, frames, height, width = video.shape
    scale = VIDEO_SCALE_FACTORS
    latent_shape = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(batch=b, frames=frames, height=height, width=width, fps=1.0),
        latent_channels=getattr(video_encoder, "latent_channels", 128),
        scale_factors=scale,
    )

    long_side = max(latent_shape.height, latent_shape.width)
    temporal_intervals = _build_temporal_intervals(latent_shape.frames, tiling_config, scale.time)
    height_intervals = _build_spatial_intervals(latent_shape.height, tiling_config, scale.height, long_side)
    width_intervals = _build_spatial_intervals(latent_shape.width, tiling_config, scale.width, long_side)

    output = None
    weights = None

    for t_idx, h_idx, w_idx in itertools.product(
        range(len(temporal_intervals.starts)),
        range(len(height_intervals.starts)),
        range(len(width_intervals.starts)),
    ):
        t_start = temporal_intervals.starts[t_idx]
        t_end = temporal_intervals.ends[t_idx]
        t_left = temporal_intervals.left_ramps[t_idx]
        t_right = temporal_intervals.right_ramps[t_idx]
        h_start = height_intervals.starts[h_idx]
        h_end = height_intervals.ends[h_idx]
        h_left = height_intervals.left_ramps[h_idx]
        h_right = height_intervals.right_ramps[h_idx]
        w_start = width_intervals.starts[w_idx]
        w_end = width_intervals.ends[w_idx]
        w_left = width_intervals.left_ramps[w_idx]
        w_right = width_intervals.right_ramps[w_idx]

        t_slice, _ = map_temporal_slice(t_start, t_end, t_left, t_right, scale.time)
        h_slice, _ = map_spatial_slice(h_start, h_end, h_left, h_right, scale.height)
        w_slice, _ = map_spatial_slice(w_start, w_end, w_left, w_right, scale.width)

        tile = video[:, :, t_slice, h_slice, w_slice]
        encoded_tile = video_encoder(tile)

        t_len = t_end - t_start
        h_len = h_end - h_start
        w_len = w_end - w_start
        encoded_tile = encoded_tile[:, :, :t_len, :h_len, :w_len]

        if output is None:
            output = torch.zeros(
                latent_shape.to_torch_shape(),
                device=encoded_tile.device,
                dtype=encoded_tile.dtype,
            )
            weights = torch.zeros(
                (1, 1, latent_shape.frames, latent_shape.height, latent_shape.width),
                device=encoded_tile.device,
                dtype=encoded_tile.dtype,
            )

        mask_t = _make_mask_1d(
            t_len,
            t_left,
            t_right,
            left_starts_from_0=True,
            device=encoded_tile.device,
            dtype=encoded_tile.dtype,
        )
        mask_h = _make_mask_1d(
            h_len,
            h_left,
            h_right,
            left_starts_from_0=False,
            device=encoded_tile.device,
            dtype=encoded_tile.dtype,
        )
        mask_w = _make_mask_1d(
            w_len,
            w_left,
            w_right,
            left_starts_from_0=False,
            device=encoded_tile.device,
            dtype=encoded_tile.dtype,
        )
        mask = mask_t[:, None, None] * mask_h[None, :, None] * mask_w[None, None, :]
        mask = mask.unsqueeze(0).unsqueeze(0)

        output[:, :, t_start:t_end, h_start:h_end, w_start:w_end] += encoded_tile * mask
        weights[:, :, t_start:t_end, h_start:h_end, w_start:w_end] += mask

    if output is None or weights is None:
        return video_encoder(video)
    weights = weights.clamp_min(1e-6)
    return output / weights


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    scale = VIDEO_SCALE_FACTORS.time
    latent_frames = _pixel_frames_to_latent_frames(num_frames, scale)
    tile_size = _pixel_frames_to_latent_frames(cfg.tile_size_in_frames, scale)
    overlap = _pixel_frames_to_latent_frames(cfg.tile_overlap_in_frames, scale)
    overlap = min(overlap, max(tile_size - 1, 0))
    if latent_frames <= tile_size:
        return 1
    intervals = split_in_temporal(tile_size, overlap)(latent_frames)
    return len(intervals.starts)


def split_in_spatial(size: int, overlap: int) -> SplitOperation:
    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_in_temporal(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_in_spatial(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:
    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
        number_of_slices = len(intervals.starts)
        for i in range(number_of_slices):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True)


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False)
