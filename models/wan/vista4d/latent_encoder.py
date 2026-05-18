from typing import Callable, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn


def patchify(x, patch_embedding, check_patchify_match=None, check_patchify_match_prefix="Patchify"):
    output_dtype = x.dtype
    input_dtype = output_dtype
    weight = getattr(patch_embedding, "weight", None)
    bias = getattr(patch_embedding, "bias", None)
    if weight is not None and getattr(weight.dtype, "is_floating_point", False):
        input_dtype = weight.dtype
    if bias is not None and getattr(bias.dtype, "is_floating_point", False):
        input_dtype = bias.dtype
    if x.dtype != input_dtype:
        x = x.to(input_dtype)
    x = patch_embedding(x)
    if x.dtype != output_dtype:
        x = x.to(output_dtype)
    b, c, f, h, w = x.shape
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    if check_patchify_match is not None and (f, h, w) != check_patchify_match:
        raise AssertionError(f"{check_patchify_match_prefix}: x={(f, h, w)} and patchify={check_patchify_match} don't match.")
    return x, (f, h, w)


class PatchEmbedding(nn.Module):
    def __init__(self, init_mode: str = "zero_init", in_channels: Optional[int] = None, wan_patch_embedding: nn.Conv3d = None):
        super().__init__()
        if init_mode not in ("zero_init", "wan_patch_embed", "wan_patch_embed_frozen"):
            raise ValueError(f"Unsupported Vista4D patch embedding init mode: {init_mode}")
        if wan_patch_embedding is None:
            raise ValueError("wan_patch_embedding is required")

        out_channels, base_in_channels, p1, p2, p3 = wan_patch_embedding.weight.shape
        if in_channels is None:
            in_channels = base_in_channels
        elif in_channels != base_in_channels:
            init_mode = "zero_init"

        if init_mode == "wan_patch_embed_frozen":
            self.patch_embedding = None
        else:
            self.patch_embedding = nn.Conv3d(in_channels, out_channels, kernel_size=(p1, p2, p3), stride=(p1, p2, p3), bias=True)
            if init_mode == "zero_init":
                nn.init.zeros_(self.patch_embedding.weight)
                nn.init.zeros_(self.patch_embedding.bias)
            else:
                self.patch_embedding.weight = nn.Parameter(wan_patch_embedding.weight.clone().detach())
                self.patch_embedding.bias = nn.Parameter(wan_patch_embedding.bias.clone().detach())

        self.init_mode = init_mode
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        wan_patch_embedding: Optional[Callable] = None,
        check_patchify_match: Optional[Tuple[int, int, int]] = None,
        check_patchify_match_prefix: str = "Patchify",
    ):
        patch_embedding = self.patch_embedding
        if patch_embedding is None:
            if wan_patch_embedding is None:
                raise ValueError("wan_patch_embedding cannot be None with init_mode='wan_patch_embed_frozen'")
            patch_embedding = wan_patch_embedding
        return patchify(x, patch_embedding, check_patchify_match=check_patchify_match, check_patchify_match_prefix=check_patchify_match_prefix)


class RGBMaskPatchEmbedding(nn.Module):
    def __init__(
        self,
        rgb_init_mode: Optional[str] = "wan_patch_embed",
        mask_init_mode: Optional[str] = None,
        wan_patch_embedding: nn.Conv3d = None,
        rgb_in_channels: Optional[int] = None,
        mask_in_channels: Optional[int] = None,
    ):
        super().__init__()
        if rgb_init_mode is not None:
            self.rgb_patchify = PatchEmbedding(init_mode=rgb_init_mode, wan_patch_embedding=wan_patch_embedding, in_channels=rgb_in_channels)
        if mask_init_mode is not None:
            self.mask_patchify = PatchEmbedding(init_mode=mask_init_mode, wan_patch_embedding=wan_patch_embedding, in_channels=mask_in_channels)
            if self.mask_patchify.init_mode != "zero_init":
                out_channels = wan_patch_embedding.weight.shape[0]
                self.projector = nn.Linear(self.mask_patchify.out_channels, out_channels, bias=True)

    def forward(
        self,
        rgb_latents: Optional[torch.Tensor] = None,
        mask_latents: Optional[torch.Tensor] = None,
        wan_patch_embedding: Optional[Callable] = None,
        check_patchify_match: Optional[Tuple[int, int, int]] = None,
        check_patchify_match_prefix: str = "Patch embedding",
    ):
        def is_batch_none(value):
            return value is None or (isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and any(item is None for item in value))

        output_latents = 0.0
        patchify_shape = None

        if hasattr(self, "rgb_patchify") and not is_batch_none(rgb_latents):
            rgb_latents, patchify_shape = self.rgb_patchify(
                rgb_latents,
                wan_patch_embedding=wan_patch_embedding,
                check_patchify_match=check_patchify_match,
                check_patchify_match_prefix=f"{check_patchify_match_prefix}, RGB",
            )
            output_latents = output_latents + rgb_latents

        if hasattr(self, "mask_patchify") and not is_batch_none(mask_latents):
            mask_latents, mask_shape = self.mask_patchify(
                mask_latents,
                wan_patch_embedding=wan_patch_embedding,
                check_patchify_match=check_patchify_match,
                check_patchify_match_prefix=f"{check_patchify_match_prefix}, mask",
            )
            if patchify_shape is None:
                patchify_shape = mask_shape
            if hasattr(self, "projector"):
                mask_latents = self.projector(mask_latents)
            output_latents = output_latents + mask_latents

        return output_latents, patchify_shape


class LatentEncoder(nn.Module):
    def __init__(
        self,
        source_init_mode: str = "wan_patch_embed",
        point_cloud_init_mode: str = "wan_patch_embed",
        mask_init_mode: str = "zero_init",
        use_source_masks: bool = True,
        use_point_cloud_masks: bool = True,
        wan_patch_embedding: nn.Conv3d = None,
        rgb_in_channels: Optional[int] = None,
        mask_in_channels: int = 2 * 4 * 8 * 8,
    ):
        super().__init__()
        self.output_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode="wan_patch_embed_frozen",
            mask_init_mode=None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=None,
        )
        self.source_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode=source_init_mode,
            mask_init_mode=mask_init_mode if use_source_masks else None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=mask_in_channels,
        )
        self.point_cloud_patch_embedding = RGBMaskPatchEmbedding(
            rgb_init_mode=point_cloud_init_mode,
            mask_init_mode=mask_init_mode if use_point_cloud_masks else None,
            wan_patch_embedding=wan_patch_embedding,
            rgb_in_channels=rgb_in_channels,
            mask_in_channels=mask_in_channels,
        )

    def forward(
        self,
        wan_patch_embedding_fn: Callable,
        x: torch.Tensor,
        source_video_latents: Optional[torch.Tensor] = None,
        source_mask_latents: Optional[torch.Tensor] = None,
        point_cloud_video_latents: Optional[torch.Tensor] = None,
        point_cloud_mask_latents: Optional[torch.Tensor] = None,
    ):
        x, patchify_shape = self.output_patch_embedding(rgb_latents=x, mask_latents=None, wan_patch_embedding=wan_patch_embedding_fn)
        source_latents, _ = self.source_patch_embedding(
            rgb_latents=source_video_latents,
            mask_latents=source_mask_latents,
            wan_patch_embedding=wan_patch_embedding_fn,
            check_patchify_match=patchify_shape,
            check_patchify_match_prefix="Source patch embedding",
        )
        point_cloud_latents, _ = self.point_cloud_patch_embedding(
            rgb_latents=point_cloud_video_latents,
            mask_latents=point_cloud_mask_latents,
            wan_patch_embedding=wan_patch_embedding_fn,
            check_patchify_match=patchify_shape,
            check_patchify_match_prefix="Point cloud patch embedding",
        )
        return x, source_latents, point_cloud_latents, patchify_shape
