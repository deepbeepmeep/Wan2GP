from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .modules.layers import RMSNorm


class RadianceEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int, *, dtype: torch.dtype | None = torch.float32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size_input = hidden_size_input
        self.max_freqs = max_freqs
        self.embedder_dtype = dtype
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True))
        if dtype is not None:
            self.embedder.to(dtype=dtype)

    @lru_cache(maxsize=4)
    def _fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, device=device, dtype=dtype)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1

        dct = (torch.cos(pos_x * freqs_x * torch.pi) * torch.cos(pos_y * freqs_y * torch.pi) * coeffs).view(
            1, -1, self.max_freqs**2
        )
        return dct

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, pixels, _ = inputs.shape
        patch_size = int(pixels**0.5)
        original_dtype = inputs.dtype
        target_dtype = self.embedder[0].weight.dtype

        inputs_cast = inputs.to(target_dtype)
        pos = self._fetch_pos(patch_size, inputs.device, target_dtype).repeat(batch, 1, 1)
        combined = torch.cat((inputs_cast, pos), dim=-1)
        embedded = self.embedder(combined)
        return embedded.to(original_dtype)


class RadianceGLUBlock(nn.Module):
    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int):
        super().__init__()
        total_params = 3 * hidden_size_x**2 * mlp_ratio
        self.param_generator = nn.Linear(hidden_size_s, total_params)
        self.norm = RMSNorm(hidden_size_x)
        self.mlp_ratio = mlp_ratio

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch, pixels, hidden = x.shape
        params = self.param_generator(s)
        gate, value, proj = params.chunk(3, dim=-1)

        gate = torch.nn.functional.normalize(gate.view(batch, hidden, hidden * self.mlp_ratio), dim=-2)
        value = torch.nn.functional.normalize(value.view(batch, hidden, hidden * self.mlp_ratio), dim=-2)
        proj = torch.nn.functional.normalize(proj.view(batch, hidden * self.mlp_ratio, hidden), dim=-2)

        residual = x
        x = self.norm(x)
        activated = torch.nn.functional.silu(torch.bmm(x, gate))
        gated = activated * torch.bmm(x, value)
        x = torch.bmm(gated, proj)
        return x + residual


class RadianceFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x.movedim(1, -1))).movedim(-1, 1)


class RadianceFinalLayerConv(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x.movedim(1, -1)).movedim(-1, 1))


@dataclass
class RadianceHead:
    patch_size: int
    img_in_patch: nn.Conv2d
    nerf_image_embedder: RadianceEmbedder
    nerf_blocks: nn.ModuleList
    nerf_final_layer: nn.Module | None
    nerf_final_layer_conv: nn.Module | None


def inject_radiance_modules(module: nn.Module, params) -> RadianceHead:
    patch_size = params.radiance_patch_size
    img_in_patch = nn.Conv2d(
        params.out_channels,
        params.hidden_size,
        kernel_size=patch_size,
        stride=patch_size,
        bias=True,
    )
    nn.init.zeros_(img_in_patch.weight)
    nn.init.zeros_(img_in_patch.bias)

    nerf_image_embedder = RadianceEmbedder(
        params.out_channels,
        params.radiance_hidden_size,
        params.radiance_max_freqs,
        dtype=torch.float32,
    )
    nerf_blocks = nn.ModuleList(
        [
            RadianceGLUBlock(
                hidden_size_s=params.hidden_size,
                hidden_size_x=params.radiance_hidden_size,
                mlp_ratio=params.radiance_mlp_ratio,
            )
            for _ in range(params.radiance_depth)
        ]
    )

    final_layer = None
    final_layer_conv = None
    if params.radiance_final_head_type == "linear":
        final_layer = RadianceFinalLayer(
            params.radiance_hidden_size,
            out_channels=params.out_channels,
        )
    elif params.radiance_final_head_type == "conv":
        final_layer_conv = RadianceFinalLayerConv(
            params.radiance_hidden_size,
            out_channels=params.out_channels,
        )
    else:
        raise ValueError(f"Unsupported radiance_final_head_type: {params.radiance_final_head_type}")

    head = RadianceHead(
        patch_size=patch_size,
        img_in_patch=img_in_patch,
        nerf_image_embedder=nerf_image_embedder,
        nerf_blocks=nerf_blocks,
        nerf_final_layer=final_layer,
        nerf_final_layer_conv=final_layer_conv,
    )

    module.patch_size = head.patch_size
    module.img_in_patch = head.img_in_patch
    module.nerf_image_embedder = head.nerf_image_embedder
    module.nerf_blocks = head.nerf_blocks
    module.nerf_final_layer = head.nerf_final_layer
    module.nerf_final_layer_conv = head.nerf_final_layer_conv

    return head


def _apply_nerf_blocks(
    module: nn.Module,
    hidden_seq: torch.Tensor,
    nerf_pixels: torch.Tensor,
) -> torch.Tensor:
    embed = module.nerf_image_embedder(nerf_pixels)
    for block in module.nerf_blocks:
        embed = block(embed, hidden_seq)
    return embed


def apply_radiance_head(
    module: nn.Module,
    hidden_seq: torch.Tensor,
    base_image: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    patch_size = module.patch_size
    out_channels = module.out_channels
    batch, num_patches, hidden = hidden_seq.shape

    nerf_hidden = hidden_seq.reshape(batch * num_patches, hidden)
    nerf_pixels = F.unfold(base_image, kernel_size=patch_size, stride=patch_size)
    nerf_pixels = nerf_pixels.transpose(1, 2)  # (B, NumPatches, C * P * P)
    nerf_pixels = nerf_pixels.reshape(batch * num_patches, out_channels, patch_size**2).transpose(1, 2)

    tile_size = getattr(module.params, "radiance_tile_size", 0)
    if tile_size > 0 and num_patches > tile_size:
        outputs = []
        for start in range(0, num_patches, tile_size):
            end = min(start + tile_size, num_patches)
            hidden_tile = nerf_hidden[start * batch : end * batch]
            pixel_tile = nerf_pixels[start * batch : end * batch]
            outputs.append(_apply_nerf_blocks(module, hidden_tile, pixel_tile))
        embed = torch.cat(outputs, dim=0)
    else:
        embed = _apply_nerf_blocks(module, nerf_hidden, nerf_pixels)

    embed = embed.transpose(1, 2)  # (B*num_patches, hidden_size_x, patch_size^2)
    embed = embed.reshape(batch, num_patches, -1).transpose(1, 2)  # (B, hidden_size_x*patch_size^2, NumPatches)
    image = F.fold(embed, output_size=(height, width), kernel_size=patch_size, stride=patch_size)

    final_layer = module.nerf_final_layer_conv or module.nerf_final_layer
    if final_layer is None:
        raise RuntimeError("Radiance head is missing a final projection layer.")
    image = final_layer(image)

    tokens = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
    return tokens
