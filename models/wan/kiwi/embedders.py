import gc
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.utils import convert_tensor_to_image


class _KiwiBaseEmbedder(nn.Module):
    IN_DIM = 48
    DIM = 3072
    PATCH_SIZE = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(self.IN_DIM, self.DIM, kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)


class KiwiSourceEmbedder(_KiwiBaseEmbedder):
    pass


class KiwiRefEmbedder(_KiwiBaseEmbedder):
    pass


def _resolve_embedder_file(embedder_file: Optional[str]) -> Optional[str]:
    if not embedder_file:
        return None
    return fl.locate_file(embedder_file, error_if_none=False)


def _load_embedder(
    embedder_cls,
    embedder_file: str,
    device: torch.device,
    dtype: torch.dtype,
):
    model = embedder_cls()
    offload.load_model_data(model, embedder_file, writable_tensors=False)
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def _release_model(model):
    if model is None:
        return
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def build_kiwi_conditions(
    vae,
    source_frames: Optional[torch.Tensor],
    ref_images: Optional[Sequence],
    width: int,
    height: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    source_embedder_file: Optional[str] = None,
    ref_embedder_file: Optional[str] = None,
    vae_tile_size: int = 0,
):
    result = {"source_condition": None, "ref_condition": None}
    source_embedder_path = _resolve_embedder_file(source_embedder_file)
    ref_embedder_path = _resolve_embedder_file(ref_embedder_file)

    if source_embedder_path is not None and source_frames is not None:
        source = source_frames
        if source.shape[-2] != height or source.shape[-1] != width:
            source = F.interpolate(
                source.permute(1, 0, 2, 3),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3).contiguous()
        source_latents = vae.encode([source], tile_size=vae_tile_size)[0].unsqueeze(0).to(device=device, dtype=dtype)
        source_embedder = None
        try:
            source_embedder = _load_embedder(
                KiwiSourceEmbedder,
                source_embedder_path,
                device=device,
                dtype=dtype,
            )
            source_cond = source_embedder(source_latents.to(dtype=source_embedder.patch_embedding.weight.dtype)).to(dtype)
            if batch_size > 1:
                source_cond = source_cond.expand(batch_size, -1, -1, -1, -1)
            result["source_condition"] = source_cond
        finally:
            _release_model(source_embedder)

    ref_image = None
    if ref_images is not None:
        if isinstance(ref_images, (list, tuple)):
            if len(ref_images) > 0:
                ref_image = ref_images[0]
        else:
            ref_image = ref_images
    if ref_embedder_path is not None and ref_image is not None:
        if torch.is_tensor(ref_image):
            ref_image = convert_tensor_to_image(ref_image)
        if not isinstance(ref_image, Image.Image):
            ref_image = Image.fromarray(ref_image)
        ref_image = ImageOps.pad(ref_image.convert("RGB"), (width, height), color="white", centering=(0.5, 0.5))
        ref_tensor = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device=device, dtype=dtype)
        ref_latents = vae.encode([ref_tensor.unsqueeze(1)], tile_size=vae_tile_size)[0].unsqueeze(0).to(device=device, dtype=dtype)
        ref_embedder = None
        try:
            ref_embedder = _load_embedder(
                KiwiRefEmbedder,
                ref_embedder_path,
                device=device,
                dtype=dtype,
            )
            ref_cond = ref_embedder(ref_latents.to(dtype=ref_embedder.patch_embedding.weight.dtype)).to(dtype)
            if batch_size > 1:
                ref_cond = ref_cond.expand(batch_size, -1, -1, -1, -1)
            result["ref_condition"] = ref_cond
        finally:
            _release_model(ref_embedder)

    return result
