"""Fit Ming Image latent-to-RGB preview coefficients from local images.

This is an offline diagnostic, not part of model loading or generation.
"""

import argparse
import inspect
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from PIL import Image, ImageOps

from mmgp import offload
from shared.RGB_factors import get_rgb_factors

from .vae import MingImageVAE


def _load_vae(checkpoints):
    project = checkpoints / "ming_image"
    config = json.loads((project / "vae_config.json").read_text(encoding="utf-8"))
    supported = inspect.signature(MingImageVAE.__init__).parameters
    with torch.device("cpu"), init_empty_weights(include_buffers=True):
        vae = MingImageVAE(**{key: value for key, value in config.items() if key in supported})
    vae.register_to_config(scaling_factor=config["scaling_factor"],
                           shift_factor=config["shift_factor"])
    offload.load_model_data(vae, str(project / "vae.safetensors"),
                            writable_tensors=False, default_dtype=None)
    return vae.eval().requires_grad_(False).to("cuda")


def _mse(xtx, xty, yty, coefficients, count):
    error = (coefficients.T @ xtx @ coefficients - 2 * coefficients.T @ xty + yty).trace()
    return (error / (count * 3)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    paths = sorted(args.images.glob("*.jpg"))
    if len(paths) < args.count or args.count < 20:
        raise ValueError(f"Need at least {args.count} JPEGs (found {len(paths)})")
    paths = paths[:args.count]
    random.Random(20260923).shuffle(paths)
    split = args.count * 4 // 5
    vae = _load_vae(args.checkpoints)
    vae.disable_tiling()
    dtype = vae.dtype
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    accum = [
        [torch.zeros((17, 17), dtype=torch.float64),
         torch.zeros((17, 3), dtype=torch.float64),
         torch.zeros((3, 3), dtype=torch.float64), 0]
        for _ in range(2)
    ]
    examples = []
    with torch.inference_mode():
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                rgb = ImageOps.fit(source.convert("RGB"), (args.size, args.size),
                                   method=Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.asarray(rgb).copy()).to(device=device, dtype=dtype)
            pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).div_(127.5).sub_(1)
            rgba = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
            z = vae.encode(rgba).latent_dist.mode()
            z = (z - vae.config.shift_factor) * vae.config.scaling_factor
            target = F.interpolate(pixels[:, :, 0].float(), size=z.shape[-2:], mode="area")
            design = z[0, :, 0].permute(1, 2, 0).reshape(-1, 16).float().cpu().double()
            design = torch.cat((design, torch.ones((design.shape[0], 1), dtype=torch.float64)), dim=1)
            target = target[0].permute(1, 2, 0).reshape(-1, 3).cpu().double()
            xtx, xty, yty, count = accum[index >= split]
            xtx.add_(design.T @ design)
            xty.add_(design.T @ target)
            yty.add_(target.T @ target)
            accum[index >= split][3] = count + design.shape[0]
            if index >= split and len(examples) < 6 and (index - split) % max(1, (args.count - split) // 6) == 0:
                examples.append((rgb.copy(), design.float()))
            del pixels, rgba, z, target, design
            if (index + 1) % 50 == 0 or index + 1 == len(paths):
                print(f"Encoded {index + 1}/{len(paths)} images", flush=True)
    train_xx, train_xy, _, train_count = accum[0]
    valid_xx, valid_xy, valid_yy, valid_count = accum[1]
    ridge = torch.eye(17, dtype=torch.float64) * (train_xx.trace() / 17) * 1e-8
    ridge[-1, -1] = 0
    coefficients = torch.linalg.solve(train_xx + ridge, train_xy)
    old_factors, old_bias = get_rgb_factors("qwen")
    old = torch.tensor(old_factors + [old_bias], dtype=torch.float64)
    result = {
        "source": str(args.images), "image_count": args.count, "train_images": split,
        "validation_images": args.count - split, "input_size": args.size,
        "train_pixels": train_count, "validation_pixels": valid_count,
        "checkpoint": str(args.checkpoints / "ming_image/vae.safetensors"),
        "latent_scaling_factor": vae.config.scaling_factor,
        "validation_mse": _mse(valid_xx, valid_xy, valid_yy, coefficients, valid_count),
        "qwen_factor_validation_mse": _mse(valid_xx, valid_xy, valid_yy, old, valid_count),
        "rgb_factors": coefficients[:16].tolist(), "rgb_bias": coefficients[16].tolist(),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1048576,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for number, (rgb, design) in enumerate(examples):
        preview = (design.double() @ coefficients).clamp(-1, 1).reshape(args.size // 8, args.size // 8, 3)
        preview = ((preview.numpy() + 1) * 127.5).astype(np.uint8)
        preview_image = Image.fromarray(preview).resize((args.size, args.size), Image.Resampling.BILINEAR)
        sheet = Image.new("RGB", (args.size * 2, args.size))
        sheet.paste(rgb, (0, 0))
        sheet.paste(preview_image, (args.size, 0))
        sheet.save(args.output.with_name(f"{args.output.stem}_sample_{number + 1}.png"))
    print(json.dumps({key: value for key, value in result.items() if key not in {"rgb_factors", "rgb_bias"}}, indent=2))


if __name__ == "__main__":
    main()
