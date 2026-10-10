"""Repack the pinned inclusionAI Design checkpoint into WanGP components.

Usage: python -m models.ming_image.prepare_checkpoints D:/ml/wangp/ckpts
The upstream snapshot must be in <root>/.ming_image_source.  Files are
verified tensor by tensor before the source shards are removed.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


TEXT = "BailingMM2-Ming-Image"
TRANSFORMER = "Ming-Image-0.1-Design"
PROJECT = "ming_image"


def _inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes checkpoint root: {path}")
    return resolved


def _source_files(source, component):
    files = sorted((source / component).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no {component} safetensors under {source}")
    return files


def _merge(groups, target, root, source_revision="208087ada1486931692c1896f38d4cd16ff3df82"):
    """Groups are (paths, key prefix, convert to BF16)."""
    _inside(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.safetensors")
    _inside(root, temp)
    if target.exists():
        print(f"Already prepared: {target}", flush=True)
        return
    tensors = {}
    expected = {}
    originals = []
    for files, prefix, cast_bf16 in groups:
        for path in files:
            _inside(root, path)
            print(f"Reading {path.name}", flush=True)
            state = load_file(str(path), device="cpu")
            for key, value in state.items():
                final_key = prefix + key
                if final_key in tensors:
                    raise ValueError(f"duplicate tensor {final_key}")
                dtype = torch.bfloat16 if cast_bf16 and value.is_floating_point() else value.dtype
                tensors[final_key] = value.to(dtype).contiguous()
                expected[final_key] = (tuple(value.shape), dtype)
            originals.append(path)
            del state
    print(f"Writing {len(tensors)} tensors to {temp}", flush=True)
    save_file(tensors, str(temp), metadata={"source_revision": source_revision})
    with safe_open(str(temp), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("merged tensor keys differ from input")
        for key, (shape, dtype) in expected.items():
            tensor = handle.get_slice(key)
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != {
                torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64",
            }.get(dtype, str(dtype)):
                raise ValueError(f"merged tensor metadata mismatch: {key}")
            if not torch.equal(handle.get_tensor(key), tensors[key]):
                raise ValueError(f"merged tensor data mismatch: {key}")
    temp.replace(target)
    print(f"Verified {target}", flush=True)
    del tensors
    for path in originals:
        path.unlink()
    for files, _, _ in groups:
        for path in files:
            index = path.parent / "model.safetensors.index.json"
            if index.exists():
                index.unlink()
            index = path.parent / "diffusion_pytorch_model.safetensors.index.json"
            if index.exists():
                index.unlink()


def _move_assets(source, root):
    assets = {
        "mllm/config.json": f"{TEXT}/config.json",
        "mllm/preprocessor_config.json": f"{TEXT}/preprocessor_config.json",
        "mllm/special_tokens_map.json": f"{TEXT}/special_tokens_map.json",
        "mllm/tokenizer.json": f"{TEXT}/tokenizer.json",
        "mllm/tokenizer_config.json": f"{TEXT}/tokenizer_config.json",
        "connector/config.json": f"{PROJECT}/connector_config.json",
        "mlp/config.json": f"{PROJECT}/mlp_config.json",
        "transformer/config.json": f"{PROJECT}/transformer_config.json",
        "vae/config.json": f"{PROJECT}/vae_config.json",
        "scheduler/scheduler_config.json": f"{PROJECT}/scheduler_config.json",
        "vae/diffusion_pytorch_model.safetensors": f"{PROJECT}/vae.safetensors",
        "LICENSE": f"{PROJECT}/LICENSE.upstream",
    }
    for original, prepared in assets.items():
        src, dst = _inside(root, source / original), _inside(root, root / prepared)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            if not dst.exists():
                raise FileNotFoundError(src)
            continue
        if dst.exists():
            raise FileExistsError(dst)
        src.replace(dst)
        print(f"Moved {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    args = parser.parse_args()
    root = args.checkpoint_root.resolve()
    source = _inside(root, root / ".ming_image_source")
    _merge([
        (_source_files(source, "mllm"), "", False),
        (_source_files(source, "connector"), "connector.", True),
        (_source_files(source, "mlp"), "", True),
    ], root / TEXT / f"{TEXT}_bf16.safetensors", root)
    _merge([
        (_source_files(source, "transformer"), "transformer.", False),
    ], root / f"{TRANSFORMER}_bf16.safetensors", root)
    _move_assets(source, root)


if __name__ == "__main__":
    main()
