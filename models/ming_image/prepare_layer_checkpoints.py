"""Repack Design-Layer using its distinct weights and the shared Bailing base.

The pinned upstream mllm shards and VAE are byte-identical to Design. The
connector and MLP differ, so this builds a separate complete encoder file by
replacing those weights in the already-verified Design encoder checkpoint.
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .prepare_checkpoints import _inside, _merge, _source_files


REVISION = "9fabca8b62a67f1f53a957d46389c00451c11e52"
ENCODER = "BailingMM2-Ming-Image-Layer"
PROJECT = "ming_image_layer"
TRANSFORMER = "Ming-Image-0.1-Design-Layer"


def _prepare_encoder(root, source):
    base = _inside(root, root / "BailingMM2-Ming-Image/BailingMM2-Ming-Image_bf16.safetensors")
    target = _inside(root, root / ENCODER / f"{ENCODER}_bf16.safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Already prepared: {target}", flush=True)
        return
    replacement = {}
    source_paths = _source_files(source, "connector") + _source_files(source, "mlp")
    for path in source_paths:
        prefix = "connector." if path.parent.name == "connector" else ""
        for key, value in load_file(str(path), device="cpu").items():
            final_key = prefix + key
            if final_key in replacement:
                raise ValueError(f"duplicate Layer encoder key: {final_key}")
            replacement[final_key] = value.to(torch.bfloat16).contiguous() if value.is_floating_point() else value.contiguous()
    with safe_open(str(base), framework="pt", device="cpu") as old:
        old_keys = set(old.keys())
        if not set(replacement) <= old_keys:
            raise ValueError(f"new Layer encoder keys absent from base: {sorted(set(replacement) - old_keys)[:8]}")
        connector_keys = {key for key in old_keys if key.startswith("connector.")}
        if connector_keys != {key for key in replacement if key.startswith("connector.")}:
            raise ValueError("Layer connector key coverage differs from Design")
        tensors = {key: old.get_tensor(key) for key in old.keys()}
        tensors.update(replacement)
        temp = _inside(root, target.with_suffix(".tmp.safetensors"))
        print(f"Writing {len(tensors)} Layer encoder tensors to {temp}", flush=True)
        save_file(tensors, str(temp), metadata={"source_revision": REVISION, "shared_base": base.name})
        with safe_open(str(temp), framework="pt", device="cpu") as merged:
            if set(merged.keys()) != old_keys:
                raise ValueError("Layer encoder key coverage changed")
            for key in old.keys():
                expected = replacement[key] if key in replacement else old.get_tensor(key)
                if not torch.equal(merged.get_tensor(key), expected):
                    raise ValueError(f"Layer encoder tensor mismatch: {key}")
        temp.replace(target)
        print(f"Verified {target}", flush=True)
        del tensors
    for path in source_paths:
        path.unlink()
    for component in ("connector", "mlp"):
        index = source / component / "model.safetensors.index.json"
        if index.exists():
            index.unlink()


def _move_assets(root, source):
    assets = {
        "mllm/config.json": f"{ENCODER}/config.json",
        "mllm/preprocessor_config.json": f"{ENCODER}/preprocessor_config.json",
        "mllm/special_tokens_map.json": f"{ENCODER}/special_tokens_map.json",
        "mllm/tokenizer.json": f"{ENCODER}/tokenizer.json",
        "mllm/tokenizer_config.json": f"{ENCODER}/tokenizer_config.json",
        "connector/config.json": f"{PROJECT}/connector_config.json",
        "mlp/config.json": f"{PROJECT}/mlp_config.json",
        "transformer/config.json": f"{PROJECT}/transformer_config.json",
        "scheduler/scheduler_config.json": f"{PROJECT}/scheduler_config.json",
        "LICENSE": f"{PROJECT}/LICENSE.upstream",
    }
    for original, prepared in assets.items():
        src = _inside(root, source / original)
        dst = _inside(root, root / prepared)
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
    source = _inside(root, root / ".ming_image_layer_source")
    _prepare_encoder(root, source)
    # The released Layer transformer shards are FP32; upstream inference and
    # the requested WanGP distribution use BF16 for these same tensors.
    _merge([(_source_files(source, "transformer"), "transformer.", True)],
           root / f"{TRANSFORMER}_bf16.safetensors", root, source_revision=REVISION)
    _move_assets(root, source)


if __name__ == "__main__":
    main()
