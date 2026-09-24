"""Split the two Ming Bailing checkpoints without materializing tensors in RAM.

The Bailing language model and image tower are byte-identical across Design
and Design-Layer. Each variant keeps its own connector and conditioning heads.
Run against the configured WanGP checkpoint root after the legacy full encoder
files have been prepared there. Existing outputs are left intact on reruns.
"""

import argparse
import json
import os
from pathlib import Path


DESIGN = "BailingMM2-Ming-Image"
LAYER = "BailingMM2-Ming-Image-Layer"
CORE = "BailingMM2-Ming-Image-Core"
CHUNK = 16 * 1024 * 1024


def read_header(path):
    with path.open("rb") as source:
        length = int.from_bytes(source.read(8), "little")
        header = json.loads(source.read(length))
    return header, 8 + length


def group_for(key):
    if key.startswith("model."):
        return "core"
    if key.startswith(("vision.", "linear_proj.")):
        return "vision"
    return "conditioning"


def write_group(source_path, target_path, group, compare_path=None):
    source_header, source_base = read_header(source_path)
    compare_header, compare_base = read_header(compare_path) if compare_path else ({}, 0)
    keys = [key for key in source_header if key != "__metadata__" and group_for(key) == group]
    keys.sort(key=lambda key: source_header[key]["data_offsets"][0])
    if not keys:
        raise ValueError(f"no {group} tensors in {source_path}")
    if compare_path and set(keys) != {
        key for key in compare_header if key != "__metadata__" and group_for(key) == group
    }:
        raise ValueError(f"{group} tensor keys differ between {source_path} and {compare_path}")

    metadata = dict(source_header.get("__metadata__", {}))
    metadata["ming_component"] = group
    header = {"__metadata__": metadata}
    offset = 0
    for key in keys:
        entry = source_header[key]
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        if compare_path:
            other = compare_header[key]
            if (entry["dtype"], entry["shape"], size) != (
                other["dtype"], other["shape"],
                other["data_offsets"][1] - other["data_offsets"][0],
            ):
                raise ValueError(f"shared tensor metadata differs: {key}")
        header[key] = {**entry, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        existing_header, base = read_header(target_path)
        if existing_header != header or target_path.stat().st_size != base + offset:
            raise FileExistsError(f"existing split checkpoint differs: {target_path}")
        print(f"Already prepared: {target_path}", flush=True)
        return
    partial = target_path.with_name(target_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with source_path.open("rb") as source, partial.open("wb") as target:
            target.write(len(encoded).to_bytes(8, "little"))
            target.write(encoded)
            if compare_path:
                comparison = compare_path.open("rb")
            else:
                comparison = None
            try:
                for key in keys:
                    start, end = source_header[key]["data_offsets"]
                    source.seek(source_base + start)
                    if comparison:
                        other_start = compare_header[key]["data_offsets"][0]
                        comparison.seek(compare_base + other_start)
                    remaining = end - start
                    while remaining:
                        chunk = source.read(min(CHUNK, remaining))
                        if not chunk or (comparison and chunk != comparison.read(len(chunk))):
                            raise ValueError(f"shared tensor bytes differ or source is truncated: {key}")
                        target.write(chunk)
                        remaining -= len(chunk)
            finally:
                if comparison:
                    comparison.close()
        if partial.stat().st_size != 8 + len(encoded) + offset:
            raise ValueError(f"incomplete split checkpoint: {partial}")
        os.replace(partial, target_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"Prepared: {target_path} ({offset:,} tensor bytes, {len(keys)} tensors)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    args = parser.parse_args()
    root = args.checkpoint_root.resolve()
    for variant in ("bf16", "int8_convrot"):
        design = root / DESIGN / f"{DESIGN}_{variant}.safetensors"
        layer = root / LAYER / f"{LAYER}_{variant}.safetensors"
        for path in (design, layer):
            if not path.is_file():
                raise FileNotFoundError(path)
        write_group(design, root / DESIGN / f"{CORE}_{variant}.safetensors", "core", layer)
        write_group(design, root / "ming_image_shared" / f"vision_encoder_{variant}.safetensors", "vision", layer)
        write_group(design, root / "ming_image" / f"conditioning_{variant}.safetensors", "conditioning")
        write_group(layer, root / "ming_image_layer" / f"conditioning_{variant}.safetensors", "conditioning")


if __name__ == "__main__":
    main()
