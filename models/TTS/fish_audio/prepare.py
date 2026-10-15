"""Repack the released Drama 3 shards without changing any tensor payload.

python -m models.TTS.fish_audio.prepare --source PATH --output PATH
python -m models.TTS.fish_audio.prepare --source BACKBONE_FILE --output PATH --split-dense NAME [--split-experts NAME] [--audio-head]
CPU experts retain released block-FP8 storage; BF16 backbone and codebook weights
are not cast. This bounded-memory conversion retains its sources and verifies
every output tensor's byte digest before marking the preparation complete.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
CODE_REVISION = "1ff3c3ed2f80d3106a83909dd25b83129c3d816f"
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)")
TOWER = {"audio_decoder.codebook_embeddings.": "codebook_embeddings.",
         "audio_decoder_mtp_shared.layers.": "shared_layers.",
         "audio_decoder_mtp.layers.": "residual_layers.",
         "audio_decoder_mtp.heads.": "codebook_heads."}


def catalog(source):
    entries, sources = {}, {}
    for shard in sorted(source.glob("model*.safetensors")):
        with shard.open("rb") as reader:
            length = struct.unpack("<Q", reader.read(8))[0]
            header = json.loads(reader.read(length))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in entries:
                raise ValueError(f"Duplicate source tensor: {key}")
            entries[key] = entry
            begin, end = entry["data_offsets"]
            sources[key] = (shard, 8 + length + begin, end - begin)
    return entries, sources


def plan(entries, config):
    backbone, tower = {}, {}
    text = config["text_config"]
    experts, hidden, intermediate = (text[key] for key in ("num_experts", "hidden_size", "moe_intermediate_size"))
    for key, spec in entries.items():
        match = EXPERT.fullmatch(key)
        if match:
            layer, expert, projection, kind = match.groups()
            group = "w2" if projection == "down_proj" else "w13"
            dtype = "F8_E4M3" if kind == "weight" else "F32"
            if spec["dtype"] != dtype:
                raise ValueError(f"Unexpected expert precision: {key} {spec['dtype']}")
            rows, cols = (hidden, intermediate) if group == "w2" else (2 * intermediate, hidden)
            if kind != "weight":
                rows, cols = rows // 128, cols // 128
            suffix = "weight" if kind == "weight" else "scale_inv"
            name = f"model.layers.{layer}.mlp.experts.{group}_{suffix}"
            entry = backbone.setdefault(name, {"dtype": dtype, "shape": [experts, rows, cols], "pieces": []})
            entry["pieces"].append(((int(expert), projection == "up_proj"), key))
        else:
            if spec["dtype"] != "BF16":
                raise ValueError(f"Unexpected non-expert precision: {key} {spec['dtype']}")
            mapping = next(((old, new) for old, new in TOWER.items() if key.startswith(old)), None)
            if mapping:
                old, new = mapping
                target, name = tower, new + key[len(old):]
            else:
                target = backbone
                name = "model." + key[len("model.language_model."):] if key.startswith("model.language_model.") else key
            if name in target:
                raise ValueError(f"Duplicate target tensor: {name}")
            target[name] = {"dtype": spec["dtype"], "shape": spec["shape"], "pieces": [((0, 0), key)]}
    for output in (backbone, tower):
        for entry in output.values():
            entry["pieces"].sort()
    return backbone, tower


def catalog_file(path):
    """Header, byte sources and metadata of one prepared safetensors file."""
    with path.open("rb") as reader:
        length = struct.unpack("<Q", reader.read(8))[0]
        header = json.loads(reader.read(length))
    metadata = header.pop("__metadata__", {})
    return header, {key: (path, 8 + length + entry["data_offsets"][0], entry["data_offsets"][1] - entry["data_offsets"][0]) for key, entry in header.items()}, metadata


def split_plan(entries, sources, config, audio_head=False):
    """Dense part and routed experts of a prepared backbone. With audio_head (for quantized dense parts), the LM head keeps only the
    rows decoding samples from (eos through the semantic band, per-row ConvRot scales included); a BF16 dense part keeps them all,
    so that the BF16 parts and the upstream FP8 experts keep every released payload."""
    start = min(config["eos_token_id"], config["semantic_start_token_id"])
    stop = max(config["eos_token_id"], config["semantic_end_token_id"]) + 1
    dense, experts = {}, {}
    for key, spec in entries.items():
        target = experts if ".mlp.experts." in key else dense
        if audio_head and key in ("lm_head.weight", "lm_head.weight_scale"):
            path, begin, size = sources[key]
            row = size // spec["shape"][0]
            sources[key + "#audio"] = (path, begin + start * row, (stop - start) * row)
            target[key] = {"dtype": spec["dtype"], "shape": [stop - start, *spec["shape"][1:]], "pieces": [((0, 0), key + "#audio")]}
        else:
            target[key] = {"dtype": spec["dtype"], "shape": spec["shape"], "pieces": [((0, 0), key)]}
    return dense, experts


def header_bytes(plan, sources, metadata=None):
    """Safetensors header (JSON padded to 8 bytes) of plan, tensors in name order, the header dictionary and the payload size."""
    header = {"__metadata__": metadata or {"format": "pt", "source_repository": "fishaudio/drama-3", "source_revision": REVISION,
                                           "expert_storage": "block_fp8_e4m3fn_128", "preparation": "payload-preserving expert stacking and stage split"}}
    offset = 0
    for name, spec in sorted(plan.items()):
        size = sum(sources[key][2] for _, key in spec["pieces"])
        elements = 1
        for dim in spec["shape"]:
            elements *= dim
        assert size == elements * {"F8_E4M3": 1, "BF16": 2, "F32": 4, "I8": 1, "U8": 1}[spec["dtype"]], name
        header[name] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return encoded + b" " * (-len(encoded) % 8), header, offset


def write_verified(target, plan, sources, metadata=None):
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing prepared checkpoint: {target}")
    encoded, header, offset = header_bytes(plan, sources, metadata)
    partial = target.with_suffix(".partial")
    handles = {path: path.open("rb") for path in {entry[0] for entry in sources.values()}}
    digests = {}
    try:
        with partial.open("xb") as writer:
            writer.write(struct.pack("<Q", len(encoded)))
            writer.write(encoded)
            for index, (name, spec) in enumerate(sorted(plan.items())):
                digest = hashlib.sha256()
                for _, key in spec["pieces"]:
                    path, begin, remaining = sources[key]
                    reader = handles[path]
                    reader.seek(begin)
                    while remaining:
                        chunk = reader.read(min(remaining, 16 << 20))
                        if not chunk:
                            raise IOError(f"Truncated source {path}: {key}")
                        writer.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                digests[name] = digest.hexdigest()
                if index % 40 == 0:
                    print(f"Preparing {target.name}: {index + 1}/{len(plan)} Tensors", flush=True)
        with partial.open("rb") as reader:
            reader.seek(8 + len(encoded))
            for name in sorted(plan):
                begin, end = header[name]["data_offsets"]
                digest, remaining = hashlib.sha256(), end - begin
                while remaining:
                    chunk = reader.read(min(remaining, 16 << 20))
                    if not chunk:
                        raise IOError(f"Truncated prepared tensor {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != digests[name]:
                    raise IOError(f"Prepared tensor failed byte verification: {name}")
        partial.rename(target)
    finally:
        for handle in handles.values():
            handle.close()
    return {"file": str(target), "payload_bytes": offset, "tensor_count": len(plan), "tensor_sha256": digests}


def split(source, output, dense_name, experts_name=None, audio_head=False):
    """Split a prepared backbone file into its dense part and, optionally, its routed experts, payloads verified byte for byte."""
    config = json.loads(Path(__file__).with_name("fish_drama3.json").read_text())
    entries, sources, metadata = catalog_file(source)
    dense, experts = split_plan(entries, sources, config, audio_head)
    metadata = dict(metadata, preparation=metadata.get("preparation", "") + "; dense/routed-expert split" + (", LM head rows eos..semantic_end" if audio_head else ""))
    report = {"source": str(source), "outputs": [write_verified(output / dense_name, dense, sources, metadata)]}
    if experts_name:
        report["outputs"].append(write_verified(output / experts_name, experts, sources, metadata))
    (output / "fish_audio").mkdir(exist_ok=True)
    (output / "fish_audio" / f"split_{Path(dense_name).stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Split Tensor Payloads Verified; Source Retained", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-dense", help="split the prepared backbone file --source into this dense file name")
    parser.add_argument("--split-experts", help="with --split-dense: also write the routed experts under this file name")
    parser.add_argument("--audio-head", action="store_true", help="with --split-dense: keep only the LM-head rows decoding uses (quantized parts)")
    args = parser.parse_args()
    if args.split_dense:
        return split(args.source, args.output, args.split_dense, args.split_experts, args.audio_head)
    config = json.loads((args.source / "config.json").read_text())
    entries, sources = catalog(args.source)
    plans = plan(entries, config)
    required = sum(entry[2] for entry in sources.values())
    import shutil
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < required + (1 << 30):
        raise RuntimeError("Not enough free disk space to preserve source and prepared checkpoints")
    report = {"source_revision": REVISION, "code_revision": CODE_REVISION, "source_tensors": len(entries),
              "source_dtype_counts": dict(Counter(spec["dtype"] for spec in entries.values())), "outputs": []}
    for name, output_plan in zip(("Fish_Drama3_backbone_fp8.safetensors", "Fish_Drama3_codebooks_bf16.safetensors"), plans):
        report["outputs"].append(write_verified(args.output / name, output_plan, sources))
    report_dir = args.output / "fish_audio"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("All Transformer Tensor Payloads Verified; Sources Retained", flush=True)


if __name__ == "__main__":
    main()
