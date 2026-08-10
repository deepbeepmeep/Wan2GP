"""Prune MiniMax H3's full AdaLN time projections into a shared curve basis."""

import argparse
import json
import math
import os
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm


DEFAULT_SOURCE = Path(r"MiniMax-H3-FL2VA_bf16.safetensors")
DEFAULT_OUTPUT = Path(r"MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors")
TIME_SUFFIX = "time_embedder.proj_in.weight"
TABLE_SUFFIX = "adaln_t_table"
ADALN_WEIGHT_SUFFIX = ".adaln_proj.linear.weight"
ADALN_BIAS_SUFFIX = ".adaln_proj.linear.bias"
DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _read_header(path):
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        return header_size, json.loads(stream.read(header_size))


def _find_prefix(names):
    matches = [name[:-len(TIME_SUFFIX)] for name in names if name.endswith(TIME_SUFFIX)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {TIME_SUFFIX}, found {len(matches)}")
    return matches[0]


def _time_curve(source, prefix, grid, device):
    w1 = source.get_tensor(prefix + "time_embedder.proj_in.weight").to(device)
    b1 = source.get_tensor(prefix + "time_embedder.proj_in.bias").to(device)
    w2 = source.get_tensor(prefix + "time_embedder.proj_out.weight").to(device)
    b2 = source.get_tensor(prefix + "time_embedder.proj_out.bias").to(device)
    t = torch.linspace(0.0, 1.0, grid, dtype=torch.float32, device=device)
    half = w1.shape[1] // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half)
    embedding = torch.cat((torch.cos(t[:, None] * frequencies), torch.sin(t[:, None] * frequencies)), dim=-1)
    curve = F.silu(F.linear(F.silu(F.linear(embedding, w1, b1)), w2, b2))
    return curve, (w1, b1, w2, b2)


def _factor_curve(curve, rank):
    curve64 = curve.double()
    mean64 = curve64.mean(dim=0)
    u, singular, vh = torch.linalg.svd(curve64 - mean64, full_matrices=False)
    table = (u[:, :rank] * singular[:rank]).float()
    basis = vh[:rank].float()
    mean = mean64.float()
    reconstructed = table @ basis + mean
    error = reconstructed - curve
    metrics = {
        "curve_rmse": error.square().mean().sqrt().item(),
        "curve_max_abs": error.abs().max().item(),
        "retained_energy": (singular[:rank].square().sum() / singular.square().sum()).item(),
    }
    return table, basis, mean, metrics


def _tensor_size(entry):
    try:
        item_size = DTYPE_SIZES[entry["dtype"]]
    except KeyError as error:
        raise ValueError(f"Unsupported safetensors dtype {entry['dtype']}") from error
    return math.prod(entry["shape"]) * item_size


def _output_entries(source_catalog, prefix, grid, rank):
    time_prefix = prefix + "time_embedder."
    table_name = prefix + TABLE_SUFFIX
    if table_name in source_catalog:
        raise ValueError("Source checkpoint is already modulation-pruned")
    names = []
    for name in source_catalog:
        if name.startswith(time_prefix):
            continue
        if name.endswith(ADALN_WEIGHT_SUFFIX):
            continue
        names.append(name)
        if name.endswith(ADALN_BIAS_SUFFIX):
            weight_name = name[:-len("bias")] + "weight"
            if weight_name not in source_catalog:
                raise KeyError(weight_name)
            names.append(weight_name)
    names.append(table_name)

    entries = {}
    offset = 0
    for name in names:
        if name == table_name:
            entry = {"dtype": "F32", "shape": [grid, rank]}
        elif name.endswith(ADALN_WEIGHT_SUFFIX):
            source_entry = source_catalog[name]
            entry = {"dtype": "F32", "shape": [source_entry["shape"][0], rank]}
        elif name.endswith(ADALN_BIAS_SUFFIX):
            entry = {"dtype": "F32", "shape": source_catalog[name]["shape"]}
        else:
            source_entry = source_catalog[name]
            entry = {"dtype": source_entry["dtype"], "shape": source_entry["shape"]}
        size = _tensor_size(entry)
        entries[name] = {**entry, "data_offsets": [offset, offset + size]}
        offset += size
    return names, entries, offset


def _write_tensor(stream, tensor):
    array = tensor.contiguous().numpy()
    data = memoryview(array).cast("B")
    if stream.write(data) != len(data):
        raise OSError("Incomplete checkpoint write")


def _copy_tensor(source_stream, target_stream, source_data_start, entry, progress, chunk_size=16 * 1024 * 1024):
    start, end = entry["data_offsets"]
    source_stream.seek(source_data_start + start)
    remaining = end - start
    while remaining:
        data = source_stream.read(min(remaining, chunk_size))
        if not data:
            raise OSError("Unexpected end of source checkpoint")
        target_stream.write(data)
        progress.update(len(data))
        remaining -= len(data)


def _compress_projection(source, bias_name, basis, curve_mean, device, row_chunk):
    weight_name = bias_name[:-len("bias")] + "weight"
    source_weight = source.get_tensor(weight_name)
    source_bias = source.get_tensor(bias_name)
    rows = source_weight.shape[0]
    compressed_weight = torch.empty(rows, basis.shape[0], dtype=torch.float32)
    compressed_bias = torch.empty(rows, dtype=torch.float32)
    basis_t = basis.T.contiguous()
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        weight = source_weight[start:stop].to(device=device, dtype=torch.float32)
        compressed_weight[start:stop].copy_((weight @ basis_t).cpu())
        bias = source_bias[start:stop].to(device=device, dtype=torch.float32)
        compressed_bias[start:stop].copy_((bias + weight @ curve_mean).cpu())
    return compressed_bias, compressed_weight


def compress_checkpoint(source_path, output_path, grid, rank, device, row_chunk):
    source_header_size, catalog = _read_header(source_path)
    metadata = catalog.pop("__metadata__", {})
    prefix = _find_prefix(catalog)
    names, output_entries, output_size = _output_entries(catalog, prefix, grid, rank)
    output_catalog = {
        "__metadata__": {
            **metadata,
            "format": "pt",
            "adaln_curve_grid": str(grid),
            "adaln_curve_rank": str(rank),
            "adaln_curve_centered": "true",
        },
        **output_entries,
    }
    header = json.dumps(output_catalog, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    source_data_start = 8 + source_header_size
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")

    with safe_open(source_path, framework="pt", device="cpu") as source:
        curve, _ = _time_curve(source, prefix, grid, device)
        table, basis, curve_mean, metrics = _factor_curve(curve, rank)
        basis = basis.to(device)
        curve_mean = curve_mean.to(device)
        pending_weights = {}
        with open(source_path, "rb") as source_stream, open(temporary_path, "wb") as target:
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            with tqdm(total=output_size, unit="B", unit_scale=True, desc="Writing pruned H3") as progress:
                for name in names:
                    if name == prefix + TABLE_SUFFIX:
                        _write_tensor(target, table.cpu())
                        progress.update(table.nbytes)
                    elif name.endswith(ADALN_BIAS_SUFFIX):
                        bias, weight = _compress_projection(source, name, basis, curve_mean, device, row_chunk)
                        _write_tensor(target, bias)
                        progress.update(bias.nbytes)
                        pending_weights[name[:-len("bias")] + "weight"] = weight
                    elif name.endswith(ADALN_WEIGHT_SUFFIX):
                        weight = pending_weights.pop(name)
                        _write_tensor(target, weight)
                        progress.update(weight.nbytes)
                    else:
                        _copy_tensor(source_stream, target, source_data_start, catalog[name], progress)
        if pending_weights:
            raise RuntimeError(f"Unwritten pruned tensors: {sorted(pending_weights)}")
    os.replace(temporary_path, output_path)
    return metrics


def _interpolated_table(table, timesteps):
    position = timesteps.clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))


def verify_checkpoint(source_path, output_path, device, row_chunk, test_timesteps=17):
    with safe_open(source_path, framework="pt", device="cpu") as source, safe_open(output_path, framework="pt", device="cpu") as compressed:
        source_prefix = _find_prefix(source.keys())
        output_prefix = source_prefix
        if any(name.startswith(source_prefix + "time_embedder.") for name in compressed.keys()):
            raise RuntimeError("Pruned checkpoint still contains the time embedder")
        table = compressed.get_tensor(output_prefix + TABLE_SUFFIX)
        rank = table.shape[1]
        projection_weights = [name for name in compressed.keys() if name.endswith(ADALN_WEIGHT_SUFFIX)]
        if len(projection_weights) != 51:
            raise RuntimeError(f"Expected 51 pruned AdaLN projections, found {len(projection_weights)}")
        if any(compressed.get_slice(name).get_shape()[1] != rank for name in projection_weights):
            raise RuntimeError("A pruned AdaLN projection has the wrong input width")
        if any(compressed.get_tensor(name).dtype != torch.float32 for name in projection_weights):
            raise RuntimeError("Pruned AdaLN projection weights are not FP32")

        timesteps = torch.linspace(0.0, 1.0, test_timesteps, device=device)
        exact_curve, _ = _time_curve(source, source_prefix, test_timesteps, device)
        coordinates = _interpolated_table(table.to(device), timesteps)
        squared_error = squared_reference = 0.0
        value_count = 0
        max_error = 0.0
        for weight_name in tqdm(projection_weights, desc="Verifying AdaLN projections"):
            bias_name = weight_name[:-len("weight")] + "bias"
            source_weight = source.get_tensor(weight_name)
            source_bias = source.get_tensor(bias_name)
            compressed_weight = compressed.get_tensor(weight_name)
            compressed_bias = compressed.get_tensor(bias_name)
            for start in range(0, source_weight.shape[0], row_chunk):
                stop = min(start + row_chunk, source_weight.shape[0])
                weight = source_weight[start:stop].to(device=device, dtype=torch.float32)
                bias = source_bias[start:stop].to(device=device, dtype=torch.float32)
                expected = F.linear(exact_curve, weight, bias)
                actual = F.linear(
                    coordinates,
                    compressed_weight[start:stop].to(device),
                    compressed_bias[start:stop].to(device),
                )
                error = actual - expected
                squared_error += error.double().square().sum().item()
                squared_reference += expected.double().square().sum().item()
                value_count += error.numel()
                max_error = max(max_error, error.abs().max().item())
        rmse = math.sqrt(squared_error / value_count)
        return {
            "tensor_count": len(compressed.keys()),
            "projection_count": len(projection_weights),
            "grid": table.shape[0],
            "rank": rank,
            "modulation_rmse": rmse,
            "modulation_relative_rmse": math.sqrt(squared_error / squared_reference),
            "modulation_max_abs": max_error,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid", type=int, default=1025)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.verify_only:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        print(json.dumps(verify_checkpoint(args.source, args.output, args.device, args.row_chunk), indent=2))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    factor_metrics = compress_checkpoint(args.source, args.output, args.grid, args.rank, args.device, args.row_chunk)
    verification = verify_checkpoint(args.source, args.output, args.device, args.row_chunk)
    print(json.dumps({"curve_factorization": factor_metrics, "checkpoint_verification": verification}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
