"""Fold codec weight normalization and prepare a chosen storage precision for validation."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file


def prepare_state(source, dtype):
    raw = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state, seen = {}, set()
    for key, value in raw.items():
        if key in seen or key.endswith(("freqs_cis", "causal_mask")):
            continue  # derived positional tables and masks, computed for the actual input length
        if key.endswith(".parametrizations.weight.original0"):
            prefix = key[:-len(".parametrizations.weight.original0")]
            other = prefix + ".parametrizations.weight.original1"
            value = torch._weight_norm(raw[other], value, 0)
            seen.add(other)
            key = prefix + ".weight"
        elif key.endswith(".weight_g"):
            prefix = key[:-len(".weight_g")]
            other = prefix + ".weight_v"
            value = torch._weight_norm(raw[other], value, 0)
            seen.add(other)
            key = prefix + ".weight"
        elif key.endswith((".weight_v", ".parametrizations.weight.original1")):
            continue
        if value.is_floating_point():
            value = value.to(dtype=dtype)
        state[key] = value.contiguous()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="FP32 reference for precision comparison only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = prepare_state(args.source, torch.float32 if args.reference else torch.bfloat16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, args.output, metadata={"format": "pt", "source_repository": "fishaudio/drama-3", "source_revision": "dd64a85cd82e9521f7202610ede97b367fc83946", "weight_normalization": "folded"})
    from safetensors import safe_open
    with safe_open(args.output, framework="pt", device="cpu") as file:
        assert set(file.keys()) == set(state)
        for key, value in state.items():
            stored = file.get_tensor(key)
            assert stored.dtype == value.dtype and stored.shape == value.shape and torch.equal(stored, value), key
    print(f"Verified {len(state)} tensors: {args.output}", flush=True)


if __name__ == "__main__":
    main()
