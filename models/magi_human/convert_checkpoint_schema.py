from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmgp import safetensors2

from models.magi_human.checkpoint_schema import convert_transformer_state_dict_to_split_experts, preprocess_magi_lora_state_dict


def _load_state_dict_and_metadata(path: str):
    with safetensors2.safe_open(path, framework="pt", device="cpu", writable_tensors=False) as reader:
        metadata = reader.metadata() or {}
        state_dict = OrderedDict((key, reader.get_tensor(key)) for key in reader.keys())
    return state_dict, metadata


def _load_sharded_state_dict_and_metadata(index_path: str):
    with open(index_path, "r", encoding="utf-8") as reader:
        index_data = json.load(reader)
    shard_to_keys = OrderedDict()
    for key, shard_name in index_data["weight_map"].items():
        shard_to_keys.setdefault(shard_name, []).append(key)
    state_dict = OrderedDict()
    metadata = dict(index_data.get("metadata") or {})
    base_dir = Path(index_path).resolve().parent
    for shard_name, keys in shard_to_keys.items():
        shard_path = base_dir / shard_name
        with safetensors2.safe_open(str(shard_path), framework="pt", device="cpu", writable_tensors=False) as reader:
            if not metadata:
                metadata.update(reader.metadata() or {})
            for key in keys:
                state_dict[key] = reader.get_tensor(key)
    return state_dict, metadata


def _resolve_input_path(path: str):
    input_path = Path(path)
    if input_path.is_dir():
        index_path = input_path / "model.safetensors.index.json"
        if index_path.exists():
            return str(index_path)
        safetensors_path = input_path / "model.safetensors"
        if safetensors_path.exists():
            return str(safetensors_path)
    return str(input_path)


def _cast_state_dict_dtype(state_dict: Mapping[str, object], dtype: torch.dtype | None):
    if dtype is None:
        return OrderedDict(state_dict.items())
    casted = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
            casted[key] = value.to(dtype)
        else:
            casted[key] = value
    return casted


def _load_any_state_dict_and_metadata(path: str, dtype: torch.dtype | None = None):
    resolved_path = _resolve_input_path(path)
    if resolved_path.endswith(".index.json"):
        state_dict, metadata = _load_sharded_state_dict_and_metadata(resolved_path)
    else:
        state_dict, metadata = _load_state_dict_and_metadata(resolved_path)
    return _cast_state_dict_dtype(state_dict, dtype), metadata


def convert_transformer_checkpoint(input_path: str, output_path: str, dtype: torch.dtype | None = torch.bfloat16) -> None:
    state_dict, metadata = _load_any_state_dict_and_metadata(input_path, dtype=dtype)
    config = metadata.get("config")
    converted = convert_transformer_state_dict_to_split_experts(state_dict)
    safetensors2.torch_write_file(converted, output_path, config=config)


def convert_lora_checkpoint(input_path: str, output_path: str, dtype: torch.dtype | None = None) -> None:
    state_dict, metadata = _load_any_state_dict_and_metadata(input_path, dtype=dtype)
    converted = preprocess_magi_lora_state_dict(state_dict)
    extra_meta = {k: v for k, v in metadata.items() if k not in {"config", "format"}}
    safetensors2.torch_write_file(converted, output_path, config=metadata.get("config"), extra_meta=extra_meta or None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Magi Human checkpoints or LoRAs to the split-expert schema.")
    parser.add_argument("input", type=str, help="Source safetensors path")
    parser.add_argument("output", type=str, help="Converted safetensors path")
    parser.add_argument("--kind", choices=("transformer", "lora"), default="transformer")
    parser.add_argument("--dtype", choices=("keep", "bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dtype = None if args.dtype == "keep" else (torch.bfloat16 if args.dtype == "bf16" else torch.float16)
    if args.kind == "transformer":
        convert_transformer_checkpoint(args.input, args.output, dtype=dtype)
    else:
        convert_lora_checkpoint(args.input, args.output, dtype=dtype if args.dtype != "keep" else None)


if __name__ == "__main__":
    main()
