"""Split the upstream LTX-2.5 MSR checkpoint without changing tensor values.

python -m models.ltx2.prepare_msr SOURCE --lora OUTPUT --slots OUTPUT
Source: LiconStudio/LTX-2.5-Multiple-Subject-Reference (Apache-2.0),
revision 9a053e6d63e54cd6970b1cda9419f1339457bd8c.
"""

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def split_checkpoint(source, lora_path, slots_path):
    source, lora_path, slots_path = map(Path, (source, lora_path, slots_path))
    if len({path.resolve() for path in (source, lora_path, slots_path)}) != 3:
        raise ValueError("Source and output paths must be distinct.")
    prefix = "diffusion_model.reference_slot_embedding."
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        metadata["source_repo"] = "LiconStudio/LTX-2.5-Multiple-Subject-Reference"
        metadata["source_revision"] = "9a053e6d63e54cd6970b1cda9419f1339457bd8c"
        with source.open("rb") as stream:
            metadata["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        if metadata["source_sha256"] != "d45cedd720e5819ccbe200a4b0ae01f7ae0e0d2966d700a24ebd4c7739515d4f":
            raise ValueError("Source does not match the documented upstream checkpoint revision.")
        groups = ({}, {})
        for key in checkpoint.keys():
            is_slot = key.startswith(prefix)
            if not is_slot and ".lora_" not in key:
                raise ValueError(f"Unexpected non-LoRA tensor: {key}")
            groups[int(is_slot)][key[len(prefix):] if is_slot else key] = checkpoint.get_tensor(key)
        if len(groups[1]) != 5 or not groups[0]:
            raise ValueError("Expected LoRA weights and five slot embedding tensors.")
        for target, tensors in zip((lora_path, slots_path), groups):
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(target), metadata=metadata)
            with safe_open(target, framework="pt", device="cpu") as result:
                assert set(result.keys()) == set(tensors)
                for key, original in tensors.items():
                    actual = result.get_tensor(key)
                    assert actual.dtype == original.dtype and torch.equal(actual, original), key
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            print(f"Verified {target}: {len(tensors)} tensors, SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--slots", required=True)
    args = parser.parse_args()
    split_checkpoint(args.source, args.lora, args.slots)
