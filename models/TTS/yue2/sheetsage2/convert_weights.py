"""Merge the pinned MERT2 backbone and SheetSage2 adapters into BF16 storage."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(root):
    root = Path(root)
    source = root / "sheetsage2" / "_source"
    output = root / "sheetsage2" / "SheetSage2_MERT2_bf16.safetensors"
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    adapter_path = source / "SheetSage2" / "model.safetensors"
    backbone_path = source / "MERT-v2-FullSong" / "model.safetensors"
    with backbone_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == config["base_model_sha256"]
    with safe_open(adapter_path, framework="pt") as adapters, safe_open(backbone_path, framework="pt") as backbone:
        state = {key: adapters.get_tensor(key).to(torch.bfloat16) for key in adapters.keys() if not key.startswith("adapter.")}
        for key in backbone.keys():
            value = backbone.get_tensor(key)
            adapter = "adapter." + key.removesuffix(".weight")
            if adapter + ".lora_A.weight" in adapters.keys():
                value = value + (adapters.get_tensor(adapter + ".lora_B.weight") @ adapters.get_tensor(adapter + ".lora_A.weight")) * (config["lora_alpha"] / config["lora_rank"])
            state["encoder." + key] = value if key.startswith("feature_extractor.") else value.to(torch.bfloat16)
        revisions = {name: (source / name / "revision.txt").read_text().strip() for name in ("SheetSage2", "MERT-v2-FullSong")}
        ties = {"token_embedding.weight": ["decoder.embed_tokens.weight", "output_projection.weight"]}
        save_file(state, output, metadata={"format": "pt", "sources": json.dumps(revisions), "tied_weights_map_base64": base64.b64encode(json.dumps(ties).encode()).decode()})
        with safe_open(output, framework="pt") as saved:
            assert set(saved.keys()) == set(state)
            for key, value in state.items():
                assert torch.equal(saved.get_tensor(key), value), key
    manifest = {"sources": revisions, "tensor_count": len(state), "bytes": output.stat().st_size, "fp32_buffers": [key for key, value in state.items() if value.dtype == torch.float32]}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    del state
    print(f"Verified {output} ({output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root")
    convert(parser.parse_args().checkpoint_root)
