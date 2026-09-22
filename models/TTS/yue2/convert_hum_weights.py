"""Split the combined Mothersuperior hum adapter and preserve native precision.

Run with the checkpoint root; original downloads are kept until runtime validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def convert(root):
    root = Path(root)
    sources = root / "yue2_hum_source"
    revisions = {}
    for repo, folder, names in (
        ("Mothersuperior/YuE2-hum-to-song", "adapter", ["hum_adapter_v1_combined.safetensors", "README.md"]),
        ("m-a-p/YuE2-Vae", "vae", ["model.safetensors", "config.json", "modeling_vae.py", "LICENSE", "THIRD_PARTY_NOTICES.md"]),
    ):
        revision = HfApi().model_info(repo).sha
        revisions[repo] = revision
        for name in names:
            hf_hub_download(repo, name, revision=revision, local_dir=sources / folder)
    adapter = load_file(str(sources / "adapter/hum_adapter_v1_combined.safetensors"), device="cpu")
    with safe_open(root / "YuE2_Acoustic_bf16.safetensors", framework="pt", device="cpu") as base:
        lora, module = {}, {}
        for key, value in adapter.items():
            if key.startswith("hum_proj."):
                module[key] = value
            elif ".lora_" in key:
                lora["model." + key + ".weight"] = value
            else:
                layer, kind = key.split(".")
                assert layer in ("vae2llm", "llm2vae") and kind in ("weight", "bias"), key
                original = base.get_tensor(key).to(value.dtype)
                delta = value - original
                torch.testing.assert_close(original + delta, value, atol=1e-6, rtol=1e-6)
                lora[layer + (".diff" if kind == "weight" else ".diff_b")] = delta
    encoder = {}
    with safe_open(sources / "vae/model.safetensors", framework="pt", device="cpu") as source:
        for key in source.keys():
            if not key.startswith("encoder.") or key.endswith(".weight_g"):
                continue
            value = source.get_tensor(key)
            if key.endswith(".weight_v"):
                value = torch._weight_norm(value, source.get_tensor(key[:-8] + "weight_g"), 0)
                key = key[:-8] + "weight"
            encoder[key] = value.contiguous()
    outputs = {"YuE2_Hum_Projections.safetensors": module, "YuE2_Hum_LoRA.safetensors": lora, "YuE2_VAE_Encoder.safetensors": encoder}
    for name, tensors in outputs.items():
        path = root / name
        save_file(tensors, path, metadata={"format": "pt", "sources": json.dumps(revisions), "license": "CC-BY-NC-4.0", "lora_scale": "1.0", "inject_layers": "[0,7,14,21]"})
        verified = load_file(str(path), device="cpu")
        assert verified.keys() == tensors.keys()
        for key, value in tensors.items():
            assert value.dtype == verified[key].dtype and torch.equal(value, verified[key]), key
        details = {"sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(), "tensors": len(tensors), "dtypes": sorted({str(t.dtype) for t in tensors.values()})}
        print("Verified", name, details, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root")
    convert(parser.parse_args().checkpoint_root)
