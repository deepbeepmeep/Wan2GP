"""Repack YuE2 into a BF16 AR text encoder and acoustic transformer."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(root):
    root = Path(root).resolve()
    assets = root / "yue2"
    text_encoder = root / "YuE2_AR"
    text_encoder.mkdir(exist_ok=True)
    source = assets / "upstream"
    source_model = source / "transformer/model.safetensors"
    config = json.loads((source / "transformer/config.json").read_text())
    common = {name: config[name] for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta", "max_position_embeddings")}
    ar_config = dict(common, model_type="qwen3", architectures=["YuE2AR"], hidden_act="silu", attention_bias=False, tie_word_embeddings=False, torch_dtype="bfloat16")
    nar_config = dict(common, model_type="yue2", architectures=["YuE2Acoustic"], latent_dim=64, max_latent_frames=config["max_latent_frames"], timestep_shift=config["timestep_shift"], ffn_chunk_size=1024, torch_dtype="bfloat16")
    configs = Path(__file__).parent
    for filename, data in (("yue2_ar.json", ar_config), ("yue2.json", nar_config)):
        (configs / filename).write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    with safe_open(source_model, framework="pt", device="cpu") as reader:
        keys = reader.keys()
        nar_keys = {key for key in keys if ".nar_" in key or key.startswith(("llm2vae.", "vae2llm.", "time_embedder.", "latent_pos_embed."))}
        for output, selected in ((text_encoder / "YuE2_AR_bf16.safetensors", set(keys) - nar_keys), (root / "YuE2_Acoustic_bf16.safetensors", nar_keys | {"model.norm.weight"})):
            if output.exists():
                raise FileExistsError(output)
            state = {key: reader.get_tensor(key).to(torch.bfloat16).contiguous() for key in sorted(selected)}
            save_file(state, output, metadata={"format": "pt", "source": "m-a-p/YuE2-3B@1a96eca688d6ae5d7f0feb88573fec89920fcd19"})
            with safe_open(output, framework="pt", device="cpu") as saved:
                assert set(saved.keys()) == selected
                for key in selected:
                    assert torch.equal(saved.get_tensor(key), state[key]), key
            del state
            print(f"Verified {output}", flush=True)
    source_model.unlink()
    vae_source = source / "vae/model.safetensors"
    with safe_open(vae_source, framework="pt", device="cpu") as reader:
        state = {}
        for key in reader.keys():
            if not key.startswith("decoder.") or key.endswith(".weight_g"):
                continue
            value = reader.get_tensor(key)
            if key.endswith(".weight_v"):
                value = torch._weight_norm(value, reader.get_tensor(key[:-8] + "weight_g"), 0)
                key = key[:-8] + "weight"
            state[key] = value.to(torch.bfloat16).contiguous()
        output = assets / "YuE2_VAE_bf16.safetensors"
        if output.exists():
            raise FileExistsError(output)
        save_file(state, output, metadata={"format": "pt", "source": "m-a-p/YuE2-Vae@95535e72a97bc0f09b8ada125d26b4009428c0e8"})
        with safe_open(output, framework="pt", device="cpu") as saved:
            for key, value in state.items():
                assert torch.equal(saved.get_tensor(key), value), key
    vae_source.unlink()
    (assets / "vae_config.json").write_bytes((source / "vae/config.json").read_bytes())
    (text_encoder / "qwen.tiktoken").write_bytes((source / "transformer/qwen.tiktoken").read_bytes())
    print(f"Verified {output}; original combined/FP32 checkpoints removed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root")
    convert(parser.parse_args().checkpoint_root)
