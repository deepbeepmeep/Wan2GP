"""
SCAIL checkpoint converter: SAT/DeepSpeed format → WanGP diffusers format.

SCAIL checkpoints use:
1. DeepSpeed format (weights in sd['module'])
2. SAT (SwissArmyTransformer) key naming with fused QKV

This converts to WanGP canonical format.
"""

import os
import re
import torch
from typing import Dict, Optional
from safetensors.torch import save_file


def convert_sat_to_wangp(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert SAT-format state dict to WanGP diffusers format.

    Key mappings:
    SAT format → WanGP format
    - model.diffusion_model.mixins.patch_embed.proj → patch_embedding
    - model.diffusion_model.mixins.patch_embed.proj_pose → pose_patch_embedding
    - model.diffusion_model.mixins.adaln_layer.adaLN_modulations.{i} → blocks.{i}.modulation
    - model.diffusion_model.mixins.adaln_layer.query_layernorm_list.{i} → blocks.{i}.self_attn.norm_q
    - model.diffusion_model.mixins.adaln_layer.key_layernorm_list.{i} → blocks.{i}.self_attn.norm_k
    - model.diffusion_model.transformer.layers.{i}.attention.query_key_value → split to q/k/v
    - model.diffusion_model.transformer.layers.{i}.attention.dense → blocks.{i}.self_attn.o
    - model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h → blocks.{i}.ffn.0
    - model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h → blocks.{i}.ffn.2
    - model.diffusion_model.time_embed → time_embedding
    - model.diffusion_model.adaln_projection.1 → time_projection.1
    - model.diffusion_model.text_embedding → text_embedding
    - model.diffusion_model.clip_proj.proj → img_emb.proj
    """
    new_sd = {}

    # Process each key
    for k, v in sd.items():
        # Strip common prefix
        key = k
        if key.startswith("model.diffusion_model."):
            key = key[len("model.diffusion_model."):]

        # Patch embeddings
        if key == "mixins.patch_embed.proj.weight":
            new_sd["patch_embedding.weight"] = v
            continue
        if key == "mixins.patch_embed.proj.bias":
            new_sd["patch_embedding.bias"] = v
            continue
        if key == "mixins.patch_embed.proj_pose.weight":
            new_sd["pose_patch_embedding.weight"] = v
            continue
        if key == "mixins.patch_embed.proj_pose.bias":
            new_sd["pose_patch_embedding.bias"] = v
            continue

        # AdaLN modulations -> block modulation
        m = re.match(r"mixins\.adaln_layer\.adaLN_modulations\.(\d+)", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.modulation"] = v
            continue

        # Query/Key layernorms
        m = re.match(r"mixins\.adaln_layer\.query_layernorm_list\.(\d+)\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.self_attn.norm_q.weight"] = v
            continue
        m = re.match(r"mixins\.adaln_layer\.key_layernorm_list\.(\d+)\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.self_attn.norm_k.weight"] = v
            continue

        # Cross-attention layernorms
        m = re.match(r"mixins\.adaln_layer\.cross_query_layernorm_list\.(\d+)\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.cross_attn.norm_q.weight"] = v
            continue
        m = re.match(r"mixins\.adaln_layer\.cross_key_layernorm_list\.(\d+)\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.cross_attn.norm_k.weight"] = v
            continue

        # CLIP feature projections -> k_img/v_img
        m = re.match(r"mixins\.adaln_layer\.clip_feature_key_layernorm_list\.(\d+)\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.cross_attn.norm_k_img.weight"] = v
            continue
        m = re.match(r"mixins\.adaln_layer\.clip_feature_key_value_list\.(\d+)\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            # Split fused KV for image features
            if suffix == "weight":
                k_img, v_img = v.chunk(2, dim=0)
                new_sd[f"blocks.{block_idx}.cross_attn.k_img.{suffix}"] = k_img
                new_sd[f"blocks.{block_idx}.cross_attn.v_img.{suffix}"] = v_img
            else:
                k_img, v_img = v.chunk(2, dim=0)
                new_sd[f"blocks.{block_idx}.cross_attn.k_img.{suffix}"] = k_img
                new_sd[f"blocks.{block_idx}.cross_attn.v_img.{suffix}"] = v_img
            continue

        # Transformer layers - attention
        m = re.match(r"transformer\.layers\.(\d+)\.attention\.query_key_value\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            # Split fused QKV into separate Q, K, V
            # QKV is typically [3*hidden, hidden] for weight, [3*hidden] for bias
            if suffix == "weight":
                hidden_size = v.shape[1]
                q, k, vv = v.chunk(3, dim=0)
                new_sd[f"blocks.{block_idx}.self_attn.q.{suffix}"] = q
                new_sd[f"blocks.{block_idx}.self_attn.k.{suffix}"] = k
                new_sd[f"blocks.{block_idx}.self_attn.v.{suffix}"] = vv
            else:  # bias
                q, k, vv = v.chunk(3, dim=0)
                new_sd[f"blocks.{block_idx}.self_attn.q.{suffix}"] = q
                new_sd[f"blocks.{block_idx}.self_attn.k.{suffix}"] = k
                new_sd[f"blocks.{block_idx}.self_attn.v.{suffix}"] = vv
            continue

        m = re.match(r"transformer\.layers\.(\d+)\.attention\.dense\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.self_attn.o.{suffix}"] = v
            continue

        # Cross-attention - separate query
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.query\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.q.{suffix}"] = v
            continue

        # Cross-attention - fused key_value (split into k and v)
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.key_value\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            # Split fused KV into separate K and V
            k, vv = v.chunk(2, dim=0)
            new_sd[f"blocks.{block_idx}.cross_attn.k.{suffix}"] = k
            new_sd[f"blocks.{block_idx}.cross_attn.v.{suffix}"] = vv
            continue

        # Cross-attention - separate k/v (fallback)
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.key\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.k.{suffix}"] = v
            continue
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.value\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.v.{suffix}"] = v
            continue

        # Cross-attention output
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.dense\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.o.{suffix}"] = v
            continue

        # Post cross-attention layer norm -> norm3
        m = re.match(r"transformer\.layers\.(\d+)\.post_cross_attention_layernorm\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.norm3.{suffix}"] = v
            continue

        # MLP / FFN
        m = re.match(r"transformer\.layers\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.ffn.0.{suffix}"] = v
            continue
        m = re.match(r"transformer\.layers\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.ffn.2.{suffix}"] = v
            continue

        # Post-attention layer norm (norm3 in WanGP)
        m = re.match(r"transformer\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.norm3.{suffix}"] = v
            continue

        # Time embedding
        m = re.match(r"time_embed\.(\d+)\.(weight|bias)", key)
        if m:
            idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"time_embedding.{idx}.{suffix}"] = v
            continue

        # AdaLN projection (time projection)
        m = re.match(r"adaln_projection\.(\d+)\.(weight|bias)", key)
        if m:
            idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"time_projection.{idx}.{suffix}"] = v
            continue

        # Text embedding
        m = re.match(r"text_embedding\.(\d+)\.(weight|bias)", key)
        if m:
            idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"text_embedding.{idx}.{suffix}"] = v
            continue

        # CLIP projection (img_emb)
        m = re.match(r"clip_proj\.proj\.(\d+)\.(weight|bias)", key)
        if m:
            idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"img_emb.proj.{idx}.{suffix}"] = v
            continue

        # Final projection / head
        if key == "mixins.final_layer.adaLN_modulation":
            new_sd["head.modulation"] = v
            continue
        m = re.match(r"mixins\.final_layer\.linear\.(weight|bias)", key)
        if m:
            suffix = m.group(1)
            new_sd[f"head.head.{suffix}"] = v
            continue

        # Cross-attention image projections (k_img, v_img, norm_k_img)
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.k_img\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.k_img.{suffix}"] = v
            continue
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.v_img\.(weight|bias)", key)
        if m:
            block_idx = m.group(1)
            suffix = m.group(2)
            new_sd[f"blocks.{block_idx}.cross_attn.v_img.{suffix}"] = v
            continue
        m = re.match(r"transformer\.layers\.(\d+)\.cross_attention\.norm_k_img\.weight", key)
        if m:
            block_idx = m.group(1)
            new_sd[f"blocks.{block_idx}.cross_attn.norm_k_img.weight"] = v
            continue

        # If no match, keep with warning
        print(f"[WARN] Unmapped key: {k}")
        # Still include it with stripped prefix
        new_sd[key] = v

    return new_sd


def load_deepspeed_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load DeepSpeed checkpoint and extract model weights."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "module" in ckpt:
        print("DeepSpeed format detected, extracting from 'module' key")
        return ckpt["module"]
    elif isinstance(ckpt, dict):
        # Regular state dict
        return ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")


def convert_scail_checkpoint(
    input_path: str,
    output_path: str,
    cast_dtype: Optional[str] = "bf16"
) -> None:
    """
    Convert SCAIL checkpoint to WanGP format.

    Args:
        input_path: Path to SCAIL .pt checkpoint
        output_path: Path to output .safetensors file
        cast_dtype: Target dtype (bf16, fp16, fp32)
    """
    # Load checkpoint
    sd = load_deepspeed_checkpoint(input_path)

    # Convert keys
    print("Converting SAT keys to WanGP format...")
    new_sd = convert_sat_to_wangp(sd)

    # Cast dtype
    if cast_dtype:
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(cast_dtype.lower())
        if target_dtype:
            print(f"Casting to {target_dtype}")
            new_sd = {k: v.to(target_dtype) for k, v in new_sd.items()}

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving to: {output_path}")
    save_file(new_sd, output_path, metadata={
        "format": "wangp_scail",
        "converted_from": "scail_sat_deepspeed",
    })

    print(f"Done! Converted {len(new_sd)} keys")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_scail.py <input.pt> [output.safetensors]")
        print("Example: python convert_scail.py c:/temp/scail.pt c:/temp/scail_wangp.safetensors")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".pt", "_wangp.safetensors")

    convert_scail_checkpoint(input_path, output_path)
