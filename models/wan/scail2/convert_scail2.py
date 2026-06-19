"""Convert SCAIL-2 SAT checkpoints to WanGP safetensors.

The output ``mbf16`` policy keeps the layers WanGP locks to float32 in float32
and casts the rest to bfloat16.
"""

import argparse
import os
from typing import Dict

import torch
from mmgp import safetensors2


class ModuleParser:
    def __init__(self, key: str):
        self.key = key
        self.modules = key.split(".")
        self.idx = 0

    def match(self, pattern: str) -> bool:
        parts = pattern.split(".")
        for offset, part in enumerate(parts):
            if self.idx + offset >= len(self.modules) or self.modules[self.idx + offset] != part:
                return False
        self.idx += len(parts)
        return True

    def step(self) -> str:
        module = self.modules[self.idx]
        self.idx += 1
        return module

    def eof(self) -> bool:
        return self.idx == len(self.modules)


def get_new_mappings(key: str, param: torch.Tensor) -> Dict[str, torch.Tensor]:
    modules = []
    parser = ModuleParser(key)
    sat_prefix = "model.diffusion_model"
    if not parser.match(sat_prefix):
        raise ValueError(key)

    if parser.match("mixins"):
        if parser.match("adaln_layer"):
            if parser.match("adaLN_modulations"):
                modules += ["blocks", parser.step(), "modulation"]
            elif parser.match("query_layernorm_list"):
                modules += ["blocks", parser.step(), "self_attn.norm_q", parser.step()]
            elif parser.match("key_layernorm_list"):
                modules += ["blocks", parser.step(), "self_attn.norm_k", parser.step()]
            elif parser.match("cross_query_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_q", parser.step()]
            elif parser.match("cross_key_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_k", parser.step()]
            elif parser.match("clip_feature_key_layernorm_list"):
                modules += ["blocks", parser.step(), "cross_attn.norm_k_img", parser.step()]
            elif parser.match("clip_feature_key_value_list"):
                modules += ["blocks", parser.step(), "cross_attn"]
                prefix = ".".join(modules)
                suffix = parser.step()
                key_param, value_param = param.chunk(2, dim=0)
                return {f"{prefix}.k_img.{suffix}": key_param, f"{prefix}.v_img.{suffix}": value_param}
            else:
                raise ValueError(key)
        elif parser.match("final_layer"):
            modules.append("head")
            if parser.match("adaLN_modulation"):
                modules.append("modulation")
            elif parser.match("linear"):
                modules += ["head", parser.step()]
            else:
                raise ValueError(key)
        elif parser.match("patch_embed"):
            if parser.match("proj"):
                modules.append("patch_embedding")
            elif parser.match("proj_pose"):
                modules.append("pose_patch_embedding")
            elif parser.match("proj_mask"):
                modules.append("mask_patch_embedding")
            else:
                raise ValueError(key)
            modules.append(parser.step())
        else:
            raise ValueError(key)
    elif parser.match("transformer.layers"):
        modules += ["blocks", parser.step()]
        if parser.match("attention"):
            modules.append("self_attn")
            if parser.match("dense"):
                modules += ["o", parser.step()]
            elif parser.match("query_key_value"):
                prefix = ".".join(modules)
                suffix = parser.step()
                query_param, key_param, value_param = param.chunk(3, dim=0)
                return {f"{prefix}.q.{suffix}": query_param, f"{prefix}.k.{suffix}": key_param, f"{prefix}.v.{suffix}": value_param}
            else:
                raise ValueError(key)
        elif parser.match("cross_attention"):
            modules.append("cross_attn")
            if parser.match("dense"):
                modules += ["o", parser.step()]
            elif parser.match("query"):
                modules += ["q", parser.step()]
            elif parser.match("key_value"):
                prefix = ".".join(modules)
                suffix = parser.step()
                key_param, value_param = param.chunk(2, dim=0)
                return {f"{prefix}.k.{suffix}": key_param, f"{prefix}.v.{suffix}": value_param}
            else:
                raise ValueError(key)
        elif parser.match("post_cross_attention_layernorm"):
            modules += ["norm3", parser.step()]
        elif parser.match("mlp"):
            modules.append("ffn")
            if parser.match("dense_h_to_4h"):
                modules.append("0")
            elif parser.match("dense_4h_to_h"):
                modules.append("2")
            else:
                raise ValueError(key)
            modules.append(parser.step())
        else:
            raise ValueError(key)
    elif parser.match("time_embed"):
        modules += ["time_embedding", parser.step(), parser.step()]
    elif parser.match("adaln_projection"):
        modules += ["time_projection", parser.step(), parser.step()]
    elif parser.match("text_embedding"):
        modules += ["text_embedding", parser.step(), parser.step()]
    elif parser.match("clip_proj"):
        if not parser.match("proj"):
            raise ValueError(key)
        modules += ["img_emb.proj", parser.step(), parser.step()]
    else:
        raise ValueError(key)

    if not parser.eof():
        raise ValueError(key)
    return {".".join(modules): param}


def load_deepspeed_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "module" in checkpoint:
        return checkpoint["module"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")


def convert_sat_to_wangp(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted = {}
    for key, value in sd.items():
        mappings = get_new_mappings(key, value)
        for new_key, new_value in mappings.items():
            if new_key in converted:
                print(f"Warning: duplicate converted key {new_key} from {key}")
            converted[new_key] = new_value
    return converted


def keep_fp32(key: str) -> bool:
    return key.startswith(("patch_embedding.", "pose_patch_embedding.", "mask_patch_embedding.", "head."))


def apply_dtype_policy(sd: Dict[str, torch.Tensor], policy: str) -> Dict[str, torch.Tensor]:
    if policy in ("", "none"):
        return sd
    if policy == "mbf16":
        for key, value in sd.items():
            sd[key] = value.float() if keep_fp32(key) else value.to(torch.bfloat16)
        return sd
    dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}
    if policy not in dtype_map:
        raise ValueError(f"Unsupported dtype policy: {policy}")
    target_dtype = dtype_map[policy]
    for key, value in sd.items():
        sd[key] = value.to(target_dtype)
    return sd


def convert_scail2_checkpoint(input_path: str, output_path: str, dtype_policy: str = "mbf16") -> None:
    state_dict = load_deepspeed_checkpoint(input_path)
    print("Converting SAT keys to WanGP format...")
    converted = convert_sat_to_wangp(state_dict)
    del state_dict
    converted = apply_dtype_policy(converted, dtype_policy)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving to: {output_path}")
    safetensors2.torch_write_file(converted, output_path, extra_meta={"format": f"wangp_scail2_{dtype_policy}", "converted_from": "scail2_sat_deepspeed"})
    print(f"Done. Converted {len(converted)} tensors.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scail-dir", default="SCAIL-2")
    parser.add_argument("--sat-model-path", default="model/1/fsdp2_rank_0000_checkpoint.pt")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--dtype-policy", default="mbf16", choices=["mbf16", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "none"])
    args = parser.parse_args()
    input_path = args.input_path or os.path.join(args.scail_dir, args.sat_model_path)
    convert_scail2_checkpoint(input_path, args.save_path, args.dtype_policy)


if __name__ == "__main__":
    main()
