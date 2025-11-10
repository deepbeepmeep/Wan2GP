
from typing import Mapping, Dict
import torch

def convert_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a ComfyUI-formatted Wan/Qwen VAE state_dict to Diffusers format.

    Input:  dict-like mapping from str -> torch.Tensor (e.g. loaded from safetensors)
    Output: new dict with Diffusers key names (no mutation of the input)
    """

    # Exact key remaps for middle resnets (encoder/decoder)
    middle_key_mapping = {
        # Encoder middle resnets
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",

        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",

        # Decoder middle resnets
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",

        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Exact key remaps for the mid attention blocks (encoder/decoder)
    attention_mapping = {
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",

        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Heads (norm_out / conv_out) for encoder/decoder
    head_mapping = {
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",

        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Latent quantization bridges
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    out: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        # 1) Direct dictionary remaps
        if key in middle_key_mapping:
            out[middle_key_mapping[key]] = value
            continue
        if key in attention_mapping:
            out[attention_mapping[key]] = value
            continue
        if key in head_mapping:
            out[head_mapping[key]] = value
            continue
        if key in quant_mapping:
            out[quant_mapping[key]] = value
            continue

        # 2) Conv-in aliases for encoder/decoder
        if key == "encoder.conv1.weight":
            out["encoder.conv_in.weight"] = value
            continue
        if key == "encoder.conv1.bias":
            out["encoder.conv_in.bias"] = value
            continue
        if key == "decoder.conv1.weight":
            out["decoder.conv_in.weight"] = value
            continue
        if key == "decoder.conv1.bias":
            out["decoder.conv_in.bias"] = value
            continue

        # 3) Encoder down path (downsamples.* -> down_blocks.*)
        if key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Residual -> (norm1/conv1/norm2/conv2), shortcut passthrough
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")

            out[new_key] = value
            continue

        # 4) Decoder up path (upsamples.* -> up_blocks.*)
        if key.startswith("decoder.upsamples."):
            parts = key.split(".")
            # format: decoder.upsamples.{block_idx}.(residual|resample|time_conv|shortcut)...
            if len(parts) >= 3 and parts[2].isdigit():
                block_idx = int(parts[2])

                # 4a) Residual groups: map flat indices -> (up_block_id, resnet_id)
                if "residual" in key:
                    if block_idx in (0, 1, 2):
                        up_block_id, resnet_id = 0, block_idx
                    elif block_idx in (4, 5, 6):
                        up_block_id, resnet_id = 1, block_idx - 4
                    elif block_idx in (8, 9, 10):
                        up_block_id, resnet_id = 2, block_idx - 8
                    elif block_idx in (12, 13, 14):
                        up_block_id, resnet_id = 3, block_idx - 12
                    else:
                        # keep unmapped residuals as-is
                        out[key] = value
                        continue

                    if ".residual.0.gamma" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.norm1.gamma"
                    elif ".residual.2.bias" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv1.bias"
                    elif ".residual.2.weight" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv1.weight"
                    elif ".residual.3.gamma" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.norm2.gamma"
                    elif ".residual.6.bias" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv2.bias"
                    elif ".residual.6.weight" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv2.weight"
                    else:
                        new_key = key
                    out[new_key] = value
                    continue

                # 4b) Shortcut convs
                if ".shortcut." in key:
                    if block_idx == 4:
                        # special-case first shortcut in block 1 -> resnets.0.conv_shortcut
                        new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                        new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                    else:
                        new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                        new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
                    out[new_key] = value
                    continue

                # 4c) Upsamplers & time conv placement (the 3,7,11 pattern)
                if (".resample." in key) or (".time_conv." in key):
                    if block_idx == 3:
                        new_key = key.replace("decoder.upsamples.3", "decoder.up_blocks.0.upsamplers.0")
                    elif block_idx == 7:
                        new_key = key.replace("decoder.upsamples.7", "decoder.up_blocks.1.upsamplers.0")
                    elif block_idx == 11:
                        new_key = key.replace("decoder.upsamples.11", "decoder.up_blocks.2.upsamplers.0")
                    else:
                        new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    out[new_key] = value
                    continue

            # default: just change the container name
            out[key.replace("decoder.upsamples.", "decoder.up_blocks.")] = value
            continue

        # 5) Fallback: preserve anything not covered above
        out[key] = value

    return out
