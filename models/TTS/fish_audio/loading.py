# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
import json
from pathlib import Path
import torch
from mmgp import offload
from .config import FishOmniConfig
from .backbone import FishOmniBackbone
from .tower import FishOmniAudioDecoder
from .layers import GatedRMSNorm, ZeroCenteredRMSNorm


def backbone_split_map(config):
    cfg = config.text
    width = cfg.num_attention_heads * cfg.head_dim
    def split_query_gate(*, src, **kwargs):
        shaped = src.reshape(cfg.num_attention_heads, 2, cfg.head_dim, *src.shape[1:])
        return [shaped[:, index].reshape(width, *src.shape[1:]).contiguous() for index in (0, 1)]
    return {"self_attn.q_proj": {"mapped_modules": ["self_attn.query_proj", "self_attn.gate_proj"],
                                  "split_sizes": [width, width], "split_handlers": {field: split_query_gate for field in ("weight", "weight._data", "weight._scale")}}}


def tower_split_map(config):
    cfg = config.audio_decoder
    q, kv = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
    return {"attention.wqkv": {"mapped_modules": ["attention.wq", "attention.wg", "attention.wk", "attention.wv"], "split_sizes": [q, q, kv, kv]}}


def audio_head_rows(config):
    """Keep only the LM-head rows decoding samples from (views of the checkpoint rows, nothing copied); a head already reduced to
    them is kept as is. ConvRot scales are per row."""
    start, stop = config.audio_logit_rows
    def preprocess(state_dict):
        for key in ("lm_head.weight", "lm_head.weight_scale"):
            if key in state_dict and state_dict[key].shape[0] == config.text.vocab_size:
                state_dict[key] = state_dict[key][start:stop]
        return state_dict
    return preprocess


def load_transformers(backbone_paths, codebook_path, engine="legacy"):
    """backbone_paths: the dense backbone file, then the routed-expert file (pageable mmap, outside MMGP's block offload)."""
    config = FishOmniConfig.from_dict(json.loads(Path(__file__).with_name("fish_drama3.json").read_text()))
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, engine, config.audio_logit_rows, config.audio_decoder.vq_backbone_codebooks)
        codebooks = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    for layer in (*codebooks.shared_layers, *codebooks.residual_layers):
        layer.transformer_layer.attention.engine = engine
    for module in (*backbone.modules(), *codebooks.modules()):
        if isinstance(module, (ZeroCenteredRMSNorm, GatedRMSNorm)):
            module.fused = engine == "vllm"
    offload.load_model_data(backbone, [str(path) for path in backbone_paths], writable_tensors=False, default_dtype=None, preprocess_sd=audio_head_rows(config),
                            fused_split_map=backbone_split_map(config))
    offload.load_model_data(codebooks, str(codebook_path), writable_tensors=False, default_dtype=None,
                            fused_split_map=tower_split_map(config))
    # The most picked experts get the GPU copies a preload allows first (pick frequencies, comparable across layers).
    usage = torch.tensor(json.loads(Path(__file__).with_name("expert_usage.json").read_text(encoding="utf-8"))["counts"], dtype=torch.float64)
    for layer, counts in zip(backbone.model.layers, usage):
        layer.mlp.experts.expert_priority = (counts / counts.sum()).tolist()
    return config, backbone.eval(), codebooks.eval()


def load_codec(path):
    from .codec import create_codec, Attention
    with torch.device("meta"):
        codec = create_codec()
    splits = {}
    for name, module in codec.named_modules():
        if isinstance(module, Attention):
            prefix = name + "."
            splits[prefix + "wqkv"] = {"mapped_modules": [prefix + key for key in ("wq", "wk", "wv")],
                "split_sizes": [module.n_head * module.head_dim, module.n_local_heads * module.head_dim, module.n_local_heads * module.head_dim]}
    offload.load_model_data(codec, str(path), writable_tensors=False, default_dtype=None, fused_split_map=splits)
    codec._convertWeightsFloatTo = None
    codec._model_dtype = next(codec.parameters()).dtype
    return codec.eval()
