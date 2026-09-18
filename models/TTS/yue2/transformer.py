"""Separate YuE2 AR and acoustic transformers, managed by MMGP."""

from dataclasses import replace

import torch
from torch import nn
import torch.nn.functional as F
from mmgp import offload

from shared.attention import pay_attention
from shared.llm_engines.nanovllm.models.qwen3 import Qwen3ForCausalLM
from shared.llm_engines.nanovllm.layers.attention import Attention as PagedAttention
from shared.llm_engines.nanovllm.layers.layernorm import RMSNorm as EngineRMSNorm
from shared.llm_engines.nanovllm.layers.activation import SiluAndMul
from shared.llm_engines.nanovllm.vllm_support import probe_vllm_runtime
from shared.llm_engines.nanovllm.utils.context import get_context, set_context

from .modules import Attention, AudioPositionEmbedding, MLP, RMSNorm, RotaryEmbedding, TimestepEmbedder


def attend(q, k, v, engine, causal=False):
    groups = q.shape[-2] // k.shape[-2]
    if groups != 1:
        k, v = k.repeat_interleave(groups, dim=-2), v.repeat_interleave(groups, dim=-2)
    return pay_attention([q, k, v], causal=causal, force_attention="flash" if engine == "vllm" else "sdpa")


class YuE2AR(Qwen3ForCausalLM):
    _offload_hooks = ["condition"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.engine = "legacy"
        self.abort_fn = lambda: False

    def configure_engine(self, engine, abort_fn):
        if engine == "vllm" and not probe_vllm_runtime()["supported"]:
            raise RuntimeError("YuE2 vllm mode requires Triton and FlashAttention 2")
        self.engine, self.abort_fn = engine, abort_fn
        self.config._prompt_enhancer_safe_legacy = engine != "vllm"
        for module in self.modules():
            if isinstance(module, PagedAttention):
                if engine != "vllm":
                    module.flash_attn_varlen_func = module.flash_attn_with_kvcache = None
                module.use_triton_kv_cache = engine == "vllm"
            elif isinstance(module, EngineRMSNorm):
                module.use_triton_rmsnorm = engine == "vllm"
            elif isinstance(module, SiluAndMul):
                module.use_triton = engine == "vllm"

    @torch.inference_mode()
    def forward(self, input_ids, positions):
        x = self.model.embed_tokens(input_ids)
        residual = None
        context = get_context()
        branches = []
        if context.is_prefill:
            offsets = context.cu_seqlens_q.tolist()
            key_offsets = context.cu_seqlens_k.tolist()
            count = len(offsets) - 1
        else:
            count = x.shape[0]
            offsets = range(count + 1)
        if count > 1:
            for index in range(count):
                start, end = offsets[index:index + 2]
                branch = replace(context, slot_mapping=context.slot_mapping[start:end], block_tables=context.block_tables[index:index + 1] if context.block_tables is not None else None)
                if context.is_prefill:
                    branch.cu_seqlens_q = context.cu_seqlens_q[index:index + 2] - start
                    branch.cu_seqlens_k = context.cu_seqlens_k[index:index + 2] - key_offsets[index]
                    branch.max_seqlen_q = end - start
                    branch.max_seqlen_k = key_offsets[index + 1] - key_offsets[index]
                else:
                    branch.context_lens = context.context_lens[index:index + 1]
                branches.append((slice(start, end), branch))
        try:
            for layer in self.model.layers:
                if branches:
                    # Both CFG paths share the same loaded layer, processed sequentially.
                    outputs, residuals = [], []
                    for span, branch in branches:
                        set_context(**vars(branch))
                        h, r = layer(positions[span], x[span], residual[span] if residual is not None else None)
                        outputs.append(h)
                        residuals.append(r)
                    x, residual = torch.cat(outputs), torch.cat(residuals)
                else:
                    x, residual = layer(positions, x, residual)
                if self.abort_fn():
                    raise InterruptedError("YuE2 AR interrupted")
        finally:
            set_context(**vars(context))
        return self.model.norm(x, residual)[0]

    @torch.inference_mode()
    def condition(self, token_ids):
        """Cache each layer's AR keys/values for acoustic flow matching."""
        positions = torch.arange(len(token_ids), device="cuda")
        x = self.model.embed_tokens(torch.tensor(token_ids, device="cuda"))
        cache = []
        for layer in self.model.layers:
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(h).view(-1, attn.num_heads, attn.head_dim))
            k = attn.k_norm(attn.k_proj(h).view(-1, attn.num_kv_heads, attn.head_dim))
            v = attn.v_proj(h).view(-1, attn.num_kv_heads, attn.head_dim)
            q, k = attn.rotary_emb(positions, q, k)
            cache.append((k, v))
            h = attend(q[None], k[None], v[None], self.engine, causal=True)
            x = x + attn.o_proj(h.flatten(2)[0])
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if self.abort_fn():
                raise InterruptedError("YuE2 acoustic conditioning interrupted")
        return cache


class AcousticLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nar_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_self_attn = Attention(config)
        self.nar_pre_mlp_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_mlp = MLP(config)
        self.ffn_chunk_size = config.ffn_chunk_size

    def forward(self, x_list, cache, cos, sin, engine):
        x = x_list.pop()
        # x remains live only as the residual; normalized activations are consumed.
        x.add_(self.nar_self_attn([self.nar_input_layernorm([x])], cache, cos, sin, engine))
        # Keep long-song FFN intermediates bounded; each token is independent.
        chunk_size = min(self.ffn_chunk_size, max(1, x.shape[1] * x.shape[2] // self.nar_mlp.gate_proj.out_features))
        for start in range(0, x.shape[1], chunk_size):
            part = x[:, start:start + chunk_size]
            part.add_(self.nar_mlp([self.nar_pre_mlp_layernorm([part])]))
        return x


class AcousticBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([AcousticLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class YuE2Acoustic(nn.Module):
    _offload_hooks = ["synthesize"]

    def preprocess_loras(self, model_type, sd):
        # Native YuE2 adapters omit the weight suffix required by MMGP.
        return {key + ".weight" if key.endswith((".lora_A", ".lora_B")) else key: value for key, value in sd.items()}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AcousticBackbone(config)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(config.max_latent_frames, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.engine = "legacy"
        self.abort_fn = lambda: False

    def forward(self, state_list, raw_t, cache, cos, sin, position_embedding):
        state = state_list.pop()
        t = torch.sigmoid(raw_t.to(dtype=state.dtype))
        shift = self.config.timestep_shift
        t = shift * t / (1 + (shift - 1) * t)
        x = self.vae2llm(F.pad(state, (0, 0, 1, 1))[None])
        del state
        x.add_(self.time_embedder(t.reshape(1))[None]).add_(position_embedding)
        for layer, layer_cache in zip(self.model.layers, cache):
            x_list = [x]
            x = None
            x = layer(x_list, layer_cache, cos, sin, self.engine)
            if self.abort_fn():
                raise InterruptedError("YuE2 acoustic synthesis interrupted")
        x_list = [x]
        x = None
        return self.llm2vae(self.model.norm(x_list))[0, 1:-1]

    @torch.inference_mode()
    def synthesize(self, noise, cache_list, ar_length, steps, report):
        cache = cache_list.pop()
        state = noise.to(device="cuda", dtype=self.vae2llm.weight.dtype)
        length = len(state) + 2
        positions = torch.arange(ar_length, ar_length + length, device=state.device)[None]
        cos, sin = self.rotary(positions)
        pos = self.latent_pos_embed(torch.arange(length, device=state.device))[None]
        raw_steps = torch.logit(torch.arange(2 * steps, 0, -1, dtype=torch.float64) / (2 * steps)).clamp(-20, 20).to(state.device)
        try:
            for step in range(steps):
                offload.set_step_no_for_lora(self, step)
                # The integrator retains state, but transfers ownership of the midpoint.
                first = self([state], raw_steps[2 * step], cache, cos, sin, pos)
                first.div_(2 * steps)
                mid_list = [state - first]
                del first
                state.sub_(self(mid_list, raw_steps[2 * step + 1], cache, cos, sin, pos).div_(steps))
                report(step)
            return state.float().cpu()
        finally:
            cache.clear()
            self.rotary._inv_freq = None
