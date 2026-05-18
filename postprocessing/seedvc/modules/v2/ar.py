import dataclasses
import json
import math
from collections import OrderedDict
from functools import partial, wraps
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # Initialize the model
    initializer_range: float = 0.02

    qk_norm: bool = False
    layerscale: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class NaiveModelArgs(BaseModelArgs):
    model_type: str = "naive"


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    token_targets: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        if init_weights:
            self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = "cuda"
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            ).to(device)

    def embed_base(self, x: Tensor, x_lens: Tensor) -> Tensor:
        for bib in range(x.size(0)):
            x[bib, x_lens[bib]:] = self.config.vocab_size - 1

        x_emb = self.embeddings(x)
        return x, x_emb

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(1)

        # Here we want to merge the embeddings of the codebooks
        # x = self.embed(inp)
        x = inp.clone()

        if input_pos is None:
            freqs_cis = self.freqs_cis[:seq_len].repeat(inp.size(0), 1, 1, 1)
        else:
            freqs_cis = self.freqs_cis[input_pos]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            mask = self.causal_mask[None, None, :seq_len, :seq_len]  # (B, N, Q, K)
            mask = mask & key_padding_mask[:, None, None, :].logical_not()

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        kv_pos: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile

        x = inp
        max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, kv_pos, :max_seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=kv_pos)

        x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs) -> None:
        super().__init__(config, init_weights=False)
        self.apply(self._init_weights)

    def forward(
        self,
        inp: Tensor,
        cond_lens: Tensor,
        target: Tensor,
        target_lens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        parent_result = super().forward(
            inp=inp,
            key_padding_mask=key_padding_mask,
            input_pos=input_pos,
        )
        token_logits = parent_result.logits

        # construct targets for token_logits
        token_targets = torch.zeros(token_logits.size(0), token_logits.size(1), dtype=torch.long,
                                    device=target.device) - 100
        for bib in range(token_targets.size(0)):
            token_targets[bib, cond_lens[bib] + 1:cond_lens[bib] + target_lens[bib] + 1] = target[bib, :target_lens[bib]]
            token_targets[bib, cond_lens[bib] + target_lens[bib] + 1] = self.config.vocab_size - 1
        return TransformerForwardResult(
            token_logits=token_logits,
            token_targets=token_targets,
        )

    def infer_slow(self, inp: Tensor, input_pos: Optional[Tensor] = None):
        # no kv cache used
        parent_result = super().forward(inp, input_pos=input_pos)
        latent = parent_result.hidden_states[:, -1]
        base_logits = parent_result.logits[:, -1]
        base_sampled, _ = topk_sampling(base_logits, top_k=-1, top_p=1.0)
        return base_sampled

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        kv_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        x = super().forward_generate(x, input_pos, kv_pos, vq_masks)
        return x

class NaiveWrapper(nn.Module):
    def __init__(self, model: NaiveTransformer) -> None:
        super().__init__()
        self.model = model
        self.sep_token_emb = nn.Parameter(torch.randn(model.config.dim))

    def setup_caches(self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = "cuda"):
        self.model.setup_caches(max_batch_size, max_seq_len, dtype, device)

    def forward(self, cond: Tensor, cond_lens: Tensor, x: Tensor, x_lens: Tensor) -> torch.Tensor:
        # style_emb = self.style_in(style).unsqueeze(1)  #  [B, 1, D]
        sep_token_emb = self.sep_token_emb.expand(x.size(0), 1, -1)
        _, x_emb = self.model.embed_base(x, x_lens)
        emb_seq_list = []
        for i in range(x.size(0)):
            emb_seq = torch.cat([
                sep_token_emb[i:i + 1],
                cond[i:i+1, :cond_lens[i]],
                sep_token_emb[i:i+1],
                x_emb[i:i+1, :x_lens[i]]], dim=1)
            emb_seq_list.append(emb_seq)
        max_len = max([emb_seq.size(1) for emb_seq in emb_seq_list])
        emb_seq = torch.cat([
            F.pad(emb_seq, (0, 0, 0, max_len - emb_seq.size(1)), value=0)
            for emb_seq in emb_seq_list
        ], dim=0)
        # input_pos = torch.arange(emb_seq.size(1), device=emb_seq.device).repeat(emb_seq.size(0), 1)
        input_pos = torch.zeros(emb_seq.size(0), emb_seq.size(1), device=emb_seq.device, dtype=torch.long)
        for i in range(x.size(0)):
            input_pos[i, :cond_lens[i] + 1] = torch.arange(cond_lens[i] + 1, device=emb_seq.device)
            input_pos[i, cond_lens[i] + 1: cond_lens[i] + x_lens[i] + 2] = torch.arange(x_lens[i] + 1, device=emb_seq.device)
        out = self.model(emb_seq, cond_lens, x, x_lens, input_pos=input_pos)
        loss = F.cross_entropy(out.token_logits.transpose(1, 2), out.token_targets.long(), ignore_index=-100)
        return loss

    @torch.no_grad()
    def infer(self, cond: Tensor) -> torch.Tensor:
        sep_token_emb = self.sep_token_emb.expand(1, 1, -1)
        emb_seq = torch.cat([sep_token_emb, cond, sep_token_emb], dim=1)
        pred_codes = []
        input_pos = torch.arange(cond.size(1) + 1, device=cond.device)
        for i in tqdm(range(4000)):
            input_pos = torch.cat([input_pos, torch.LongTensor([i]).to(cond.device)], dim=0)
            base = self.model.infer_slow(emb_seq, input_pos)
            if base == self.model.config.vocab_size - 1:
                break
            new_emb = self.model.embed_base(base, torch.LongTensor([1]).to(base.device))[1]
            emb_seq = torch.cat([emb_seq, new_emb], dim=1)
            pred_codes.append(base)
        return torch.cat(pred_codes, dim=-1)

    @torch.no_grad()
    def generate(
            self,
            prompt_text,
            prompt_target,
            compiled_decode_fn = None,
            **sampling_kwargs,
    ):
        sep_token_emb = self.sep_token_emb.expand(1, 1, -1)
        emb_seq = torch.cat([sep_token_emb, prompt_text, sep_token_emb], dim=1)
        input_pos = torch.arange(prompt_text.size(1) + 1, device=emb_seq.device)
        input_pos = torch.cat([input_pos, torch.LongTensor([0]).to(emb_seq.device)])
        prompt_target_emb = self.model.embed_base(prompt_target,torch.LongTensor([prompt_target.size(1)]).to(prompt_target.device))[1]
        emb_seq = torch.cat([emb_seq, prompt_target_emb], dim=1)
        input_pos = torch.cat([input_pos, torch.arange(prompt_target_emb.size(1)).to(input_pos.device) + 1])

        pred_codes = []
        kv_pos = torch.arange(emb_seq.size(1), device=emb_seq.device)
        next_tokens = self.decode_one_token_ar(emb_seq, input_pos, kv_pos, suppress_tokens=[self.model.config.vocab_size - 1], **sampling_kwargs)
        pred_base = next_tokens[0]
        pred_codes.append(pred_base)
        new_emb = self.model.embed_base(pred_base.unsqueeze(0), torch.LongTensor([1]).to(pred_base.device))[1]
        emb_seq = torch.cat([emb_seq, new_emb], dim=1)
        for _ in tqdm(range(4000)):
            suppress_eos = len(pred_codes) < 10
            input_pos = input_pos[-1:] + 1
            kv_pos = kv_pos[-1:] + 1
            next_tokens = self.decode_one_token_ar(
                emb_seq[:, -1:].reshape(1, 1, -1),
                input_pos.reshape(1),
                kv_pos.reshape(1),
                previous_tokens=torch.cat(pred_codes),
                suppress_tokens=[self.model.config.vocab_size - 1] if suppress_eos else None,
                compiled_decode_fn=compiled_decode_fn,
                **sampling_kwargs)
            pred_base = next_tokens[0]
            if pred_base == self.model.config.vocab_size - 1:
                break
            pred_codes.append(pred_base.clone())
            new_emb = self.model.embed_base(pred_base.unsqueeze(0), torch.LongTensor([1]).to(pred_base.device))[1]
            emb_seq = torch.cat([emb_seq, new_emb], dim=1)
        return torch.stack(pred_codes, dim=-1)

    def decode_one_token_ar(
            self,
            x: torch.Tensor,
            input_pos: torch.Tensor,
            kv_pos: torch.Tensor,
            previous_tokens: torch.Tensor = None,
            compiled_decode_fn = None,
            **sampling_kwargs,
    ) -> torch.Tensor:
        if compiled_decode_fn is not None:
            x = compiled_decode_fn(x, input_pos, kv_pos)
        else:
            x = self.model.forward_generate(x, input_pos, kv_pos)

        sampling_kwargs_main = sampling_kwargs.copy()
        codebooks = [
            sample(
                x.logits,
                previous_tokens=(
                    previous_tokens[0] if previous_tokens is not None else None
                ),
                **sampling_kwargs_main,
            )[0]
        ]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.qk_norm = config.qk_norm
        self.qk_norm_groups = 1
        self.qk_norm_scale = 10
        self.qk_norm_dim_scale = False
        self.qk_norm_q_scale = self.qk_norm_k_scale = 1

        if self.qk_norm and self.qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(self.n_head, 1, self.head_dim))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(self.n_head, 1, self.head_dim))
    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                    # No third party attn_mask here to use flash_attention
                )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(x.size(0), xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    current_logprobs = logprobs[torch.arange(logprobs.shape[0]), token.squeeze(1)]
    return token, current_logprobs

def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[List[int]] = None,
    temperature: torch.Tensor = 0.7,
    top_p: torch.Tensor = 0.7,
    repetition_penalty: torch.Tensor = 1.5,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)
    if suppress_tokens is not None:
        for token in suppress_tokens:
            logits[token] = -float("Inf")

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs
