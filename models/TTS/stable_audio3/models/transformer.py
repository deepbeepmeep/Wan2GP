from functools import reduce, partial
from packaging import version
import logging
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from torch.nn.utils.parametrizations import weight_norm
from typing import Callable, Literal, Optional
from shared.attention import pay_attention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    flex_attention_available = False

try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
except ImportError as e:
    print(e)
    print('flash_attn varlen/bert_padding not available, disabling varlen attention')
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    index_first_axis = None


def precompute_varlen_metadata(padding_mask: torch.Tensor):
    """
    Precompute varlen attention metadata once to avoid recomputation in every attention layer.

    Args:
        padding_mask: Boolean tensor of shape (batch, seq_len) where True = valid

    Returns:
        Dict with cu_seqlens, max_seqlen, indices, batch_size, seq_len for use in attention
    """
    if padding_mask is None or unpad_input is None:
        return None

    batch_size, seq_len = padding_mask.shape

    # Compute cumulative sequence lengths (same for all of q, k, v)
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()

    # Compute indices for gathering valid tokens
    # indices maps from packed position -> original (batch, seq) position
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()

    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "indices": indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

def _left_pad_to_match(emb, target_len):
    """Left-pad or right-trim emb along seq dim to match target_len.

    Used for local conditioning embeddings that need to align with x
    without affecting prepended tokens (memory tokens, global cond, etc.).
    """
    emb_len = emb.shape[-2]
    if emb_len < target_len:
        return F.pad(emb, (0, 0, target_len - emb_len, 0), value=0.)
    elif emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb

if flex_attention_available:
    try:
        flex_attention_compiled = flex_attention
    except Exception as e:
        logging.debug(f"Could not compile flex_attention, using uncompiled version: {e}")
        flex_attention_compiled = flex_attention
else:
    flex_attention_compiled = None


# Cache band block_masks for sliding-window attention fallback (flex_attention path).
# Keyed by (seq_q, seq_k, w_left, w_right, device). create_block_mask is expensive
# but the result is reused across all transformer layers and forward passes.
_SLIDING_WINDOW_BLOCK_MASK_CACHE = {}

def _get_sliding_window_block_mask(seq_q, seq_k, w_left, w_right, device):
    key = (seq_q, seq_k, int(w_left), int(w_right), str(device))
    bm = _SLIDING_WINDOW_BLOCK_MASK_CACHE.get(key)
    if bm is None:
        wl, wr = int(w_left), int(w_right)
        def _band_mod(b, h, q_idx, kv_idx):
            delta = kv_idx - q_idx
            return (delta >= -wl) & (delta <= wr)
        bm = create_block_mask(_band_mod, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_k, device=device)
        _SLIDING_WINDOW_BLOCK_MASK_CACHE[key] = bm
    return bm

def _sliding_window_additive_mask(seq_q, seq_k, w_left, w_right, device, dtype):
    """Build a (seq_q, seq_k) additive mask for masked SDPA fallback.
    0 inside the band [i - w_left, i + w_right], -inf outside.
    """
    ii = torch.arange(seq_q, device=device)
    jj = torch.arange(seq_k, device=device)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
    mask = torch.zeros((seq_q, seq_k), dtype=dtype, device=device)
    return mask.masked_fill(~in_band, float('-inf'))


# Chunked-halo SDPA fallback. Math-equivalent to masked SDPA with a band
# mask, but processes queries in non-overlapping chunks with a (w_left,
# w_right) halo of keys/values on each side â€” every query stays inside its
# chunk's softmax. Avoids materializing the O(N^2) mask.
#
# At realistic SAME-L decoder shapes (N=69632, W=17, packed sequence is
# latent_length * (stride+1)): ~34x faster than full masked SDPA, and
# ~140x less peak mask memory (~1 MB per chunk vs 9.7 GB for one N x N mask).
# Chunk size is a tunable; 1024 is a good default at typical pretransform
# decoder shapes. Larger chunks waste more compute on out-of-band tiles;
# smaller chunks suffer from launch overhead.
_SLIDING_WINDOW_CHUNK_SIZE = 1024

def _sliding_window_chunked_halo_sdpa(q, k, v, w_left, w_right, chunk_size=_SLIDING_WINDOW_CHUNK_SIZE):
    B, H, N, D = q.shape
    outs = []
    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        k_start = max(0, q_start - int(w_left))
        k_end = min(N, q_end + int(w_right))
        q_c = q[..., q_start:q_end, :]
        k_c = k[..., k_start:k_end, :]
        v_c = v[..., k_start:k_end, :]
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        delta = k_idx[None, :] - q_idx[:, None]
        in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device).masked_fill(~in_band, float('-inf'))
        outs.append(F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=mask, is_causal=False))
    return torch.cat(outs, dim=-2)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    # Preserve autocast context during recomputation to avoid dtype mismatches
    if "context_fn" not in kwargs:
        from torch.amp import autocast
        import functools
        # Get current autocast state
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype('cuda')
            def get_contexts():
                return (
                    autocast('cuda', dtype=dtype),
                    autocast('cuda', dtype=dtype),
                )
            kwargs["context_fn"] = get_contexts
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class RunningInstanceNorm(nn.Module):
    def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1,1,dim))
        self.register_buffer("running_std", torch.ones(1,1,dim))
        self.saturate = saturate
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.trainable_gain = trainable_gain
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(1))
    
    def _update_stats(self, x):
        self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
        self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        x = (x - self.running_mean) / self.running_std
        if self.saturate:
            x = torch.asinh(x)
        if self.trainable_gain:
            x = x * self.gain
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    #@autocast("cuda", enabled = False)
    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, fix_scale=False, force_fp32=False, eps=1e-5):
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=self.eps)
        else:
            output = F.rms_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), eps=self.eps)
            return output.to(x.dtype)

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(3.14159265359 * x)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        sinusoidal = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU() if not sinusoidal else Sin()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x, varlen_metadata=None):
        if varlen_metadata is not None and index_first_axis is not None and pad_input is not None:
            # Pack valid tokens for efficient FFN computation (skip padding tokens)
            # Padding positions become zeros after unpack, which is fine since FFN output
            # is added to residual, preserving values at padding positions
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            indices = varlen_metadata["indices"]
            dim = x.shape[-1]

            # Pack to (N_valid, D)
            x_packed = index_first_axis(x.reshape(-1, dim), indices)

            # FFN on packed representation with pseudo-batch dim
            x_packed = self.ff(x_packed.unsqueeze(0)).squeeze(0)

            # Unpack back to (B, T, D)
            return pad_input(x_packed, indices, batch_size, seq_len)
        else:
            return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm_eps = 1e-6,
        qk_norm: Literal['l2', 'ln', 'rms', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads

        self.differential = differential

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            if differential:
                self.to_q = nn.Linear(dim, dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            if differential:
                self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'rms', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "rms" ,"dyt", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k


    def apply_attn(self, q, k, v, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None, flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None):
        if self.num_heads != self.kv_heads:
             heads_per_kv_head = self.num_heads // self.kv_heads
             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        if padding_mask is not None:
            v = v * padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)

        q, k, v = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k, v))
        qkv_list = [q, k, v]
        window_size = flash_attn_sliding_window if flash_attn_sliding_window is not None else (-1, -1)
        force_attention = "flash" if flash_attn_sliding_window is not None else None
        out = pay_attention(qkv_list, causal=causal if causal is not None else False, window_size=window_size, force_attention=force_attention)
        return rearrange(out, "b n h d -> b h n d")


    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        rotary_pos_emb_k = None,
        causal = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
                k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)
            # q = self.q_norm(q)
            # k = self.k_norm(k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
        
            q_freqs = freqs

            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs

                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    q_freqs, k_freqs = ratio * freqs, freqs

            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        
        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.differential:
            q, q_diff = q.unbind(dim = 1)
            k, k_diff = k.unbind(dim = 1)
            out = self.apply_attn(q, k, v,  causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out = out - out_diff
        else:
            out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(out.dtype)  # (b, n, 1)
                out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = out.mean(dim=-2, keepdim=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            local_add_cond_dim = None,
            modular_local_cond_configs = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            add_rope = False,
            layer_scale = False,
            norm_type = 'layer_norm',
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads,dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
       
        if layer_scale and zero_init_branch_outputs:
            print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
            zero_init_branch_outputs = False
        
        if norm_type not in ['layer_norm', 'rms_norm', 'dyt']:
            raise ValueError(f'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], got {norm_type}')

        norm_layer_map = {
            'layer_norm': LayerNorm,
            'rms_norm': RMSNorm,
            'dyt': DynamicTanh
        }
        norm_layer = norm_layer_map[norm_type]

        self.pre_norm = norm_layer(dim,**norm_kwargs)
        self.add_rope = add_rope

        self.self_attn = Attention(
            dim,
            dim_heads = self.dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads = self.dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.layer_ix = layer_ix

        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

        self.local_add_cond_dim = local_add_cond_dim

        if local_add_cond_dim is not None:
            self.to_local_embed = nn.Sequential(
                nn.Linear(local_add_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

            nn.init.zeros_(self.to_local_embed[-1].weight)
            nn.init.zeros_(self.to_local_embed[-1].bias)

        else:
            self.to_local_embed = None

        # Modular local conditioning - independent projections per conditioning ID
        self.modular_local_cond_configs = modular_local_cond_configs or []
        self.modular_local_embeds = nn.ModuleDict()

        for config in self.modular_local_cond_configs:
            cond_id = config["id"]
            cond_dim = config["dim"]
            proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
            # Zero-init output layer so new conditioning doesn't affect model initially
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.modular_local_embeds[cond_id] = proj

        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None

    def _apply_local_conditioning(self, x, local_add_cond, modular_local_cond):
        """Apply local additive and modular local conditioning to x."""
        if local_add_cond is not None and self.to_local_embed is not None:
            local_emb = self.to_local_embed(local_add_cond)
            x = x + _left_pad_to_match(local_emb, x.shape[-2])

        if modular_local_cond is not None and len(self.modular_local_embeds) > 0:
            modular_sum = None
            for cond_id, proj in self.modular_local_embeds.items():
                if cond_id in modular_local_cond:
                    local_emb = proj(modular_local_cond[cond_id])
                    local_emb = _left_pad_to_match(local_emb, x.shape[-2])
                    modular_sum = local_emb if modular_sum is None else modular_sum + local_emb
            if modular_sum is not None:
                x = x + modular_sum

        return x

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        rotary_pos_emb = None,
        cross_attn_rotary_pos_emb = None,
        self_attention_block_mask = None,
        self_attention_score_mod = None,
        cross_attention_block_mask = None,
        cross_attention_score_mod = None,
        self_attention_flash_sliding_window = None,
        cross_attention_flash_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x, varlen_metadata=varlen_metadata)
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual

        else:
            x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata))

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                    
            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            x = x + self.ff_scale(self.ff(self.ff_norm(x), varlen_metadata=varlen_metadata))
            

        return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        local_add_cond_dim=None,
        modular_local_cond_configs=None,
        causal=False,
        rotary_pos_emb=True,
        cross_attn_rotary_pos_emb=False,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        if cross_attn_rotary_pos_emb:
            self.cross_attn_rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.cross_attn_rotary_pos_emb = None

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    local_add_cond_dim = local_add_cond_dim,
                    modular_local_cond_configs = modular_local_cond_configs,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        context = None,
        prepend_embeds = None,
        global_cond = None,
        local_add_cond = None,
        modular_local_cond = None,
        return_info = False,
        use_checkpointing = True,
        exit_layer_ix = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.cross_attn_rotary_pos_emb is not None:
            cross_attn_rotary_pos_emb = self.cross_attn_rotary_pos_emb.forward_from_seq_len(context.shape[-1])
        else:
            cross_attn_rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Extend padding mask for prepended tokens if provided
        extended_padding_mask = None
        varlen_metadata = None
        if padding_mask is not None:
            # Compute total prepend length (memory tokens + prepend_embeds)
            prepend_length = self.num_memory_tokens
            if prepend_embeds is not None:
                prepend_length += prepend_embeds.shape[1]

            # Prepend tokens are always valid for attention
            if prepend_length > 0:
                prepend_valid = torch.ones(batch, prepend_length, device=device, dtype=torch.bool)
                extended_padding_mask = torch.cat([prepend_valid, padding_mask], dim=-1)
            else:
                extended_padding_mask = padding_mask

            # Precompute varlen metadata once for all layers (major performance optimization)
            # Only compute if varlen attention is actually available
            if flash_attn_varlen_func is not None and index_first_axis is not None:
                varlen_metadata = precompute_varlen_metadata(extended_padding_mask)

        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):
            abort_callback = getattr(self, "abort_callback", None)
            if callable(abort_callback) and abort_callback():
                raise InterruptedError("Stable Audio 3 generation aborted.")

            layer_kwargs = {
                "context": context,
                "rotary_pos_emb": rotary_pos_emb,
                "cross_attn_rotary_pos_emb": cross_attn_rotary_pos_emb,
                "global_cond": global_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "self_attention_flash_sliding_window": self.sliding_window,
                "padding_mask": extended_padding_mask,
                "varlen_metadata": varlen_metadata
            }

            if use_checkpointing:
                x = checkpoint(layer, x, **layer_kwargs, **kwargs)
            else:
                x = layer(x, **layer_kwargs, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

            if callable(abort_callback) and abort_callback():
                raise InterruptedError("Stable Audio 3 generation aborted.")

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x

