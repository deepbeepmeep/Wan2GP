"""SageAttention (1): INT8 q/k per block of tokens and fp16 v, the Triton kernels of sageattn_varlen (SageAttention 1.0.6, or the copy
SageAttention 2 keeps in its triton package), launched without its device sync (the sizes of its scales read on the GPU) so that CUDA
graphs can record them, and in stages for the attention kit: v, k and q quantized as soon as each is computed, the output written into
q. Same values as sageattn_varlen for sequences of the same length (all batch items), head_dim 64 or 128, no mask."""
import torch

try:  # SageAttention 2
    from sageattention.triton.quant_per_block_varlen import quant_per_block_int8_kernel
    from sageattention.triton.attn_qk_int8_block_varlen import _attn_fwd
    _SM_SCALE = {64: 1.0 / 64 ** 0.5, 128: 1.0 / 128 ** 0.5}  # sageattn_varlen's softmax scales, computed as each version does
except ImportError:  # SageAttention 1.0.6
    from sageattention.quant_per_block_varlen import quant_per_block_int8_kernel
    from sageattention.attn_qk_int8_block_varlen import _attn_fwd
    _SM_SCALE = {64: 64 ** -0.5, 128: 128 ** -0.5}

SETTINGS = {}  # the staged functions take the settings of the staged_* functions of shared.sage2_core, none here
HEAD_DIMS = tuple(_SM_SCALE)


def _offsets(batch, length, device):
    """[0, length, 2 length, ...] of the batch items, made on the GPU (a host copy would not be allowed in a CUDA graph)."""
    return torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device=device)


def _quantize(x, block, sm_scale):
    """x (batch, tokens, heads, head_dim), batch items one after another in memory, as INT8 rows (batch * tokens, heads, head_dim) and
    scales per block of tokens and head, with the offsets of the batch items in both."""
    batch, tokens, heads, head_dim = x.shape
    blocks = (tokens + block - 1) // block
    x_int8 = torch.empty((batch * tokens, heads, head_dim), dtype=torch.int8, device=x.device)
    scale = torch.empty((batch * blocks, heads), dtype=torch.float32, device=x.device)
    offsets, scale_offsets = _offsets(batch, tokens, x.device), _offsets(batch, blocks, x.device)
    quant_per_block_int8_kernel[(blocks, heads, batch)](x, x_int8, scale, offsets, scale_offsets, x.stride(2), x.stride(1), x_int8.stride(1), x_int8.stride(0),
                                                        sm_scale=sm_scale, H=heads, C=head_dim, BLK=block)
    return x_int8, scale, offsets, scale_offsets


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    """fp16 v; v is released."""
    v = v_list[0]
    v_list.clear()
    return v.to(torch.float16), None, None


def _quantize_k(k, in_place):
    """INT8 k minus its mean over the tokens of all batch items, with its scales."""
    rows = k.view(-1, *k.shape[2:])
    mean = rows.mean(dim=0, keepdim=True)
    return _quantize((rows.sub_(mean) if in_place else rows - mean).view(k.shape), 64, 1.0)


@torch.compiler.disable
def staged_quantize_k(k_list, settings):
    """INT8 k minus its mean over the tokens of all batch items, with its scales; k is released (the mean is subtracted in place)."""
    k = k_list[0]
    k_list.clear()
    return _quantize_k(k, True)


@torch.compiler.disable
def staged_quantize_q(q, settings):
    """INT8 q with its scales (the softmax scale, in base 2, goes into them); q is not released (staged_attention can write the output
    into it)."""
    return _quantize(q, 128, _SM_SCALE[q.shape[-1]] * 1.44269504)


@torch.compiler.disable
def staged_attention(out, quantized_list, settings, attn_mask=None):
    """Attention of the staged [q, k, v], written into out and returned: out has q's shape (batch, tokens, heads, head_dim) and its
    batch items one after another in memory: q itself, or a slice of heads of a larger output. Fewer heads in k/v: grouped query
    attention."""
    (q_int8, q_scale, q_offsets, q_scale_offsets), (k_int8, k_scale, k_offsets, k_scale_offsets), (v, _, _) = quantized_list
    quantized_list.clear()
    batch, tokens, heads, head_dim = out.shape
    rows, v = out.view(-1, heads, head_dim), v.view(-1, *v.shape[2:])
    _attn_fwd[((tokens + 127) // 128, heads, batch)](
        q_int8, k_int8, v, q_offsets, k_offsets, q_scale, k_scale, q_scale_offsets, k_scale_offsets, rows,
        q_int8.stride(1), q_int8.stride(0), k_int8.stride(1), k_int8.stride(0), v.stride(1), v.stride(0), rows.stride(1), rows.stride(0),
        heads, heads // k_int8.shape[1], BLOCK_M=128, BLOCK_N=64, HEAD_DIM=head_dim, STAGE=1, num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)
    return out


@torch.compiler.disable
def attention(qkv_list, recycle_q=False):
    """sageattn_varlen of q, k and v (batch, tokens, heads, head_dim, batch items one after another in memory) handed off in qkv_list,
    every batch item with all its tokens. k and v are not modified (others may hold them); the output is written into q with
    recycle_q."""
    q, k, v = qkv_list
    qkv_list.clear()
    quantized = [None, _quantize_k(k, False), staged_quantize_v([v], SETTINGS)]
    k = v = None
    quantized[0] = staged_quantize_q(q, SETTINGS)
    return staged_attention(q if recycle_q else torch.empty_like(q, memory_format=torch.contiguous_format), quantized, SETTINGS)
