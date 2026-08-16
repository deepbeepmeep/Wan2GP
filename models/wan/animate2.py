import math

import torch

from shared.attention import pay_attention
from .modules.posemb_layers import apply_rotary_emb, apply_rotary_emb_single


_attention_backends = {}


def _attention_with_lse(q, k, v):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    backend_key = (q.device, q.dtype, q.shape[-1])
    backend = _attention_backends.get(backend_key)
    if backend is None:
        can_use_cudnn = getattr(torch.backends.cuda, "can_use_cudnn_attention", None)
        backend = "cudnn" if can_use_cudnn is not None and can_use_cudnn(torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)) else "efficient"
        _attention_backends[backend_key] = backend
        print(f"[Wan Animate 2] Exact attention backend: {backend}")
    if backend == "cudnn":
        output, lse, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention.default(q, k, v, None, True, 0.0, False, False, scale=None)
        return output.transpose(1, 2), lse.squeeze(-1)
    output, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention.default(q, k, v, None, True, 0.0, False, scale=None)
    return output.transpose(1, 2), lse[..., :q.shape[2]]


def _generation_attention(qkv_list, ref_kv_list, frames, tokens_per_frame, log_scale):
    q, k, v = qkv_list
    qkv_list.clear()
    ref_k, ref_v = ref_kv_list
    batch, _, heads, head_dim = q.shape
    reference_frames = frames - 1
    if ref_k.shape[0] != batch:
        ref_k = ref_k.expand(batch, -1, -1, -1)
        ref_v = ref_v.expand(batch, -1, -1, -1)

    output, lse = _attention_with_lse(q, k, v)
    if log_scale:
        # Reweight the first generated-frame key partition without a dense additive mask.
        scaled_output, scaled_lse = _attention_with_lse(q, k[:, tokens_per_frame:2 * tokens_per_frame], v[:, tokens_per_frame:2 * tokens_per_frame])
        scale_delta = math.expm1(float(log_scale))
        scaled_ratio = torch.exp(scaled_lse - lse).transpose(1, 2)
        denominator = 1 + scale_delta * scaled_ratio
        output.mul_((1 / denominator).to(output.dtype).unsqueeze(-1))
        output.addcmul_(scaled_output, (scale_delta * scaled_ratio / denominator).to(output.dtype).unsqueeze(-1))
        lse.add_(torch.log(denominator.transpose(1, 2)))
        del scaled_output, scaled_lse, scaled_ratio, denominator

    # Batch all matching generation/reference frame pairs, then merge their softmax partitions by LSE.
    ref_output, ref_lse = _attention_with_lse(q[:, tokens_per_frame:].reshape(batch * reference_frames, tokens_per_frame, heads, head_dim), ref_k.reshape(batch * reference_frames, tokens_per_frame, heads, head_dim), ref_v.reshape(batch * reference_frames, tokens_per_frame, heads, head_dim))
    ref_output = ref_output.reshape(batch, reference_frames * tokens_per_frame, heads, head_dim)
    ref_lse = ref_lse.reshape(batch, reference_frames, heads, tokens_per_frame).permute(0, 1, 3, 2).reshape(batch, reference_frames * tokens_per_frame, heads)
    generation_lse = lse[..., tokens_per_frame:].transpose(1, 2)
    combined_lse = torch.logaddexp(generation_lse, ref_lse)
    output_tail = output[:, tokens_per_frame:]
    output_tail.mul_(torch.exp(generation_lse - combined_lse).to(output.dtype).unsqueeze(-1))
    output_tail.addcmul_(ref_output, torch.exp(ref_lse - combined_lse).to(output.dtype).unsqueeze(-1))
    return output


def _reshape_frames(x, frames):
    return x.reshape(x.shape[0], frames, -1, x.shape[-1])


def _restore_tokens(x):
    return x.reshape(x.shape[0], -1, x.shape[-1])


def _modulated_self_attention_input(block, x, e):
    e = e.to(x)
    dtype = x.dtype
    attention_dtype = block.self_attn.q.weight.dtype
    frames = e.shape[0]
    e = (block.modulation.weight + e).chunk(6, dim=1)
    x_mod = _reshape_frames(block.norm1(x), frames)
    x_mod *= 1 + e[1]
    x_mod += e[0]
    x_mod = _restore_tokens(x_mod).to(attention_dtype)
    return x_mod, e, dtype


def _pre_self_attention(block, x, e, freqs):
    x_mod, e, dtype = _modulated_self_attention_input(block, x, e)

    b, s = x_mod.shape[:2]
    heads, head_dim = block.self_attn.num_heads, block.self_attn.head_dim
    qkv_list = [block.self_attn.q(x_mod), block.self_attn.k(x_mod), block.self_attn.v(x_mod)]
    x_mod = None
    block.self_attn.norm_q(qkv_list[0])
    block.self_attn.norm_k(qkv_list[1])
    qkv_list = [tensor.view(b, s, heads, head_dim) for tensor in qkv_list]
    qk_list = qkv_list[:2]
    qkv_list[0] = qkv_list[1] = None
    qkv_list[:2] = apply_rotary_emb(qk_list, freqs, head_first=False)
    return qkv_list, e, dtype


def _reference_kv(block, x, e, freqs):
    x_mod, _, _ = _modulated_self_attention_input(block, x, e)
    b, s = x_mod.shape[:2]
    heads, head_dim = block.self_attn.num_heads, block.self_attn.head_dim
    ref_k = block.self_attn.k(x_mod)
    ref_v = block.self_attn.v(x_mod).view(b, s, heads, head_dim)
    x_mod = None
    block.self_attn.norm_k(ref_k)
    ref_k = apply_rotary_emb_single([ref_k.view(b, s, heads, head_dim)], freqs, head_first=False)
    return [ref_k, ref_v]


def _post_attention_block(block, x_handoff, attention_handoff, e, context, grid_sizes, dtype):
    x = x_handoff[0]
    x_handoff.clear()
    attention_output = attention_handoff[0]
    attention_handoff.clear()
    y = block.self_attn.o(attention_output.flatten(2)).to(dtype)
    attention_output = None
    frames = e[0].shape[0]
    x = _reshape_frames(x, frames)
    y = _reshape_frames(y, frames)
    x.addcmul_(y, e[2])
    x = _restore_tokens(x)

    if context is not None:
        context = context.to(x.device)
        y = block.norm3(x).to(block.cross_attn.q.weight.dtype)
        y_list = [y]
        y = None
        x += block.cross_attn(y_list, context, grid_sizes, audio_proj=None, audio_scale=None, audio_context_lens=None).to(dtype)

    y = _reshape_frames(block.norm2(x), frames)
    y *= 1 + e[4]
    y += e[3]
    y = _restore_tokens(y).to(block.ffn[0].weight.dtype)
    y_shape = y.shape
    y = y.view(-1, y_shape[-1])
    chunk_size = int(y.shape[0] / 2.7)
    for y_chunk in torch.split(y, chunk_size):
        mlp_chunk = block.ffn[1](block.ffn[0](y_chunk))
        y_chunk[...] = block.ffn[2](mlp_chunk)
        mlp_chunk = None
    y = y.view(y_shape).to(dtype)
    x = _reshape_frames(x, frames)
    y = _reshape_frames(y, frames)
    x.addcmul_(y, e[5])
    return _restore_tokens(x)


def animate2_reference_block(block, x_handoff, e, context, grid_sizes, freqs):
    x = x_handoff[0]
    x_handoff.clear()
    qkv_list, e, dtype = _pre_self_attention(block, x, e, freqs)
    ref_kv_list = qkv_list[1:]
    attention_output = pay_attention(qkv_list, recycle_q=True)
    x_handoff, attention_handoff = [x], [attention_output]
    x = attention_output = None
    return _post_attention_block(block, x_handoff, attention_handoff, e, context, grid_sizes, dtype), ref_kv_list


def animate2_generation_block(block, outputs, index, e, context, grid_sizes, freqs, ref_kv_list, ref_grid_sizes, log_scale):
    x = outputs[index]
    outputs[index] = None
    qkv_list, e, dtype = _pre_self_attention(block, x, e, freqs)
    frames, height, width = (int(value) for value in grid_sizes)
    ref_frames, ref_height, ref_width = (int(value) for value in ref_grid_sizes)
    if frames != ref_frames + 1 or height != ref_height or width != ref_width:
        raise RuntimeError(f"Wan Animate 2 requires one generation identity frame followed by one frame per driving latent; got generation {tuple(grid_sizes)} and driving {tuple(ref_grid_sizes)}")

    tokens_per_frame = height * width
    attention_output = _generation_attention(qkv_list, ref_kv_list, frames, tokens_per_frame, log_scale)
    x_handoff, attention_handoff = [x], [attention_output]
    x = attention_output = None
    outputs[index] = _post_attention_block(block, x_handoff, attention_handoff, e, context, grid_sizes, dtype)


def _animate2_generation_blocks(block, ref_kv_list, ref_grid_sizes, generation):
    outputs = generation["x_list"]
    for index, (context, should_calc, unconditional) in enumerate(zip(generation["contexts"], generation["should_calc"], generation["unconditional"])):
        if should_calc and not (unconditional and block.block_no == 9):
            animate2_generation_block(block, outputs, index, generation["e"], context, generation["grid_sizes"], generation["freqs"], ref_kv_list, ref_grid_sizes, generation["log_scale"])
    ref_kv_list.clear()
    return outputs


def animate2_attention_block(block, ref_x_list, ref_e, ref_context, ref_grid_sizes, ref_freqs, generation):
    ref_x, ref_kv_list = animate2_reference_block(block, ref_x_list, ref_e, ref_context, ref_grid_sizes, ref_freqs)
    outputs = _animate2_generation_blocks(block, ref_kv_list, ref_grid_sizes, generation)
    return ref_x, outputs


def animate2_cached_attention_block(block, ref_x_list, ref_e, ref_grid_sizes, ref_freqs, generation):
    ref_x = ref_x_list[0]
    ref_x_list.clear()
    ref_kv_list = _reference_kv(block, ref_x, ref_e, ref_freqs)
    ref_x = None
    return _animate2_generation_blocks(block, ref_kv_list, ref_grid_sizes, generation)
