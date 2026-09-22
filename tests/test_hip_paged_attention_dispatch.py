"""CPU-only regression for AMD FA2's unsupported block_table argument."""
import pytest
import torch

from shared.llm_engines.nanovllm.layers import attention
from shared.llm_engines.nanovllm.utils.context import Context


def make_case(mode):
    torch.manual_seed(723)
    lengths = [7] if mode == 'speculative' else [7, 5]
    query_lengths = [3] if mode == 'speculative' else [2, 1] if mode == 'prefill' else [1, 1]
    tables = torch.tensor([[2, 0], [4, 1]][:len(lengths)], dtype=torch.int32, device='cpu')
    slots = []
    for sequence, (total, count) in enumerate(zip(lengths, query_lengths)):
        for token in range(total - count, total):
            slots.append(int(tables[sequence, token // 4]) * 4 + token % 4)
    q = torch.randn(sum(query_lengths), 4, 8, device='cpu')
    k = torch.randn(sum(query_lengths), 2, 8, device='cpu')
    v = torch.randn_like(k)
    layer = attention.Attention(4, 8, 8 ** -.5, 2)
    layer.use_triton_kv_cache = False
    layer.k_cache = torch.randn(5, 4, 2, 8, device='cpu')
    layer.v_cache = torch.randn_like(layer.k_cache)
    context = Context(
        is_prefill=mode == 'prefill', speculative_verify=mode == 'speculative',
        block_tables=tables, slot_mapping=torch.tensor(slots, dtype=torch.int64, device='cpu'),
        context_lens=torch.tensor(lengths, dtype=torch.int32, device='cpu'),
        cu_seqlens_q=torch.tensor([0] + query_lengths, dtype=torch.int32, device='cpu').cumsum(0).int(),
        cu_seqlens_k=torch.tensor([0] + lengths, dtype=torch.int32, device='cpu').cumsum(0).int(),
        max_seqlen_q=max(query_lengths), max_seqlen_k=max(lengths),
    )
    return layer, context, q, k, v, lengths, query_lengths


def reference(layer, context, q, lengths, query_lengths):
    outputs, offset = [], 0
    for sequence, (total, count) in enumerate(zip(lengths, query_lengths)):
        # Independently reconstruct logical token order from physical pages.
        keys = torch.stack([layer.k_cache[int(context.block_tables[sequence, t // 4]), t % 4] for t in range(total)])
        values = torch.stack([layer.v_cache[int(context.block_tables[sequence, t // 4]), t % 4] for t in range(total)])
        keys, values = (x.repeat_interleave(2, dim=1).transpose(0, 1) for x in (keys, values))
        queries = q[offset:offset + count].transpose(0, 1)
        scores = queries @ keys.transpose(-1, -2) * layer.scale
        positions = torch.arange(total, device='cpu')
        causal = positions[None, :] <= (total - count + torch.arange(count, device='cpu'))[:, None]
        scores.masked_fill_(~causal, -float('inf'))
        outputs.append((scores.softmax(-1) @ values).transpose(0, 1))
        offset += count
    return torch.cat(outputs)


@pytest.mark.parametrize('mode', ['prefill', 'decode', 'speculative'])
def test_hip_paged_attention_bypasses_unsupported_flash_and_preserves_prefix(monkeypatch, mode):
    layer, context, q, k, v, lengths, query_lengths = make_case(mode)
    monkeypatch.setattr(torch.version, 'hip', '7.13')
    monkeypatch.setattr(attention, 'get_context', lambda: context)
    def unsupported(*args, **kwargs):
        raise NotImplementedError('block_table / paged attention is not supported in AMD Triton FA2 varlen_fwd.')
    layer.flash_attn_varlen_func = layer.flash_attn_with_kvcache = unsupported
    for _ in range(2):
        actual = layer(q, k, v)
        if mode == 'decode':
            actual = actual.squeeze(1)
        torch.testing.assert_close(actual, reference(layer, context, q, lengths, query_lengths), rtol=1e-5, atol=1e-6)
        q.add_(.1)
        v.add_(.2)


@pytest.mark.parametrize('mode', ['prefill', 'decode', 'speculative'])
def test_nvidia_paged_flash_dispatch_is_unchanged(monkeypatch, mode):
    layer, context, q, k, v, _, _ = make_case(mode)
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(attention, 'get_context', lambda: context)
    calls = []
    def flash(query, *args, **kwargs):
        calls.append(kwargs['block_table'])
        return torch.full_like(query, 17.)
    layer.flash_attn_varlen_func = layer.flash_attn_with_kvcache = flash
    assert torch.all(layer(q, k, v) == 17)
    assert len(calls) == 1 and calls[0] is context.block_tables


def test_hip_nonpaged_prefill_still_uses_flash(monkeypatch):
    layer, context, q, k, v, _, _ = make_case('prefill')
    context.block_tables = None
    context.cu_seqlens_k = context.cu_seqlens_q
    context.max_seqlen_k = context.max_seqlen_q
    monkeypatch.setattr(torch.version, 'hip', '7.13')
    monkeypatch.setattr(attention, 'get_context', lambda: context)
    def flash(query, key, value, **kwargs):
        assert kwargs['block_table'] is None
        assert key is k and value is v
        return torch.full_like(query, 23.)
    layer.flash_attn_varlen_func = flash
    assert torch.all(layer(q, k, v) == 23)
