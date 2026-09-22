"""Qwen GDN decode fusions, with numerical probes on other NVIDIA targets."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from fla.modules.convolution import ShortConvolution, causal_conv1d_update_kernel


class GDNShortConvolution(ShortConvolution):
    def step(self, x, residual, cache, output_final_state=False, cu_seqlens=None):
        # Installed only on the four-tap Qwen convolution. The
        # original FLA kernel's arithmetic and state update remain unchanged.
        batch, _, channels = x.shape
        if output_final_state and cache is None:
            cache = x.new_zeros((batch, channels, 4))
        y = torch.empty_like(x)
        causal_conv1d_update_kernel[(triton.cdiv(channels, 64), batch)](
            x=x, cache=cache, residual=residual, y=y, weight=self.weight,
            bias=self.bias, D=channels, W=4, BD=64, BW=4,
            ACTIVATION=self.activation, num_warps=4,
        )
        return y, cache


@triton.jit
def _prepare_kernel(Q, K, A, B, SSM_A, SSM_DT, Q_OUT, K_OUT, G, BETA,
                    Q_BATCH: tl.constexpr, Q_HEAD: tl.constexpr,
                    A_BATCH: tl.constexpr, B_BATCH: tl.constexpr,
                    HEADS: tl.constexpr, REPEAT: tl.constexpr, DIM: tl.constexpr,
                    BLOCK: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    source = batch * Q_BATCH + (head // REPEAT) * Q_HEAD + cols
    dest = (batch * HEADS + head) * DIM + cols
    q = tl.load(Q + source, cols < DIM, 0)
    k = tl.load(K + source, cols < DIM, 0)
    tl.store(Q_OUT + dest, q, cols < DIM)
    tl.store(K_OUT + dest, k, cols < DIM)
    a = tl.load(A + batch * A_BATCH + head).to(tl.float32)
    b = tl.load(B + batch * B_BATCH + head).to(tl.float32)
    dt = a + tl.load(SSM_DT + head).to(tl.float32)
    softplus = tl.where(dt > 20., dt, libdevice.log1p(tl.exp(dt)))
    g = tl.load(SSM_A + head).to(tl.float32) * softplus
    tl.store(G + batch * HEADS + head, g)
    tl.store(BETA + batch * HEADS + head, tl.sigmoid(b))


def prepare_decode(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    batch, _, num_k_heads, dim = query.shape
    q = query.new_empty((batch, 1, num_v_heads, dim))
    k = torch.empty_like(q)
    g = torch.empty((batch, 1, num_v_heads), device=query.device, dtype=torch.float32)
    beta = b.new_empty((batch, 1, num_v_heads))
    _prepare_kernel[(num_v_heads, batch)](
        query, key, a, b, ssm_a, ssm_dt, q, k, g, beta,
        query.stride(0), query.stride(2), a.stride(0), b.stride(0),
        num_v_heads, num_v_heads // num_k_heads, dim,
        triton.next_power_of_2(dim), num_warps=4,
    )
    return q, k, g, beta


def install_gdn_decode(model):
    """Install after checkpoint layout configuration; weights stay MMGP-owned."""
    if torch.version.hip is not None:
        return
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        return
    validated_launches = capability == (12, 0)
    count = 0
    for block in model.blk:
        # The 27B hidden-width reduction benefits from more threads for a
        # single decode row. Other widths/batches retain defaults.
        for norm in (block.attn_norm, block.post_attention_norm):
            if validated_launches and norm.weight.numel() == 5120:
                norm._small_batch_num_warps = 16
        if block.layer_type != "linear_attention":
            continue
        if (block._fast_recurrent_gated_delta_rule is not None
                and block._use_short_convolution
                and block.ssm_conv1d.backend == "triton"
                and block.conv_kernel_size == 4):
            if validated_launches:
                block.ssm_conv1d.__class__ = GDNShortConvolution
            block._gdn_prepare_decode = prepare_decode if validated_launches else prepare_decode_checked
            block._gdn_recurrent_raw = recurrent_raw_gates if validated_launches else recurrent_raw_gates_checked
            count += 1
    if count:
        mode = "Validated SM120 Launches" if validated_launches else "Numerical Checks Before Capture"
        print(f"[Qwen][GDN] Decode Fusions Available For {count} Layers ({mode}).")


# Adapted from the FLA-derived recurrent_verify_kernel in
# nanovllm/layers/speculative_state.py; its MIT notice applies.
from fla.ops.utils.op import exp


@triton.jit
def _recurrent_raw_kernel(Q, K, V, A, BETA_IN, SSM_A, DT, STATE, OUT, SNAPSHOTS,
                          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                          DK: tl.constexpr, DV: tl.constexpr,
                          QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
                          KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                          VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
                          AS0: tl.constexpr, AS1: tl.constexpr,
                          BS0: tl.constexpr, BS1: tl.constexpr,
                          BATCH: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                          SAVE_PREFIX: tl.constexpr,
                          V_HEADS_TILED: tl.constexpr, SSM_PARAMS_TILED: tl.constexpr,
                          INTERLEAVE_AB: tl.constexpr):
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    tiled_head = (head % (HV // H)) * H + kh
    interleaved_head = head // 2 + (head % 2) * (HV // 2)
    value_head = tiled_head if V_HEADS_TILED else head
    gate_head = tiled_head if V_HEADS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    param_head = tiled_head if SSM_PARAMS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(STATE + state_index, mask, 0).to(tl.float32)
    sa = tl.load(SSM_A + param_head).to(tl.float32)
    dt = tl.load(DT + param_head).to(tl.float32)
    for token in range(T):
        q = tl.load(Q + batch * QS0 + token * QS1 + kh * QS2 + keys, keys < DK, 0).to(tl.float32)
        k = tl.load(K + batch * KS0 + token * KS1 + kh * KS2 + keys, keys < DK, 0).to(tl.float32)
        v = tl.load(V + batch * VS0 + token * VS1 + value_head * VS2 + values, values < DV, 0).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= DK ** -0.5
        a = tl.load(A + batch * AS0 + token * AS1 + gate_head).to(tl.float32) + dt
        softplus = tl.where(a > 20., a, libdevice.log1p(tl.exp(a)))
        g = sa * softplus
        raw_beta = tl.load(BETA_IN + batch * BS0 + token * BS1 + gate_head)
        # Preserve the checkpoint compute dtype's sigmoid rounding before
        # FP32 recurrence, matching the existing materialized beta tensor.
        beta = tl.sigmoid(raw_beta.to(tl.float32)).to(BETA_IN.dtype.element_ty).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        out = tl.sum(state * q[:, None], 0)
        tl.store(OUT + ((batch * T + token) * HV + value_head) * DV + values, out, values < DV)
        # Keep the optional pointer behind a constexpr-only branch: older
        # Triton versions type-check both sides of a mixed static/dynamic test.
        if SAVE_PREFIX:
            if token + 1 < T:
                tl.store(SNAPSHOTS + token * BATCH * HV * DK * DV + state_index, state, mask)
    tl.store(STATE + state_index, state, mask)


def recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None,
                        *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    """Read projection layouts directly; keep state and snapshots in grouped order."""
    batch, tokens, heads, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    _recurrent_raw_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        q, k, v, a, b, ssm_a, ssm_dt, initial, output, snapshots,
        tokens, heads, hv, dim_k, dim_v,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *a.stride()[:2], *b.stride()[:2], batch,
        triton.next_power_of_2(dim_k), 8, snapshots is not None,
        v_heads_tiled, ssm_params_tiled, interleave_ab,
        num_warps=1, num_stages=3,
    )
    return output, initial


_portable_choices = {}
_MAX_PROBE_STATE_BYTES = 32 * 1024 * 1024


def _probe_key(kind, tensors, extra=()):
    return (kind, tuple(None if x is None else (x.device, x.dtype, tuple(x.shape), tuple(x.stride()))
                        for x in tensors), extra)


def _is_tracing(tensor):
    from torch._subclasses.fake_tensor import FakeTensor
    return isinstance(tensor, FakeTensor) or torch.compiler.is_compiling()


def _probe_blocked(tensor):
    return _is_tracing(tensor) or torch.cuda.is_current_stream_capturing()


def _record_choice(key, passed, reason='Numerical Check'):
    _portable_choices[key] = passed
    print(f"[Qwen][GDN] {key[0]}: {'Fused Kernel' if passed else 'Existing Kernel'} ({reason}).")


def _prepare_reference(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    repeat = num_v_heads // query.shape[2]
    return (query.repeat_interleave(repeat, 2), key.repeat_interleave(repeat, 2),
            ssm_a.float() * torch.nn.functional.softplus(a.float() + ssm_dt), b.sigmoid())


def prepare_decode_checked(query, key, a, b, ssm_a, ssm_dt, num_v_heads):
    args = (query, key, a, b, ssm_a, ssm_dt, num_v_heads)
    if _is_tracing(query):
        return _prepare_reference(*args)
    signature = _probe_key('Gate Preparation', args[:-1], (num_v_heads,))
    choice = _portable_choices.get(signature)
    if choice is True:
        return prepare_decode(*args)
    expected = _prepare_reference(*args)
    if choice is False or _probe_blocked(query):
        return expected
    # Probe only outside capture. No tensors are retained in the decision cache.
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError
    try:
        actual = prepare_decode(*args)
        for index, (result, reference) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(result, reference, rtol=1e-6 if index == 2 else 0,
                                       atol=1e-7 if index == 2 else 0)
    except AssertionError:
        _record_choice(signature, False)
    except (OutOfResources, CompilationError) as exc:
        _record_choice(signature, False, f'Kernel Unavailable: {exc}')
    else:
        _record_choice(signature, True)
    return expected


def _recurrent_reference(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots,
                         *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    from shared.llm_engines.nanovllm.layers.speculative_state import recurrent_verify
    # Preserve the established materialized path when a portable numerical
    # probe cannot use the direct-layout kernel, including during capture.
    from shared.llm_engines.nanovllm.models.qwen3_5 import (
        _interleave_axis_halves, _reorder_v_head_axis_tiled_to_grouped,
        _reorder_v_head_axis_grouped_to_tiled, _reorder_v_heads_tiled_to_grouped,
    )
    heads, value_heads = q.shape[2], v.shape[2]
    if v_heads_tiled:
        v = _reorder_v_head_axis_tiled_to_grouped(v, 2, heads, value_heads)
        a = _reorder_v_heads_tiled_to_grouped(a, -1, heads, value_heads, 1)
        b = _reorder_v_heads_tiled_to_grouped(b, -1, heads, value_heads, 1)
    elif interleave_ab:
        a, b = (_interleave_axis_halves(x, -1) for x in (a, b))
    if ssm_params_tiled:
        ssm_a, ssm_dt = (_reorder_v_heads_tiled_to_grouped(x, 0, heads, value_heads, 1) for x in (ssm_a, ssm_dt))
    elif interleave_ab:
        ssm_a, ssm_dt = (_interleave_axis_halves(x, 0) for x in (ssm_a, ssm_dt))
    if snapshots is None and q.shape[1] > 1:
        snapshots = initial.new_empty((q.shape[1] - 1, *initial.shape))
    g = ssm_a.float() * torch.nn.functional.softplus(a.float() + ssm_dt)
    output, state = recurrent_verify(q, k, v, g, b.sigmoid(), initial, snapshots)
    if v_heads_tiled:
        output = _reorder_v_head_axis_grouped_to_tiled(output, 2, heads, value_heads)
    return output, state


def recurrent_raw_gates_checked(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None,
                                *, v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    args = (q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots)
    layout = dict(v_heads_tiled=v_heads_tiled, ssm_params_tiled=ssm_params_tiled, interleave_ab=interleave_ab)
    if _is_tracing(q):
        return _recurrent_reference(*args, **layout)
    signature = _probe_key('Recurrence', args, tuple(layout.values()))
    choice = _portable_choices.get(signature)
    if choice is True:
        return recurrent_raw_gates(*args, **layout)
    if choice is False or _probe_blocked(q):
        return _recurrent_reference(*args, **layout)
    prefix = snapshots[:q.shape[1] - 1] if snapshots is not None else None
    scratch_bytes = initial.numel() * initial.element_size()
    if prefix is not None:
        scratch_bytes += prefix.numel() * prefix.element_size()
    if scratch_bytes > _MAX_PROBE_STATE_BYTES:
        _record_choice(signature, False, 'Probe Memory Limit')
        return _recurrent_reference(*args, **layout)
    # Run the candidate on disposable state; only the established reference
    # advances live state during validation, including speculative prefixes.
    probe_state = probe_prefix = None
    try:
        probe_state = initial.clone()
        probe_prefix = torch.empty_like(prefix) if prefix is not None else None
    except torch.OutOfMemoryError:
        probe_state = probe_prefix = None
        _record_choice(signature, False, 'Probe Allocation Limit')
        return _recurrent_reference(*args, **layout)
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError
    try:
        actual, _ = recurrent_raw_gates(*args[:7], probe_state, probe_prefix, **layout)
    except (OutOfResources, CompilationError) as exc:
        _record_choice(signature, False, f'Kernel Unavailable: {exc}')
        return _recurrent_reference(*args, **layout)
    expected, state = _recurrent_reference(*args, **layout)
    pairs = [(actual, expected), (probe_state, state)]
    if prefix is not None:
        pairs.append((probe_prefix, prefix))
    try:
        for result, reference in pairs:
            torch.testing.assert_close(result, reference, atol=2e-5,
                                       rtol=torch.finfo(reference.dtype).eps * 1.1)
    except AssertionError:
        _record_choice(signature, False)
    else:
        _record_choice(signature, True)
    return expected, state


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
