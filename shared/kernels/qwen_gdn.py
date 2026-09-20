"""Shared Qwen GDN decode optimizations for the validated SM120 vLLM path."""
import torch
import triton
import triton.language as tl
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
    softplus = tl.where(dt > 20., dt, tl.extra.cuda.libdevice.log1p(tl.exp(dt)))
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
    if torch.cuda.get_device_capability(0) != (12, 0):
        return
    count = 0
    for block in model.blk:
        # The 27B hidden-width reduction benefits from more threads for a
        # single decode row. Other widths/batches retain defaults.
        for norm in (block.attn_norm, block.post_attention_norm):
            if norm.weight.numel() == 5120:
                norm._small_batch_num_warps = 16
        if block.layer_type != "linear_attention":
            continue
        if (block._fast_recurrent_gated_delta_rule is not None
                and block._use_short_convolution
                and block.ssm_conv1d.backend == "triton"
                and block.conv_kernel_size == 4):
            block.ssm_conv1d.__class__ = GDNShortConvolution
            block._gdn_prepare_decode = prepare_decode
            block._gdn_recurrent_raw = recurrent_raw_gates
            count += 1
    if count:
        print(f"[Qwen][GDN] Fused gates, in-place recurrence and convolution enabled for {count} layers (SM120).")


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
                          SAVE_PREFIX: tl.constexpr):
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(STATE + state_index, mask, 0).to(tl.float32)
    sa = tl.load(SSM_A + head).to(tl.float32)
    dt = tl.load(DT + head).to(tl.float32)
    for token in range(T):
        q = tl.load(Q + batch * QS0 + token * QS1 + kh * QS2 + keys, keys < DK, 0).to(tl.float32)
        k = tl.load(K + batch * KS0 + token * KS1 + kh * KS2 + keys, keys < DK, 0).to(tl.float32)
        v = tl.load(V + batch * VS0 + token * VS1 + head * VS2 + values, values < DV, 0).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= DK ** -0.5
        a = tl.load(A + batch * AS0 + token * AS1 + head).to(tl.float32) + dt
        softplus = tl.where(a > 20., a, tl.extra.cuda.libdevice.log1p(tl.exp(a)))
        g = sa * softplus
        raw_beta = tl.load(BETA_IN + batch * BS0 + token * BS1 + head)
        # Preserve the checkpoint compute dtype's sigmoid rounding before
        # FP32 recurrence, matching the existing materialized beta tensor.
        beta = tl.sigmoid(raw_beta.to(tl.float32)).to(BETA_IN.dtype.element_ty).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        out = tl.sum(state * q[:, None], 0)
        tl.store(OUT + ((batch * T + token) * HV + head) * DV + values, out, values < DV)
        # Keep the optional pointer behind a constexpr-only branch: older
        # Triton versions type-check both sides of a mixed static/dynamic test.
        if SAVE_PREFIX:
            if token + 1 < T:
                tl.store(SNAPSHOTS + token * BATCH * HV * DK * DV + state_index, state, mask)
    tl.store(STATE + state_index, state, mask)


def recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None):
    """Read strided projections directly; retain FP32 recurrence and prefix states."""
    batch, tokens, heads, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    _recurrent_raw_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        q, k, v, a, b, ssm_a, ssm_dt, initial, output, snapshots,
        tokens, heads, hv, dim_k, dim_v,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *a.stride()[:2], *b.stride()[:2], batch,
        triton.next_power_of_2(dim_k), 8, snapshots is not None,
        num_warps=1, num_stages=3,
    )
    return output, initial


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
