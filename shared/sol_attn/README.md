# Sol-Attn Triton backend

This reusable package vendors the portable Triton implementation from
[`NVlabs/Sana` at commit `46031940`](https://github.com/NVlabs/Sana/tree/46031940ba8af5d18054217e571149579424c0b1/techniques/sparse_backends/sol_attn),
under its Apache-2.0 license.

The `saganaki/` path bundles the optimized INT8-QK implementation from
[`ComfyUI-sol-attn` v0.5.2](https://github.com/Saganaki22/ComfyUI-sol-attn/tree/e2fc225), also
under Apache-2.0. It is used on Ada/RTX 40-series hardware to keep exact routed blocks faster than
Sage2; the original BF16 implementation remains available through the same public interface.

`sol_attn(q, k, v, ...)` expects BF16 tensors in `[batch, tokens, heads, 128]` layout. The head
dimension must be contiguous; the batch, token, and head strides may come directly from a fused
QKV projection.
The optimized Triton backend supports SM89, SM90, SM100, and SM120, covering Ada RTX 40-series
and newer supported architectures. It requires CUDA 12.8 and Triton 3.6 or newer.

The package contains only the model-independent kernels and input contract. Model-specific routing
policy, dense warmup, exact prefix handling, and cache policy belong at each model call site.

`qk_rms_norm_rope_(...)` is the reusable zero-copy preparation path. It applies Q/K RMSNorm and
RoPE in place to either separate projection outputs or strided views of a fused QKV output.

`query_start` skips complete query blocks before the requested token. Their output is intentionally
left uninitialized, so callers using it must replace that prefix with an exact attention result.
