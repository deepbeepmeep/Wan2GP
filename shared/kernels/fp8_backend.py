# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""FP8 math kernels of FP8 checkpoints (Quanto qfloat8 weights and shared.qtypes.scaled_fp8 weights), chosen in the configuration
(fp8_kernels):
- pytorch: each weight is dequantized whole (copies of its size in 16 bits) before a 16-bit matmul;
- triton: weight-only Triton matmul (fp8_w8a16_triton): FP8 weight tiles decoded in registers, no copy of the weight, same values as
  pytorch up to the order of the sums;
- cuda: FP8 tensor cores (torch._scaled_mm, cuBLASLt, RTX 40xx and later): the activations are also quantized to FP8 (one scale per
  call), fastest, lower precision;
- auto: the fastest available (cuda, then triton, then pytorch), as WanGP's FP8 checkpoints used FP8 tensor cores before.
A choice this GPU cannot run gives way to the next one (cuda -> triton -> pytorch), and the kernels in use are printed."""
import torch
from torch._subclasses.fake_tensor import is_fake

CHOICES = [("Auto (fastest available)", "auto"), ("PyTorch (slowest, precise)", "pytorch"), ("Triton (fast, precise)", "triton"),
           ("CUDA FP8 (fastest, least precise)", "cuda")]
_LABELS = {"pytorch": "PyTorch", "triton": "Triton", "cuda": "CUDA FP8 Tensor Cores"}
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
backend = "pytorch"  # kernels of the Quanto qfloat8 weights
scaled_backend = "pytorch"  # kernels of the scaled FP8 weights (shared.qtypes.scaled_fp8)
revision = 0  # part of the CUDA graph keys: graphs record the kernels of the backend
_ops_registered = False
_cuda_skips_noted = False


def _available(selection):
    """(kernels this GPU runs for a pytorch / triton / cuda selection, reason when they differ)."""
    reason = None
    if selection == "cuda":
        from shared.qtypes import scaled_fp8
        scaled_fp8._init_scaled_mm_support()
        if scaled_fp8._scaled_mm_available(torch.float8_e4m3fn):
            return "cuda", None
        selection, reason = "triton", "FP8 tensor cores need an RTX 40xx or later GPU"
    if selection == "triton":
        try:
            from shared.kernels import fp8_w8a16_triton  # noqa: F401
            return "triton", reason
        except ImportError as exc:
            return "pytorch", f"Triton is unavailable ({exc})"
    return "pytorch", reason


def resolve_backend(selection):
    """((kernels of the Quanto qfloat8 weights, kernels of the scaled FP8 weights), reason when they differ from the selection)."""
    if selection not in {value for _, value in CHOICES}:
        raise ValueError(f"Unknown FP8 kernel backend: {selection!r}")
    resolved, reason = _available("cuda" if selection == "auto" else selection)
    return (resolved, resolved), reason


def configure(selection, resolved=None):
    global backend, scaled_backend, revision
    (quanto, scaled), reason = resolve_backend(selection) if resolved is None else resolved
    if (quanto, scaled) != (backend, scaled_backend):
        revision += 1
    backend, scaled_backend = quanto, scaled
    if (quanto, scaled) != ("pytorch", "pytorch"):
        _register_ops()
    kernels = _LABELS[quanto] if quanto == scaled else f"{_LABELS[quanto]}, {_LABELS[scaled]} for scaled FP8 checkpoints"
    print(f"[FP8] Backend: {kernels} (setting: {selection}{'; ' + reason if reason else ''}).")
    return quanto, scaled


def _cuda_runs(weight, x):
    from shared.qtypes import scaled_fp8
    return scaled_fp8._scaled_mm_available(weight.dtype) and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0 and weight.is_contiguous()


def _cuda_linear(x, weight, scale, bias):
    """FP8 tensor cores: x quantized to FP8 with one scale, the weight scale applied by cuBLASLt (one value) or to the output."""
    from shared.qtypes import scaled_fp8
    scale_b, output_scale = scaled_fp8._scaled_mm_weight_scale(scale.reshape(-1) if scale.numel() > 1 else scale, weight)
    x8, scale_a = scaled_fp8._quantize_activation(x.contiguous(), weight.dtype)
    out = torch._scaled_mm(x8, weight.t(), scale_a, scale_b.to(torch.float32), bias=bias if output_scale is None else None, out_dtype=x.dtype)
    if output_scale is not None:
        out.mul_(output_scale.to(out.dtype).view(1, -1))
        if bias is not None:
            out.add_(bias)
    return out


def _linear_impl(x, weight, scale, bias, cuda):
    global _cuda_skips_noted
    if cuda:
        if _cuda_runs(weight, x):
            return _cuda_linear(x, weight, scale, bias)
        if not _cuda_skips_noted:
            _cuda_skips_noted = True
            print(f"[FP8] CUDA FP8 kernels cannot compute some layers (e.g. {weight.dtype}, {tuple(weight.shape)}): Triton computes them.")
    from shared.kernels import fp8_w8a16_triton
    return fp8_w8a16_triton.linear(x, weight, scale, bias, x.new_empty((x.shape[0], weight.shape[0])))


def _register_ops():
    global _ops_registered
    if _ops_registered:
        return

    @torch.library.custom_op("wan2gp_fp8::linear", mutates_args=())
    def linear_op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, cuda: bool) -> torch.Tensor:
        return _linear_impl(x, weight, scale, bias, cuda)

    @linear_op.register_fake
    def fake(x, weight, scale, bias, cuda):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _ops_registered = True


def matmul(x, weight, scale, bias, cuda):
    """x (M, K) @ (weight (N, K) float8 * scale (1 or N values, in x's dtype))^T + bias with the selected kernels (FP8 tensor cores
    when cuda, else the Triton weight-only kernel), in x's dtype."""
    if torch.compiler.is_compiling() or is_fake(x):  # tracing (Dynamo also runs tensor methods on fake tensors): the opaque op
        return torch.ops.wan2gp_fp8.linear(x, weight, scale, bias, cuda)
    return _linear_impl(x, weight, scale, bias, cuda)


def linear(input, weight, bias=None):
    """input @ weight^T + bias for a Quanto qfloat8 weight with the selected kernels, computed as Quanto does in the dtype of the
    weight's scale."""
    data, scale = weight._data, weight._scale
    x = input.reshape(-1, input.shape[-1])
    if x.dtype != scale.dtype:
        x = x.to(scale.dtype)
    fused_bias = bias if bias is not None and bias.dtype == scale.dtype else None
    out = matmul(x, data, scale, fused_bias, backend == "cuda").reshape(*input.shape[:-1], data.shape[0])
    return out + bias if bias is not None and fused_bias is None else out
