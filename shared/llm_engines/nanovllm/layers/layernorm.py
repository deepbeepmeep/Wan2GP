from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_USE_TRITON_RMSNORM = triton is not None and tl is not None


def configure_layernorm_safe_legacy_kernels(enabled: bool) -> None:
    global _USE_TRITON_RMSNORM
    _USE_TRITON_RMSNORM = (not enabled) and triton is not None and tl is not None


if triton is not None:
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        residual_out_ptr,
        x_row_stride,
        residual_row_stride,
        y_row_stride,
        residual_out_row_stride,
        n_cols,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        STORE_RESIDUAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols

        x = tl.load(x_ptr + row * x_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(
                residual_ptr + row * residual_row_stride + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x = x + residual
            if STORE_RESIDUAL:
                tl.store(residual_out_ptr + row * residual_out_row_stride + offs, x, mask=mask)

        var = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.rsqrt(var + eps)
        weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * weight
        tl.store(y_ptr + row * y_row_stride + offs, y, mask=mask)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.use_triton_rmsnorm = _USE_TRITON_RMSNORM

    def _can_use_triton(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> bool:
        if not self.use_triton_rmsnorm or triton is None or tl is None:
            return False
        if not x.is_cuda or self.weight.device.type != "cuda":
            return False
        if x.stride(-1) != 1 or self.weight.stride(-1) != 1:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if residual is not None:
            if residual.device != x.device or residual.dtype != x.dtype or residual.stride(-1) != 1:
                return False
        return int(x.shape[-1]) <= 8192

    def _fallback_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        return (x_float * torch.rsqrt(var + self.eps)).to(orig_dtype) * self.weight

    def _fallback_add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        summed = x.float() + residual.float()
        residual_out = summed.to(orig_dtype)
        var = summed.pow(2).mean(dim=-1, keepdim=True)
        normed = (summed * torch.rsqrt(var + self.eps)).to(orig_dtype) * self.weight
        return normed, residual_out

    def _triton_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        y_2d = torch.empty_like(x_2d)
        block_size = _next_power_of_2(int(x_2d.shape[1]))
        _rmsnorm_kernel[(x_2d.shape[0],)](
            x_2d,
            x_2d,
            self.weight,
            y_2d,
            y_2d,
            x_2d.stride(0),
            x_2d.stride(0),
            y_2d.stride(0),
            y_2d.stride(0),
            x_2d.shape[1],
            self.eps,
            HAS_RESIDUAL=False,
            STORE_RESIDUAL=False,
            BLOCK_SIZE=block_size,
        )
        return y_2d.view_as(x)

    def _triton_add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_2d = x.reshape(-1, x.shape[-1])
        residual_2d = residual.reshape(-1, residual.shape[-1])
        y_2d = torch.empty_like(x_2d)
        residual_out_2d = torch.empty_like(x_2d)
        block_size = _next_power_of_2(int(x_2d.shape[1]))
        _rmsnorm_kernel[(x_2d.shape[0],)](
            x_2d,
            residual_2d,
            self.weight,
            y_2d,
            residual_out_2d,
            x_2d.stride(0),
            residual_2d.stride(0),
            y_2d.stride(0),
            residual_out_2d.stride(0),
            x_2d.shape[1],
            self.eps,
            HAS_RESIDUAL=True,
            STORE_RESIDUAL=True,
            BLOCK_SIZE=block_size,
        )
        return y_2d.view_as(x), residual_out_2d.view_as(x)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self._can_use_triton(x):
                return self._triton_rms_forward(x)
            return self._fallback_rms_forward(x)

        if self._can_use_triton(x, residual):
            return self._triton_add_rms_forward(x, residual)
        return self._fallback_add_rms_forward(x, residual)
