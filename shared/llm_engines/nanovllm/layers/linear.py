import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math

try:
    from shared.kernels import quanto_int8_triton as _shared_quanto_int8_triton
except Exception:  # pragma: no cover
    _shared_quanto_int8_triton = None  # type: ignore


def _shared_kernel_available() -> bool:
    if _shared_quanto_int8_triton is None:
        return False
    is_available = getattr(_shared_quanto_int8_triton, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


_TRITON_AVAILABLE = _shared_kernel_available()


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _get_tp_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _flatten_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale.view(-1)
    if scale.ndim == 1:
        return scale
    return scale.reshape(-1)


def _run_triton_fused_int8_mm(
    x2d: torch.Tensor,
    qweight_t: torch.Tensor,
    qweight_scale_fp32: torch.Tensor,
    input_scale: float | None = None,
) -> torch.Tensor:
    del input_scale
    if _shared_quanto_int8_triton is None:
        raise RuntimeError("shared.kernels.quanto_int8_triton is unavailable")
    run_kernel = getattr(_shared_quanto_int8_triton, "fused_quant_scaled_mm_transposed", None)
    if run_kernel is None:
        raise RuntimeError("shared.kernels.quanto_int8_triton.fused_quant_scaled_mm_transposed is missing")

    m, k = x2d.shape
    k2, n = qweight_t.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: x={x2d.shape}, w_t={qweight_t.shape}")
    qweight_scale_fp32 = _flatten_scale(qweight_scale_fp32)
    if qweight_scale_fp32.numel() != n:
        raise RuntimeError(
            f"Triton int8 qweight_scale length mismatch: expected {n}, got {qweight_scale_fp32.numel()}"
        )
    return run_kernel(x2d, qweight_t, qweight_scale_fp32, out_dtype=x2d.dtype)


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank, self.tp_size = _get_tp_info()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        self.register_buffer("qweight_data", torch.empty(0, dtype=torch.int8))
        self.register_buffer("qweight_t", torch.empty(0, dtype=torch.int8))
        self.register_buffer("qweight_scale", torch.empty(0))
        self.register_buffer("qweight_scale_fp32", torch.empty(0, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones((), dtype=torch.bfloat16))
        self.register_buffer("output_scale", torch.ones((), dtype=torch.bfloat16))
        self.use_int8_weight = False
        self.use_triton_int8 = False
        self._input_scale_value = 1.0
        self._quant_expected_shards = 1
        self._quant_data_loaded = set()
        self._quant_scale_loaded = set()
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _shard_weight(self, loaded_weight: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        return loaded_weight

    def _shard_weight_scale(self, loaded_scale: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        return loaded_scale

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        param.data.copy_(self._shard_weight(loaded_weight, loaded_shard_id))

    def quant_weight_data_loader(self, loaded_weight: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device
        shard = self._shard_weight(loaded_weight, loaded_shard_id).contiguous()
        q = shard.to(device=device, dtype=torch.int8, non_blocking=True)
        self.qweight_data = q
        shard_key = loaded_shard_id if loaded_shard_id is not None else 0
        self._quant_data_loaded.add(shard_key)

    def quant_weight_scale_loader(self, loaded_scale: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device if self.weight.numel() != 0 else self.qweight_data.device
        shard_scale = self._shard_weight_scale(loaded_scale, loaded_shard_id).contiguous()
        if shard_scale.dim() == 2 and shard_scale.size(1) == 1:
            shard_scale = shard_scale.squeeze(1)
        self.qweight_scale = shard_scale.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        shard_key = loaded_shard_id if loaded_shard_id is not None else 0
        self._quant_scale_loaded.add(shard_key)

    def quant_input_scale_loader(self, loaded_scale: torch.Tensor):
        device = self.weight.device if self.weight.numel() != 0 else self.qweight_data.device
        self.input_scale = loaded_scale.to(device=device, dtype=torch.bfloat16, non_blocking=True)

    def quant_output_scale_loader(self, loaded_scale: torch.Tensor):
        device = self.weight.device if self.weight.numel() != 0 else self.qweight_data.device
        self.output_scale = loaded_scale.to(device=device, dtype=torch.bfloat16, non_blocking=True)

    def finalize_quantized(self):
        if self.qweight_data.numel() == 0 or self.qweight_scale.numel() == 0:
            return
        if len(self._quant_data_loaded) < self._quant_expected_shards:
            return
        if len(self._quant_scale_loaded) < self._quant_expected_shards:
            return
        # Keep int8 weights in KxN layout for int8 GEMM paths.
        self.qweight_t = self.qweight_data.transpose(0, 1).contiguous()
        self.qweight_data = torch.empty(0, dtype=torch.int8, device=self.qweight_t.device)
        self.qweight_scale_fp32 = self.qweight_scale.to(dtype=torch.float32)
        if not torch.is_tensor(self.input_scale):
            raise RuntimeError(f"Invalid input_scale type: {type(self.input_scale)}")
        input_scale_flat = self.input_scale.reshape(-1)
        if input_scale_flat.numel() != 1:
            raise RuntimeError(
                f"Expected scalar input_scale, got shape={tuple(self.input_scale.shape)}"
            )
        input_scale_value = float(input_scale_flat[0].item())
        if not math.isfinite(input_scale_value) or input_scale_value <= 0:
            raise RuntimeError(f"Invalid input_scale value: {input_scale_value}")
        self._input_scale_value = max(input_scale_value, 1e-8)
        self.use_triton_int8 = bool(_TRITON_AVAILABLE and self.qweight_t.is_cuda)
        self.use_int8_weight = True
        device = self.qweight_t.device
        if self.weight.numel() != 0:
            self.weight = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.data.to(device=device), requires_grad=False)

    def _quant_int8_mm(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x2d = x.reshape(-1, x_shape[-1])
        if self.use_triton_int8 and x2d.is_cuda and self.qweight_t.numel() != 0 and self.qweight_scale_fp32.numel() != 0:
            y = _run_triton_fused_int8_mm(
                x2d,
                self.qweight_t,
                self.qweight_scale_fp32,
            )
            return y.view(*x_shape[:-1], y.size(-1))

        # Fallback path: use Quanto qbytes_mm for dynamic activation quantization parity.
        if not hasattr(torch.ops, "quanto") or not hasattr(torch.ops.quanto, "qbytes_mm"):
            raise RuntimeError("quanto.qbytes_mm op unavailable for int8 fallback path")
        if self.qweight_t.numel() == 0 and self.qweight_data.numel() != 0:
            self.qweight_t = self.qweight_data.transpose(0, 1).contiguous()
        qweight = self.qweight_t.transpose(0, 1).contiguous()
        scales = self.qweight_scale_fp32
        if scales.numel() == 0:
            scales = self.qweight_scale.to(torch.float32)
        scales = _flatten_scale(scales).reshape(-1, 1).contiguous()
        out = torch.ops.quanto.qbytes_mm(x2d, qweight, scales).to(x2d.dtype)
        return out.view(*x_shape[:-1], out.size(-1))

    def prepare_for_quantized_load(self):
        device = None
        if self.qweight_t.numel() != 0:
            device = self.qweight_t.device
        elif self.qweight_data.numel() != 0:
            device = self.qweight_data.device
        elif self.weight.numel() != 0:
            device = self.weight.device
        elif self.qweight_scale.numel() != 0:
            device = self.qweight_scale.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.weight.numel() != 0:
            self.weight = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

        # Reset all quantized buffers/state so each reload is complete and deterministic.
        self.qweight_data = torch.empty(0, dtype=torch.int8, device=device)
        self.qweight_t = torch.empty(0, dtype=torch.int8, device=device)
        self.qweight_scale = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.qweight_scale_fp32 = torch.empty(0, dtype=torch.float32, device=device)
        self.input_scale = torch.ones((), dtype=torch.bfloat16, device=device)
        self.output_scale = torch.ones((), dtype=torch.bfloat16, device=device)
        self.use_int8_weight = False
        self.use_triton_int8 = False
        self._input_scale_value = 1.0
        self._quant_data_loaded.clear()
        self._quant_scale_loaded.clear()


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_int8_weight:
            y = self._quant_int8_mm(x)
            if self.bias is not None:
                y = y + self.bias
            return y
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = _get_tp_info()[1]
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def _shard_weight(self, loaded_weight: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        shard_size = divide(loaded_weight.size(self.tp_dim), self.tp_size)
        start_idx = self.tp_rank * shard_size
        return loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

    def _shard_weight_scale(self, loaded_scale: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        if loaded_scale.dim() == 0:
            return loaded_scale
        shard_size = divide(loaded_scale.size(0), self.tp_size)
        start_idx = self.tp_rank * shard_size
        return loaded_scale.narrow(0, start_idx, shard_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_int8_weight:
            y = self._quant_int8_mm(x)
            if self.bias is not None:
                y = y + self.bias
            return y
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
        self._quant_expected_shards = len(output_sizes)

    def _merged_shard_meta(self, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        return shard_offset, shard_size

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset, shard_size = self._merged_shard_meta(loaded_shard_id)
        param_slice = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(shard)

    def quant_weight_data_loader(self, loaded_weight: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device
        if self.qweight_data.numel() == 0:
            self.qweight_data = torch.empty((self.output_size, self.input_size), dtype=torch.int8, device=device)
        shard_offset, shard_size = self._merged_shard_meta(int(loaded_shard_id))
        param_slice = self.qweight_data.narrow(0, shard_offset, shard_size)
        shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(shard.to(device=device, dtype=torch.int8, non_blocking=True))
        self._quant_data_loaded.add(int(loaded_shard_id))

    def quant_weight_scale_loader(self, loaded_scale: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device if self.weight.numel() != 0 else self.qweight_data.device
        if self.qweight_scale.numel() == 0:
            self.qweight_scale = torch.empty((self.output_size,), dtype=torch.bfloat16, device=device)
        shard_offset, shard_size = self._merged_shard_meta(int(loaded_shard_id))
        scale_slice = self.qweight_scale.narrow(0, shard_offset, shard_size)
        shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
        if shard.dim() == 2 and shard.size(1) == 1:
            shard = shard.squeeze(1)
        scale_slice.copy_(shard.to(device=device, dtype=torch.bfloat16, non_blocking=True))
        self._quant_scale_loaded.add(int(loaded_shard_id))


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = _get_tp_info()[1]
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)
        self._quant_expected_shards = 3

    def _qkv_offset_size(self, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        return shard_offset, shard_size

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        shard_offset, shard_size = self._qkv_offset_size(loaded_shard_id)
        param_slice = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(shard)

    def quant_weight_data_loader(self, loaded_weight: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device
        if self.qweight_data.numel() == 0:
            self.qweight_data = torch.empty((self.output_size, self.input_size), dtype=torch.int8, device=device)
        shard_offset, shard_size = self._qkv_offset_size(str(loaded_shard_id))
        param_slice = self.qweight_data.narrow(0, shard_offset, shard_size)
        shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(shard.to(device=device, dtype=torch.int8, non_blocking=True))
        self._quant_data_loaded.add(str(loaded_shard_id))

    def quant_weight_scale_loader(self, loaded_scale: torch.Tensor, loaded_shard_id=None):
        device = self.weight.device if self.weight.numel() != 0 else self.qweight_data.device
        if self.qweight_scale.numel() == 0:
            self.qweight_scale = torch.empty((self.output_size,), dtype=torch.bfloat16, device=device)
        shard_offset, shard_size = self._qkv_offset_size(str(loaded_shard_id))
        scale_slice = self.qweight_scale.narrow(0, shard_offset, shard_size)
        shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
        if shard.dim() == 2 and shard.size(1) == 1:
            shard = shard.squeeze(1)
        scale_slice.copy_(shard.to(device=device, dtype=torch.bfloat16, non_blocking=True))
        self._quant_scale_loaded.add(str(loaded_shard_id))


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = _get_tp_info()[1]
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def _shard_weight(self, loaded_weight: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        shard_size = divide(loaded_weight.size(self.tp_dim), self.tp_size)
        start_idx = self.tp_rank * shard_size
        return loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

    def _shard_weight_scale(self, loaded_scale: torch.Tensor, loaded_shard_id=None) -> torch.Tensor:
        # Output rows are not sharded in RowParallelLinear.
        return loaded_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_int8_weight:
            y = self._quant_int8_mm(x)
            if self.bias is not None and self.tp_rank == 0:
                y = y + self.bias
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
