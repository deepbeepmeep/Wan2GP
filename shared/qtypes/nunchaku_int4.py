import ast
import os
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import torch

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes


HANDLER_NAME = "nunchaku_int4"

_NUNCHAKU_INT4_QTYPE_NAME = "nunchaku_int4"
if _NUNCHAKU_INT4_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_NUNCHAKU_INT4_QTYPE_NAME] = _quanto_qtype(
        _NUNCHAKU_INT4_QTYPE_NAME,
        is_floating_point=False,
        bits=4,
        dtype=torch.int8,
        qmin=-8.0,
        qmax=7.0,
    )

_NUNCHAKU_INT4_QTYPE = _quanto_qtypes[_NUNCHAKU_INT4_QTYPE_NAME]
_NUNCHAKU_OPS = None
_NUNCHAKU_FALLBACK_NOTICE = False
_NUNCHAKU_SPLIT_FIELDS = {
    "qweight": 0,
    "wscales": 1,
    "wzeros": 1,
    "wcscales": 0,
    "proj_up": 0,
    "bias": 0,
}
_NUNCHAKU_SHARED_FIELDS = (
    "proj_down",
    "smooth_factor",
    "smooth_factor_orig",
    "input_scale",
    "output_scale",
    "wtscale",
)


def get_nunchaku_split_kwargs():
    return {
        "split_fields": dict(_NUNCHAKU_SPLIT_FIELDS),
        "share_fields": _NUNCHAKU_SHARED_FIELDS,
        "split_handlers": {
            "wscales": split_packed_wscales,
            "wzeros": split_packed_wscales,
            "wcscales": split_packed_scale_vector,
        },
    }


def make_nunchaku_splitter(split_map):
    def _split(state_dict, verboseLevel=1):
        from mmgp import offload
        split_kwargs = get_nunchaku_split_kwargs()
        return offload.sd_split_linear(
            state_dict,
            split_map,
            verboseLevel=verboseLevel,
            **split_kwargs,
        )
    return _split


def _install_nunchaku_shim(candidate_root):
    candidate_pkg = candidate_root / "nunchaku"
    if not candidate_pkg.exists():
        return False
    if "nunchaku" in sys.modules:
        del sys.modules["nunchaku"]
    shim = types.ModuleType("nunchaku")
    shim.__path__ = [str(candidate_pkg)]
    sys.modules["nunchaku"] = shim
    return True


def _load_nunchaku_ops():
    global _NUNCHAKU_OPS
    if os.environ.get("WAN2GP_FORCE_NUNCHAKU_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on"):
        _NUNCHAKU_OPS = False
        _notify_nunchaku_fallback("Forced fallback (WAN2GP_FORCE_NUNCHAKU_FALLBACK=1).")
        return _NUNCHAKU_OPS

    if _NUNCHAKU_OPS is not None:
        return _NUNCHAKU_OPS
    _NUNCHAKU_OPS = False

    try:
        from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
        from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
        from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "kernels" / "nunchaku"
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        _install_nunchaku_shim(candidate)
        try:
            from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
            from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
            from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
        except Exception:
            _NUNCHAKU_OPS = False
            _notify_nunchaku_fallback("Nunchaku kernels unavailable; using Python fallback.")
            return _NUNCHAKU_OPS

    _NUNCHAKU_OPS = SimpleNamespace(
        svdq_gemm_w4a4_cuda=svdq_gemm_w4a4_cuda,
        svdq_quantize_w4a4_act_fuse_lora_cuda=svdq_quantize_w4a4_act_fuse_lora_cuda,
        awq_gemv_w4a16_cuda=awq_gemv_w4a16_cuda,
    )
    return _NUNCHAKU_OPS


def _as_dtype_name(dtype_str):
    if dtype_str.startswith("torch."):
        return dtype_str.split(".", 1)[1]
    return dtype_str


def _notify_nunchaku_fallback(reason):
    global _NUNCHAKU_FALLBACK_NOTICE
    if _NUNCHAKU_FALLBACK_NOTICE:
        return
    print(f"[nunchaku_int4] {reason}")
    _NUNCHAKU_FALLBACK_NOTICE = True


def _is_float8_dtype(dtype):
    return "float8" in str(dtype).lower() or "f8" in str(dtype).lower()


def _expand_group_scales(scales, group_size):
    scales_t = scales.transpose(0, 1)
    return scales_t.repeat_interleave(group_size, dim=1)


def _unpack_nunchaku_wscales(wscales, out_features, in_features, group_size):
    if wscales is None or wscales.ndim != 2:
        return wscales
    if in_features % group_size != 0:
        return wscales
    groups = in_features // group_size
    if wscales.shape != (groups, out_features):
        return wscales
    warp_n = 128
    num_lanes = 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if out_features % warp_s != 0:
        return wscales
    packed = wscales.view(
        out_features // warp_s,
        groups,
        num_s_packs,
        num_s_lanes // 4,
        4,
        s_pack_size // 2,
        2,
    )
    unpacked = packed.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(out_features, groups).transpose(0, 1).contiguous()


def _pack_nunchaku_wscales(wscales, out_features, in_features, group_size):
    if wscales is None or wscales.ndim != 2:
        return wscales
    if in_features % group_size != 0:
        return wscales
    groups = in_features // group_size
    if wscales.shape != (groups, out_features):
        return wscales
    warp_n = 128
    num_lanes = 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if out_features % warp_s != 0:
        return wscales
    unpacked = wscales.transpose(0, 1).contiguous()
    unpacked = unpacked.view(
        out_features // warp_s,
        num_s_packs,
        num_s_lanes // 4,
        s_pack_size // 2,
        4,
        2,
        groups,
    )
    packed = unpacked.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return packed.view(groups, out_features)


def split_packed_wscales(src, *, dim, split_sizes, context):
    if src is None or dim != 1:
        return None
    ctx = context or {}
    field_tensors = ctx.get("field_tensors") or {}
    qweight = field_tensors.get("qweight")
    if not torch.is_tensor(qweight) or qweight.dtype not in (torch.int8, torch.int32):
        return None
    info = ctx.get("info") or {}
    if qweight.dtype == torch.int8:
        out_features = qweight.size(0)
    else:
        out_features = qweight.size(0) * 4
    in_features = qweight.size(1) * 2
    group_size = info.get("group_size", ctx.get("group_size", None))
    if not isinstance(group_size, int) or group_size <= 0:
        if torch.is_tensor(src) and src.ndim == 2 and src.shape[0] > 0 and in_features % src.shape[0] == 0:
            group_size = in_features // src.shape[0]
        else:
            group_size = 64
    total = sum(split_sizes)
    unpacked = _unpack_nunchaku_wscales(src, out_features, in_features, group_size)
    if not torch.is_tensor(unpacked):
        return None
    if unpacked.shape != (in_features // group_size, total):
        return None
    unpacked_chunks = torch.split(unpacked, split_sizes, dim=1)
    return [
        _pack_nunchaku_wscales(chunk, size, in_features, group_size)
        for chunk, size in zip(unpacked_chunks, split_sizes)
    ]


def _unpack_nunchaku_scale_vector(scale, size):
    if scale is None or scale.ndim != 1 or scale.numel() != size:
        return scale
    warp_n = 128
    num_lanes = 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if size % warp_s != 0:
        return scale
    packed = scale.reshape(size // warp_s, 1, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2)
    unpacked = packed.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(size)


def _pack_nunchaku_scale_vector(scale, size):
    if scale is None or scale.ndim != 1 or scale.numel() != size:
        return scale
    warp_n = 128
    num_lanes = 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if size % warp_s != 0:
        return scale
    unpacked = scale.reshape(
        size // warp_s,
        num_s_packs,
        num_s_lanes // 4,
        s_pack_size // 2,
        4,
        2,
        1,
    )
    packed = unpacked.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return packed.view(size)


def split_packed_scale_vector(src, *, dim, split_sizes, context):
    if src is None or dim != 0:
        return None
    total = sum(split_sizes)
    unpacked = _unpack_nunchaku_scale_vector(src, total)
    if not torch.is_tensor(unpacked) or unpacked.numel() != total:
        return None
    chunks = torch.split(unpacked, split_sizes, dim=0)
    if unpacked is src:
        return list(chunks)
    return [_pack_nunchaku_scale_vector(chunk, size) for chunk, size in zip(chunks, split_sizes)]


def _unpack_int4_from_int8(qweight):
    q = qweight.to(torch.uint8)
    low = q & 0x0F
    high = (q >> 4) & 0x0F
    low = low.to(torch.int16)
    high = high.to(torch.int16)
    low -= (low >= 8).to(torch.int16) * 16
    high -= (high >= 8).to(torch.int16) * 16
    stacked = torch.stack((low, high), dim=-1)
    out = stacked.reshape(qweight.shape[0], qweight.shape[1] * 2)
    return out


def _unpack_nunchaku_w4a4_weight(qweight, out_features, in_features):
    if qweight.dtype != torch.int8:
        return _unpack_int4_from_int8(qweight)
    if qweight.numel() != out_features * in_features // 2:
        return _unpack_int4_from_int8(qweight)
    mem_n = 128
    mem_k = 64
    num_k_unrolls = 2
    if out_features % mem_n != 0 or in_features % (mem_k * num_k_unrolls) != 0:
        return _unpack_int4_from_int8(qweight)

    n_tiles = out_features // mem_n
    k_tiles = in_features // mem_k
    packed_i32 = qweight.view(torch.int32)
    packed_i32 = packed_i32.view(n_tiles, k_tiles, 1, 8, 8, 4, 2, 2, 1)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=packed_i32.device, dtype=torch.int32)
    vals = (packed_i32.unsqueeze(-1) >> shifts) & 0xF
    vals = vals.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    vals = vals.view(out_features, in_features).to(torch.int16)
    vals -= (vals >= 8).to(torch.int16) * 16
    return vals


def _unpack_int4_from_int32(qweight, out_features, in_features):
    if (
        qweight.dtype == torch.int32
        and out_features % 4 == 0
        and in_features % 64 == 0
        and qweight.shape[0] * 4 == out_features
        and qweight.shape[1] * 2 == in_features
    ):
        packed_i16 = qweight.view(torch.int16)
        packed = packed_i16.view(out_features // 4, in_features // 64, 4, 16)
        packed = packed.permute(0, 2, 1, 3).contiguous().view(-1, 8)
        packed = packed.to(torch.int32) & 0xFFFF
        vals0 = packed & 0xF
        vals1 = (packed >> 4) & 0xF
        vals2 = (packed >> 8) & 0xF
        vals3 = (packed >> 12) & 0xF
        vals = torch.stack((vals0, vals1, vals2, vals3), dim=1)
        return vals.reshape(out_features, in_features)

    q = qweight.view(torch.int32).reshape(out_features, in_features // 8)
    q = q.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64)
    mask = torch.tensor(0x0F, device=q.device, dtype=torch.int64)
    vals = (q.unsqueeze(-1) >> shifts) & mask
    return vals.reshape(out_features, in_features)


def _unpack_lowrank_weight(weight, down):
    if weight is None or weight.ndim != 2:
        return weight
    c, r = weight.shape
    reg_n = 1
    reg_k = 2
    n_pack_size = 2
    k_pack_size = 2
    num_n_lanes = 8
    num_k_lanes = 4
    pack_n = n_pack_size * num_n_lanes * reg_n
    pack_k = k_pack_size * num_k_lanes * reg_k
    if down:
        if r % pack_n != 0 or c % pack_k != 0:
            return weight
        r_packs, c_packs = r // pack_n, c // pack_k
    else:
        if c % pack_n != 0 or r % pack_k != 0:
            return weight
        c_packs, r_packs = c // pack_n, r // pack_k
    weight = weight.view(
        c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k
    )
    weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
    weight = weight.view(c_packs, r_packs, pack_n, pack_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def _nunchaku_qfallback(callable, *args, **kwargs):
    args, kwargs = torch.utils._pytree.tree_map_only(
        NunchakuBaseWeightTensor, lambda x: x.dequantize(), (args, kwargs or {})
    )
    return callable(*args, **kwargs)


class NunchakuBaseWeightTensor(QTensor):
    def __init__(self, qtype, axis):
        super().__init__(qtype, axis)

    def get_quantized_subtensors(self):
        raise NotImplementedError

    def set_quantized_subtensors(self, sub_tensors):
        raise NotImplementedError


class NunchakuSVDQWeightTensor(NunchakuBaseWeightTensor):
    @staticmethod
    def create(
        qweight,
        wscales,
        smooth_factor,
        proj_down,
        proj_up,
        size,
        stride,
        dtype,
        device=None,
        requires_grad=False,
    ):
        device = qweight.device if device is None else device
        if qweight.device != device:
            qweight = qweight.to(device)
        if wscales.device != device:
            wscales = wscales.to(device)
        if smooth_factor.device != device:
            smooth_factor = smooth_factor.to(device)
        if proj_down.device != device:
            proj_down = proj_down.to(device)
        if proj_up.device != device:
            proj_up = proj_up.to(device)
        if dtype in (torch.float16, torch.bfloat16):
            if wscales.dtype in (torch.float16, torch.bfloat16) and wscales.dtype != dtype:
                wscales = wscales.to(dtype)
            if smooth_factor.dtype in (torch.float16, torch.bfloat16) and smooth_factor.dtype != dtype:
                smooth_factor = smooth_factor.to(dtype)
            if proj_down.dtype in (torch.float16, torch.bfloat16) and proj_down.dtype != dtype:
                proj_down = proj_down.to(dtype)
            if proj_up.dtype in (torch.float16, torch.bfloat16) and proj_up.dtype != dtype:
                proj_up = proj_up.to(dtype)
        return NunchakuSVDQWeightTensor(
            qtype=_NUNCHAKU_INT4_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            qweight=qweight,
            wscales=wscales,
            smooth_factor=smooth_factor,
            proj_down=proj_down,
            proj_up=proj_up,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(
        cls,
        qtype,
        axis,
        size,
        stride,
        qweight,
        wscales,
        smooth_factor,
        proj_down,
        proj_up,
        dtype,
        requires_grad=False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=qweight.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        qtype,
        axis,
        size,
        stride,
        qweight,
        wscales,
        smooth_factor,
        proj_down,
        proj_up,
        dtype,
        requires_grad=False,
    ):
        super().__init__(qtype, axis)
        self._qweight = qweight
        self._wscales = wscales
        self._smooth_factor = smooth_factor
        self._proj_down = proj_down
        self._proj_up = proj_up
        self._group_size = 64
        self._act_unsigned = False

    def dequantize(self, dtype=None, device=None):
        ref = getattr(self, "_nunchaku_transpose_ref", None)
        if ref is not None:
            return ref.dequantize(dtype=dtype, device=device).t()
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        qweight = self._qweight.to(device)
        wscales = self._wscales.to(device)
        proj_down = self._proj_down.to(device)
        proj_up = self._proj_up.to(device)

        out_features, in_features = self.shape
        qvals = _unpack_nunchaku_w4a4_weight(qweight, out_features, in_features).to(dtype)
        wscales = _unpack_nunchaku_wscales(wscales, self.shape[0], self.shape[1], self._group_size)
        scales = _expand_group_scales(wscales, self._group_size).to(dtype)
        weight = qvals * scales
        smooth = _unpack_nunchaku_scale_vector(self._smooth_factor, in_features)
        if torch.is_tensor(smooth):
            smooth = smooth.to(device=device, dtype=dtype)
            weight = weight / smooth
        proj_down = _unpack_lowrank_weight(proj_down, down=True)
        proj_up = _unpack_lowrank_weight(proj_up, down=False)
        weight = weight + proj_up.to(dtype) @ proj_down.to(dtype)
        return weight

    def _linear_fallback(self, input, bias=None):
        x = input.reshape(-1, input.shape[-1])
        out_features, in_features = self.shape
        qweight = self._qweight.to(device=x.device)
        wscales = self._wscales.to(device=x.device)
        proj_down = self._proj_down.to(device=x.device)
        proj_up = self._proj_up.to(device=x.device)

        qvals = _unpack_nunchaku_w4a4_weight(qweight, out_features, in_features).to(x.dtype)
        wscales = _unpack_nunchaku_wscales(wscales, out_features, in_features, self._group_size)
        scales = _expand_group_scales(wscales, self._group_size).to(x.dtype)
        base_weight = qvals * scales

        proj_down = _unpack_lowrank_weight(proj_down, down=True).to(x.dtype)
        proj_up = _unpack_lowrank_weight(proj_up, down=False).to(x.dtype)

        smooth = _unpack_nunchaku_scale_vector(self._smooth_factor, in_features)
        if torch.is_tensor(smooth):
            smooth = smooth.to(device=x.device, dtype=x.dtype)
            x_base = x / smooth
        else:
            x_base = x

        out = x_base.matmul(base_weight.t())
        lora_act = x.matmul(proj_down.t())
        out.add_(lora_act.matmul(proj_up.t()))
        if bias is not None:
            out.add_(bias)
        return out.reshape(*input.shape[:-1], out_features)

    def _linear_cuda(self, input, bias=None):
        ops = _load_nunchaku_ops()
        if not ops:
            return self._linear_fallback(input, bias=bias)

        x = input.reshape(-1, input.shape[-1])
        qx, ascales, lora_act = ops.svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self._proj_down,
            smooth=self._smooth_factor,
            fp4=False,
            pad_size=256,
        )
        out = torch.empty(qx.shape[0], self.shape[0], dtype=self._proj_up.dtype, device=qx.device)
        if bias is not None and bias.dtype != out.dtype:
            bias = bias.to(out.dtype)
        ops.svdq_gemm_w4a4_cuda(
            act=qx,
            wgt=self._qweight,
            out=out,
            ascales=ascales,
            wscales=self._wscales,
            lora_act_in=lora_act,
            lora_up=self._proj_up,
            bias=bias,
            fp4=False,
            alpha=1.0,
            wcscales=None,
            act_unsigned=self._act_unsigned,
        )
        out = out[: x.shape[0]]
        return out.reshape(*input.shape[:-1], self.shape[0])

    def linear(self, input, bias=None):
        if torch.is_tensor(input) and input.device.type == "cuda":
            return self._linear_cuda(input, bias=bias)
        return self._linear_fallback(input, bias=bias)

    def get_quantized_subtensors(self):
        return [
            ("qweight", self._qweight),
            ("wscales", self._wscales),
            ("smooth_factor", self._smooth_factor),
            ("proj_down", self._proj_down),
            ("proj_up", self._proj_up),
        ]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        if "qweight" in sub_map and sub_map["qweight"] is not None:
            self._qweight = sub_map["qweight"]
        if "wscales" in sub_map and sub_map["wscales"] is not None:
            self._wscales = sub_map["wscales"]
        if "smooth_factor" in sub_map and sub_map["smooth_factor"] is not None:
            self._smooth_factor = sub_map["smooth_factor"]
        if "proj_down" in sub_map and sub_map["proj_down"] is not None:
            self._proj_down = sub_map["proj_down"]
        if "proj_up" in sub_map and sub_map["proj_up"] is not None:
            self._proj_up = sub_map["proj_up"]

    def __tensor_flatten__(self):
        inner_tensors = ["_qweight", "_wscales", "_smooth_factor", "_proj_down", "_proj_up"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "group_size": str(self._group_size),
            "kind": "svdq",
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_name = _as_dtype_name(meta.get("dtype", "torch.float16"))
        dtype = getattr(torch, dtype_name, torch.float16)
        return NunchakuSVDQWeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            qweight=inner_tensors["_qweight"],
            wscales=inner_tensors["_wscales"],
            smooth_factor=inner_tensors["_smooth_factor"],
            proj_down=inner_tensors["_proj_down"],
            proj_up=inner_tensors["_proj_up"],
            dtype=dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        kwargs = kwargs or {}
        if op in _VIEW_OPS:
            t = args[0]
            shape = _view_shape_from_args(args, t.numel())
            if shape is not None and shape == tuple(t.shape):
                return _wrap_nunchaku_weight(t, size=shape, stride=t.stride())
        if op is torch.ops.aten.t:
            t = args[0]
            if isinstance(t, NunchakuSVDQWeightTensor):
                return _transpose_nunchaku_weight(t)
        if op is torch.ops.aten.transpose:
            t, dim0, dim1 = args[:3]
            if isinstance(t, NunchakuSVDQWeightTensor):
                if (dim0, dim1) in ((0, 1), (1, 0), (-2, -1), (-1, -2)):
                    return _transpose_nunchaku_weight(t)
        if op is torch.ops.aten.addmm:
            input, mat1, mat2 = args[:3]
            if isinstance(mat2, NunchakuSVDQWeightTensor):
                ref = getattr(mat2, "_nunchaku_transpose_ref", None)
                if ref is not None:
                    alpha = kwargs.get("alpha", 1)
                    beta = kwargs.get("beta", 1)
                    out = ref.linear(mat1, bias=None)
                    if alpha != 1:
                        out.mul_(alpha)
                    if beta != 0 and input is not None:
                        if input.dtype != out.dtype:
                            input = input.to(out.dtype)
                        out.add_(input, alpha=beta)
                    return _apply_mod_params_if_needed(ref, out)
        if op is torch.ops.aten.mm or op is torch.ops.aten.matmul:
            mat1, mat2 = args[:2]
            if isinstance(mat2, NunchakuSVDQWeightTensor):
                ref = getattr(mat2, "_nunchaku_transpose_ref", None)
                if ref is not None:
                    out = ref.linear(mat1, bias=None)
                    return _apply_mod_params_if_needed(ref, out)
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, NunchakuSVDQWeightTensor):
                out = weight.linear(input, bias=bias)
                return _apply_mod_params_if_needed(weight, out)
        if op is torch.ops.aten.detach:
            t = args[0]
            out = NunchakuSVDQWeightTensor.create(
                qweight=op(t._qweight),
                wscales=op(t._wscales),
                smooth_factor=op(t._smooth_factor),
                proj_down=op(t._proj_down),
                proj_up=op(t._proj_up),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
            )
            return _copy_mod_flags(t, out)
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_qweight = op(t._qweight, device=device, **(kwargs or {}))
            out_wscales = op(t._wscales, device=device, **(kwargs or {}))
            out_smooth = op(t._smooth_factor, device=device, **(kwargs or {}))
            out_proj_down = op(t._proj_down, device=device, **(kwargs or {}))
            out_proj_up = op(t._proj_up, device=device, **(kwargs or {}))
            out = NunchakuSVDQWeightTensor.create(
                qweight=out_qweight,
                wscales=out_wscales,
                smooth_factor=out_smooth,
                proj_down=out_proj_down,
                proj_up=out_proj_up,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
            )
            return _copy_mod_flags(t, out)
        return _nunchaku_qfallback(op, *args, **(kwargs or {}))


class NunchakuAWQWeightTensor(NunchakuBaseWeightTensor):
    @staticmethod
    def create(
        qweight,
        wscales,
        wzeros,
        size,
        stride,
        dtype,
        device=None,
        requires_grad=False,
    ):
        device = qweight.device if device is None else device
        if qweight.device != device:
            qweight = qweight.to(device)
        if wscales.device != device:
            wscales = wscales.to(device)
        if wzeros.device != device:
            wzeros = wzeros.to(device)
        if dtype in (torch.float16, torch.bfloat16):
            if wscales.dtype in (torch.float16, torch.bfloat16) and wscales.dtype != dtype:
                wscales = wscales.to(dtype)
            if wzeros.dtype in (torch.float16, torch.bfloat16) and wzeros.dtype != dtype:
                wzeros = wzeros.to(dtype)
        return NunchakuAWQWeightTensor(
            qtype=_NUNCHAKU_INT4_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            qweight=qweight,
            wscales=wscales,
            wzeros=wzeros,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(
        cls,
        qtype,
        axis,
        size,
        stride,
        qweight,
        wscales,
        wzeros,
        dtype,
        requires_grad=False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=qweight.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        qtype,
        axis,
        size,
        stride,
        qweight,
        wscales,
        wzeros,
        dtype,
        requires_grad=False,
    ):
        super().__init__(qtype, axis)
        self._qweight = qweight
        self._wscales = wscales
        self._wzeros = wzeros
        self._group_size = 64

    def dequantize(self, dtype=None, device=None):
        ref = getattr(self, "_nunchaku_transpose_ref", None)
        if ref is not None:
            return ref.dequantize(dtype=dtype, device=device).t()
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        out_features, in_features = self.shape
        qweight = self._qweight.to(device)
        wscales = self._wscales.to(device)
        wzeros = self._wzeros.to(device)
        qvals = _unpack_int4_from_int32(qweight, out_features, in_features).to(dtype)
        scales = _expand_group_scales(wscales, self._group_size).to(dtype)
        zeros = _expand_group_scales(wzeros, self._group_size).to(dtype)
        return qvals * scales + zeros

    def _linear_fallback(self, input, bias=None):
        x = input.reshape(-1, input.shape[-1])
        weight = self.dequantize(dtype=x.dtype, device=x.device)
        out = x.matmul(weight.t())
        if bias is not None:
            out.add_(bias)
        return out.reshape(*input.shape[:-1], weight.shape[0])

    def _linear_cuda(self, input, bias=None):
        ops = _load_nunchaku_ops()
        if not ops:
            return self._linear_fallback(input, bias=bias)

        x = input.reshape(-1, input.shape[-1])
        out = ops.awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self._qweight,
            scaling_factors=self._wscales,
            zeros=self._wzeros,
            m=x.shape[0],
            n=self.shape[0],
            k=x.shape[1],
            group_size=self._group_size,
        )
        if bias is not None:
            view_shape = [1] * (out.ndim - 1) + [-1]
            out.add_(bias.view(view_shape))
        return out.reshape(*input.shape[:-1], out.shape[-1])

    def linear(self, input, bias=None):
        if torch.is_tensor(input) and input.device.type == "cuda":
            return self._linear_cuda(input, bias=bias)
        return self._linear_fallback(input, bias=bias)

    def get_quantized_subtensors(self):
        return [
            ("qweight", self._qweight),
            ("wscales", self._wscales),
            ("wzeros", self._wzeros),
        ]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        if "qweight" in sub_map and sub_map["qweight"] is not None:
            self._qweight = sub_map["qweight"]
        if "wscales" in sub_map and sub_map["wscales"] is not None:
            self._wscales = sub_map["wscales"]
        if "wzeros" in sub_map and sub_map["wzeros"] is not None:
            self._wzeros = sub_map["wzeros"]

    def __tensor_flatten__(self):
        inner_tensors = ["_qweight", "_wscales", "_wzeros"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "group_size": str(self._group_size),
            "kind": "awq",
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_name = _as_dtype_name(meta.get("dtype", "torch.float16"))
        dtype = getattr(torch, dtype_name, torch.float16)
        return NunchakuAWQWeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            qweight=inner_tensors["_qweight"],
            wscales=inner_tensors["_wscales"],
            wzeros=inner_tensors["_wzeros"],
            dtype=dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        kwargs = kwargs or {}
        if op in _VIEW_OPS:
            t = args[0]
            shape = _view_shape_from_args(args, t.numel())
            if shape is not None and shape == tuple(t.shape):
                return _wrap_nunchaku_weight(t, size=shape, stride=t.stride())
        if op is torch.ops.aten.t:
            t = args[0]
            if isinstance(t, NunchakuAWQWeightTensor):
                return _transpose_nunchaku_weight(t)
        if op is torch.ops.aten.transpose:
            t, dim0, dim1 = args[:3]
            if isinstance(t, NunchakuAWQWeightTensor):
                if (dim0, dim1) in ((0, 1), (1, 0), (-2, -1), (-1, -2)):
                    return _transpose_nunchaku_weight(t)
        if op is torch.ops.aten.addmm:
            input, mat1, mat2 = args[:3]
            if isinstance(mat2, NunchakuAWQWeightTensor):
                ref = getattr(mat2, "_nunchaku_transpose_ref", None)
                if ref is not None:
                    alpha = kwargs.get("alpha", 1)
                    beta = kwargs.get("beta", 1)
                    out = ref.linear(mat1, bias=None)
                    if alpha != 1:
                        out.mul_(alpha)
                    if beta != 0 and input is not None:
                        if input.dtype != out.dtype:
                            input = input.to(out.dtype)
                        out.add_(input, alpha=beta)
                    return _apply_mod_params_if_needed(ref, out)
        if op is torch.ops.aten.mm or op is torch.ops.aten.matmul:
            mat1, mat2 = args[:2]
            if isinstance(mat2, NunchakuAWQWeightTensor):
                ref = getattr(mat2, "_nunchaku_transpose_ref", None)
                if ref is not None:
                    out = ref.linear(mat1, bias=None)
                    return _apply_mod_params_if_needed(ref, out)
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, NunchakuAWQWeightTensor):
                out = weight.linear(input, bias=bias)
                return _apply_mod_params_if_needed(weight, out)
        if op is torch.ops.aten.detach:
            t = args[0]
            out = NunchakuAWQWeightTensor.create(
                qweight=op(t._qweight),
                wscales=op(t._wscales),
                wzeros=op(t._wzeros),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
            )
            return _copy_mod_flags(t, out)
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_qweight = op(t._qweight, device=device, **(kwargs or {}))
            out_wscales = op(t._wscales, device=device, **(kwargs or {}))
            out_wzeros = op(t._wzeros, device=device, **(kwargs or {}))
            out = NunchakuAWQWeightTensor.create(
                qweight=out_qweight,
                wscales=out_wscales,
                wzeros=out_wzeros,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
            )
            return _copy_mod_flags(t, out)
        return _nunchaku_qfallback(op, *args, **(kwargs or {}))


class QLinearNunchakuInt4(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls,
        module,
        weights,
        activations=None,
        optimizer=None,
        device=None,
    ):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._nunchaku_default_dtype = dtype

    def set_default_dtype(self, dtype):
        self._nunchaku_default_dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if isinstance(qweight, NunchakuBaseWeightTensor):
            out = qweight.linear(input, bias=self.bias)
        else:
            out = torch.nn.functional.linear(input, qweight, bias=self.bias)
        if getattr(self, "_nunchaku_mod_reorder", False):
            out = _reorder_mod_params_output(out)
        if getattr(self, "_nunchaku_mod_scale_shift", False):
            out = _apply_mod_scale_shift(out)
        return out

    @property
    def qweight(self):
        if self.weight_qtype == _NUNCHAKU_INT4_QTYPE:
            return self.weight
        return super().qweight

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _NUNCHAKU_INT4_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        _load_nunchaku_ops()
                
        qweight_key = prefix + "qweight"
        wscales_key = prefix + "wscales"
        wzeros_key = prefix + "wzeros"
        smooth_key = prefix + "smooth_factor"
        smooth_orig_key = prefix + "smooth_factor_orig"
        proj_down_key = prefix + "proj_down"
        proj_up_key = prefix + "proj_up"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        qweight = state_dict.pop(qweight_key, None)
        wscales = state_dict.pop(wscales_key, None)
        wzeros = state_dict.pop(wzeros_key, None)
        smooth_factor = state_dict.pop(smooth_key, None)
        state_dict.pop(smooth_orig_key, None)
        proj_down = state_dict.pop(proj_down_key, None)
        proj_up = state_dict.pop(proj_up_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if qweight is None:
            missing_keys.append(qweight_key)
        if wscales is None:
            missing_keys.append(wscales_key)

        target_dtype = self._nunchaku_default_dtype or self.weight.dtype
        if qweight is not None and wscales is not None:
            if qweight.dtype == torch.int8:
                if smooth_factor is None:
                    missing_keys.append(smooth_key)
                if proj_down is None:
                    missing_keys.append(proj_down_key)
                if proj_up is None:
                    missing_keys.append(proj_up_key)
                if smooth_factor is not None and proj_down is not None and proj_up is not None:
                    nunchaku_weight = NunchakuSVDQWeightTensor.create(
                        qweight=qweight,
                        wscales=wscales,
                        smooth_factor=smooth_factor,
                        proj_down=proj_down,
                        proj_up=proj_up,
                        size=self.weight.size(),
                        stride=self.weight.stride(),
                        dtype=target_dtype,
                        device=qweight.device,
                        requires_grad=False,
                    )
                    self.weight = torch.nn.Parameter(nunchaku_weight, requires_grad=False)
                    if prefix.endswith("img_mlp.net.2.") or prefix.endswith("txt_mlp.net.2."):
                        self.weight._act_unsigned = True
            elif qweight.dtype == torch.int32:
                if wzeros is None:
                    missing_keys.append(wzeros_key)
                if wzeros is not None:
                    nunchaku_weight = NunchakuAWQWeightTensor.create(
                        qweight=qweight,
                        wscales=wscales,
                        wzeros=wzeros,
                        size=self.weight.size(),
                        stride=self.weight.stride(),
                        dtype=target_dtype,
                        device=qweight.device,
                        requires_grad=False,
                    )
                    self.weight = torch.nn.Parameter(nunchaku_weight, requires_grad=False)
                    if prefix.endswith("img_mod.1.") or prefix.endswith("txt_mod.1."):
                        self._nunchaku_mod_reorder = True
                        self._nunchaku_mod_scale_shift = True
                        for target in (nunchaku_weight, self.weight):
                            try:
                                target._nunchaku_mod_reorder = True
                                target._nunchaku_mod_scale_shift = True
                            except Exception:
                                pass

        if bias is not None:
            if target_dtype is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
            self.bias = torch.nn.Parameter(bias)

        if torch.is_tensor(qweight):
            scale_device = qweight.device
        elif torch.is_tensor(self.weight):
            scale_device = self.weight.device
        elif torch.is_tensor(bias):
            scale_device = bias.device
        else:
            scale_device = torch.device("cpu")

        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        else:
            if not hasattr(self, "input_scale") or self.input_scale.is_meta:
                scale_dtype = self.input_scale.dtype if hasattr(self, "input_scale") else torch.float32
                self.input_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        else:
            if not hasattr(self, "output_scale") or self.output_scale.is_meta:
                scale_dtype = self.output_scale.dtype if hasattr(self, "output_scale") else torch.float32
                self.output_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        return


def _collect_nunchaku_specs(state_dict):
    for key, tensor in state_dict.items():
        if key.endswith(".wscales") and _is_float8_dtype(tensor.dtype):
            return []
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".qweight"):
            continue
        base = key[:-8]
        wscales = state_dict.get(base + ".wscales", None)
        if wscales is None:
            continue
        if _is_float8_dtype(wscales.dtype):
            continue
        if tensor.dtype == torch.int8:
            if (
                base + ".smooth_factor" in state_dict
                and base + ".proj_down" in state_dict
                and base + ".proj_up" in state_dict
            ):
                specs.append({"name": base, "kind": "svdq"})
        elif tensor.dtype == torch.int32:
            if base + ".wzeros" in state_dict:
                specs.append({"name": base, "kind": "awq"})
    return specs


def _patch_mod_bias_scale_shift(state_dict, verboseLevel=1):
    patched = 0
    for key, bias in state_dict.items():
        if not (key.endswith(".img_mod.1.bias") or key.endswith(".txt_mod.1.bias")):
            continue
        base = key[:-5]
        qweight = state_dict.get(base + ".qweight", None)
        if qweight is None or qweight.dtype != torch.int32:
            continue
        if bias.numel() % 6 != 0:
            continue
        dim = bias.numel() // 6
        if dim == 0:
            continue
        one = torch.ones(dim, device=bias.device, dtype=bias.dtype)
        bias[dim : 2 * dim].sub_(one)
        bias[4 * dim : 5 * dim].sub_(one)
        patched += 1
    if patched and verboseLevel >= 1:
        print(f"[nunchaku_int4] Patched {patched} mod bias scale shifts")


def _reorder_mod_params_output(out):
    if out.shape[-1] % 6 != 0:
        return out
    dim = out.shape[-1] // 6
    return out.reshape(*out.shape[:-1], dim, 6).transpose(-2, -1).reshape(*out.shape[:-1], out.shape[-1])


def _apply_mod_scale_shift(out):
    if out.shape[-1] % 6 != 0:
        return out
    dim = out.shape[-1] // 6
    out[..., dim : 2 * dim].sub_(1.0)
    out[..., 4 * dim : 5 * dim].sub_(1.0)
    return out


def _copy_mod_flags(src, dst):
    if getattr(src, "_nunchaku_mod_reorder", False):
        dst._nunchaku_mod_reorder = True
    if getattr(src, "_nunchaku_mod_scale_shift", False):
        dst._nunchaku_mod_scale_shift = True
    if getattr(src, "_act_unsigned", False):
        dst._act_unsigned = True
    return dst


def _apply_mod_params_if_needed(weight, out):
    weight = getattr(weight, "_nunchaku_transpose_ref", weight)
    if getattr(weight, "_nunchaku_mod_reorder", False):
        out = _reorder_mod_params_output(out)
    if getattr(weight, "_nunchaku_mod_scale_shift", False):
        out = _apply_mod_scale_shift(out)
    return out


_VIEW_OPS = (
    torch.ops.aten.view,
    torch.ops.aten.reshape,
    torch.ops.aten._unsafe_view,
    torch.ops.aten._reshape_alias,
)
_NO_REF = object()


def _view_shape_from_args(args, numel):
    if len(args) < 2:
        return None
    shape = args[1] if len(args) == 2 and isinstance(args[1], (tuple, list, torch.Size)) else args[1:]
    shape = tuple(int(s) for s in shape)
    if -1 not in shape:
        return shape
    known = 1
    neg = None
    for i, dim in enumerate(shape):
        if dim == -1:
            if neg is not None:
                return shape
            neg = i
        else:
            known *= dim
    if neg is None or known == 0:
        return shape
    shape = list(shape)
    shape[neg] = numel // known
    return tuple(shape)


def _wrap_nunchaku_weight(weight, size=None, stride=None, transpose_ref=_NO_REF):
    size = tuple(size) if size is not None else weight.size()
    stride = tuple(stride) if stride is not None else weight.stride()
    if isinstance(weight, NunchakuSVDQWeightTensor):
        out = NunchakuSVDQWeightTensor.create(
            qweight=weight._qweight,
            wscales=weight._wscales,
            smooth_factor=weight._smooth_factor,
            proj_down=weight._proj_down,
            proj_up=weight._proj_up,
            size=size,
            stride=stride,
            dtype=weight.dtype,
            device=weight.device,
            requires_grad=weight.requires_grad,
        )
    elif isinstance(weight, NunchakuAWQWeightTensor):
        out = NunchakuAWQWeightTensor.create(
            qweight=weight._qweight,
            wscales=weight._wscales,
            wzeros=weight._wzeros,
            size=size,
            stride=stride,
            dtype=weight.dtype,
            device=weight.device,
            requires_grad=weight.requires_grad,
        )
    else:
        return weight
    if transpose_ref is _NO_REF:
        transpose_ref = getattr(weight, "_nunchaku_transpose_ref", None)
    if transpose_ref is not None:
        out._nunchaku_transpose_ref = transpose_ref
    return _copy_mod_flags(weight, out)


def _transpose_nunchaku_weight(weight):
    ref = getattr(weight, "_nunchaku_transpose_ref", None)
    if ref is not None:
        return _wrap_nunchaku_weight(ref, size=ref.size(), stride=ref.stride())
    if weight.ndim != 2:
        return weight
    size = (weight.size(1), weight.size(0))
    stride = (weight.stride(1), weight.stride(0))
    return _wrap_nunchaku_weight(weight, size=size, stride=stride, transpose_ref=weight)


def detect(state_dict, verboseLevel=1):
    specs = _collect_nunchaku_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs][:8]
    return {"matched": True, "kind": "nunchaku_int4", "details": {"count": len(specs), "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_nunchaku_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {spec["name"]: {"weights": "nunchaku_int4", "activations": "none"} for spec in specs}
    return {"state_dict": state_dict, "quant_map": quant_map}


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    return quantization_map or {}, []
