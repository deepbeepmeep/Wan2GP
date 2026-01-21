import ast
import os
import re

import torch
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes


HANDLER_NAME = "gguf"

try:
    import gguf
except Exception:
    gguf = None


_GGUF_QTYPE_NAME = "gguf"
if _GGUF_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_GGUF_QTYPE_NAME] = _quanto_qtype(
        _GGUF_QTYPE_NAME,
        is_floating_point=False,
        bits=6,
        dtype=torch.uint8,
        qmin=-32.0,
        qmax=31.0,
    )
_GGUF_QTYPE = _quanto_qtypes[_GGUF_QTYPE_NAME]

_GGUF_DEFAULT_DTYPE = None
_GGUF_LABEL_CACHE = {}


def _normalize_gguf_path(file_path):
    try:
        return os.path.normcase(os.path.abspath(file_path))
    except Exception:
        return str(file_path).lower()


def normalize(file_path):
    return _normalize_gguf_path(file_path)


def _set_default_dtype_from_loader(dtype):
    global _GGUF_DEFAULT_DTYPE
    if dtype is None:
        return
    _GGUF_DEFAULT_DTYPE = dtype


def _resolve_default_dtype(dtype, fallback=None):
    if dtype is None:
        return _GGUF_DEFAULT_DTYPE or fallback
    if _GGUF_DEFAULT_DTYPE is not None and fallback is not None and dtype == fallback:
        return _GGUF_DEFAULT_DTYPE
    return dtype
def read_gguf_metadata(file_path):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    reader = gguf.GGUFReader(file_path)
    metadata = {}
    field = reader.get_field("config")
    if field is not None:
        try:
            metadata["config"] = field.contents() if callable(getattr(field, "contents", None)) else field.contents
        except Exception:
            pass
    return metadata


def _filter_state_dict_basic(state_dict, base_model_prefix, keep_prefix=False):
    new_state_dict = {}
    start = -1
    if keep_prefix:
        for k, v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_start = len(base_model_prefix)
            else:
                pos = k.find("." + base_model_prefix)
                if pos < 0:
                    continue
                new_start = pos + len(base_model_prefix) + 1
            if start != -1 and start != new_start:
                new_state_dict = state_dict
                break
            start = new_start
            new_state_dict[k[start:]] = v
    return new_state_dict


def _gguf_get_orig_shape(reader, tensor_name):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad GGUF shape metadata for {field_key}: {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def _gguf_resolve_prefix(tensor_names, prefixes):
    for prefix in prefixes:
        if any(name.startswith(prefix) for name in tensor_names):
            return prefix
    return None


def load_gguf_state_dict(
    file_path,
    filters=None,
    keep_prefixes=False,
    writable_tensors=True,
    verboseLevel=1,
    default_dtype=None,
    pin_to_memory=False,
):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    if pin_to_memory:
        raise Exception("Pinning to memory while loading GGUF files is not supported")

    import warnings

    def _cast_plain_tensor(torch_tensor, tensor_type):
        if tensor_type == gguf.GGMLQuantizationType.F16:
            if torch_tensor.dtype in (torch.uint8, torch.uint16):
                torch_tensor = torch_tensor.view(torch.float16)
            elif torch_tensor.dtype != torch.float16:
                torch_tensor = torch_tensor.to(torch.float16)
        elif tensor_type == gguf.GGMLQuantizationType.BF16:
            if torch_tensor.dtype in (torch.uint8, torch.uint16):
                torch_tensor = torch_tensor.view(torch.bfloat16)
            elif torch_tensor.dtype != torch.bfloat16:
                torch_tensor = torch_tensor.to(torch.bfloat16)
        elif tensor_type == gguf.GGMLQuantizationType.F32:
            if torch_tensor.dtype in (torch.uint8, torch.uint16, torch.uint32):
                torch_tensor = torch_tensor.view(torch.float32)
            elif torch_tensor.dtype != torch.float32:
                torch_tensor = torch_tensor.to(torch.float32)
        return torch_tensor

    def _tensor_type_from_dtype(dtype):
        if dtype == torch.float16:
            return gguf.GGMLQuantizationType.F16
        if dtype == torch.bfloat16:
            return gguf.GGMLQuantizationType.BF16
        if dtype == torch.float32:
            return gguf.GGMLQuantizationType.F32
        return None

    reader = gguf.GGUFReader(file_path)
    if verboseLevel >= 2:
        try:
            from mmgp import safetensors2
            safetensors2.verboseLevel = verboseLevel
            tracker = safetensors2.MmapTracker(file_path)
            tracker.register(reader.data, 0, 0, int(reader.data.nbytes))
        except Exception:
            tracker = None
    tensor_names = [tensor.name for tensor in reader.tensors]
    prefix = _gguf_resolve_prefix(tensor_names, ("model.diffusion_model.", "diffusion_model."))

    state_dict = {}
    qtype_counts = {}
    for tensor in reader.tensors:
        name = tensor.name
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = _gguf_get_orig_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            torch_tensor = _cast_plain_tensor(torch_tensor, tensor.tensor_type)
            torch_tensor = torch_tensor.view(*shape)
        wrapped = GGUFSourceTensor.wrap(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if name.endswith(".bias"):
            wrapped._gguf_bias_orig_dtype = wrapped.dtype
            wrapped._gguf_bias_orig_tensor_type = tensor.tensor_type
        state_dict[name] = wrapped
        type_name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1

    if verboseLevel >= 2 and qtype_counts:
        print("GGUF qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_counts.items()))

    if filters is not None:
        if not isinstance(filters, list):
            filters = [filters]
        new_sd = {}
        for one_filter in filters:
            new_sd.update(_filter_state_dict_basic(state_dict, one_filter, keep_prefixes))
        state_dict = new_sd

    if state_dict:
        for name, bias in list(state_dict.items()):
            if not name.endswith(".bias") or not torch.is_tensor(bias):
                continue
            weight_name = name[:-5] + ".weight"
            weight = state_dict.get(weight_name)
            target_dtype = None
            weight_type = getattr(weight, "tensor_type", None) if torch.is_tensor(weight) else None
            if torch.is_tensor(weight) and weight_type is None:
                target_dtype = weight.dtype if weight.dtype.is_floating_point else default_dtype
            elif weight_type in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.F32,
            ):
                target_dtype = weight.dtype
            else:
                target_dtype = default_dtype
            if target_dtype is None or bias.dtype == target_dtype:
                continue
            casted = bias.to(target_dtype)
            if isinstance(casted, GGUFSourceTensor):
                casted._gguf_bias_orig_dtype = getattr(bias, "_gguf_bias_orig_dtype", bias.dtype)
                casted._gguf_bias_orig_tensor_type = getattr(
                    bias, "_gguf_bias_orig_tensor_type", getattr(bias, "tensor_type", None)
                )
                new_tensor_type = _tensor_type_from_dtype(target_dtype)
                if new_tensor_type is not None:
                    casted.tensor_type = new_tensor_type
                casted.tensor_shape = getattr(bias, "tensor_shape", casted.shape)
            state_dict[name] = casted

    return state_dict, None, None


def load_state_dict(*args, **kwargs):
    return load_gguf_state_dict(*args, **kwargs)


class GGUFSourceTensor(torch.Tensor):
    @staticmethod
    def wrap(tensor, *, tensor_type, tensor_shape):
        wrapped = tensor.as_subclass(GGUFSourceTensor)
        wrapped.tensor_type = tensor_type
        wrapped.tensor_shape = tensor_shape
        return wrapped

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.shape)
        return new

    def clone(self, *args, **kwargs):
        cloned = super().clone(*args, **kwargs).as_subclass(GGUFSourceTensor)
        cloned.tensor_type = getattr(self, "tensor_type", None)
        cloned.tensor_shape = getattr(self, "tensor_shape", cloned.shape)
        return cloned

    def detach(self, *args, **kwargs):
        detached = super().detach(*args, **kwargs).as_subclass(GGUFSourceTensor)
        detached.tensor_type = getattr(self, "tensor_type", None)
        detached.tensor_shape = getattr(self, "tensor_shape", detached.shape)
        return detached

    def get_quantized_subtensors(self):
        return [("data", self)]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            data = sub_tensors.get("data")
        else:
            data = dict(sub_tensors).get("data")
        if data is None or data is self:
            return
        torch.utils.swap_tensors(self, data)


def _split_gguf_tensor(src, *, dim, split_sizes, context):
    if not torch.is_tensor(src):
        return None
    tensor_type = getattr(src, "tensor_type", None)
    if tensor_type is None:
        return None
    tensor_shape = getattr(src, "tensor_shape", None) or src.shape
    total = sum(split_sizes)
    if dim >= len(tensor_shape) or tensor_shape[dim] != total:
        return None
    chunks = torch.split(src, split_sizes, dim=dim)
    out = []
    for chunk, size in zip(chunks, split_sizes):
        new_shape = list(tensor_shape)
        new_shape[dim] = size
        wrapped = GGUFSourceTensor.wrap(chunk, tensor_type=tensor_type, tensor_shape=tuple(new_shape))
        if hasattr(src, "_gguf_bias_orig_dtype"):
            wrapped._gguf_bias_orig_dtype = getattr(src, "_gguf_bias_orig_dtype")
        if hasattr(src, "_gguf_bias_orig_tensor_type"):
            wrapped._gguf_bias_orig_tensor_type = getattr(src, "_gguf_bias_orig_tensor_type")
        out.append(wrapped)
    return out


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload
    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields={"weight": 0, "bias": 0},
        split_handlers={"weight": _split_gguf_tensor, "bias": _split_gguf_tensor},
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )


def _is_gguf_qtype(qtype_obj):
    if gguf is None:
        return False
    if qtype_obj is None:
        return False
    return qtype_obj not in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    )


def _gguf_qtype_name(qtype_obj):
    if qtype_obj is None:
        return None
    return getattr(qtype_obj, "name", None) or str(qtype_obj)


def _guess_variant_from_filename(filename):
    base = os.path.basename(str(filename))
    match = re.search(r"(?i)(?:^|[_-])(Q\d+_K|Q\d+_\d|Q\d+|IQ\d+_\w+)(?:$|[_.-])", base)
    if match:
        return match.group(1).upper()
    return None


def detect_gguf_quantization_variant(file_path, verboseLevel=1):
    if gguf is None:
        return None
    try:
        reader = gguf.GGUFReader(file_path)
    except Exception:
        return None
    counts = {}
    for tensor in reader.tensors:
        qtype = getattr(tensor, "tensor_type", None)
        if qtype in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            continue
        name = _gguf_qtype_name(qtype)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_quantization_kind_for_file(file_path, verboseLevel=1):
    if not file_path or str(file_path).lower().endswith(".gguf") is False:
        return None
    if gguf is None:
        return None
    return "gguf"


def detect_quantization_label_from_filename(filename, verboseLevel=1):
    if not filename or str(filename).lower().endswith(".gguf") is False:
        return ""
    key = _normalize_gguf_path(filename)
    cached = _GGUF_LABEL_CACHE.get(key)
    if cached:
        return cached
    variant = _guess_variant_from_filename(filename)
    if not variant and os.path.isfile(filename):
        variant = detect_gguf_quantization_variant(filename, verboseLevel=verboseLevel)
    if variant:
        label = f"GGUF-{variant}"
    else:
        label = "GGUF"
    _GGUF_LABEL_CACHE[key] = label
    return label


def _gguf_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(GGUFWeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


def _reshape_scale(scale, weight):
    if scale.ndim == 0 or scale.numel() == 1:
        return scale
    if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
        return scale.view(weight.shape[0], *([1] * (weight.ndim - 1)))
    return scale


def _gguf_dequantize_tensor(raw, qtype_obj, oshape, dtype=None):
    if gguf is None:
        raise RuntimeError("gguf package is required to dequantize GGUF weights.")
    if qtype_obj in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    ):
        out = raw.view(*oshape)
        return out.to(dtype) if dtype is not None else out
    if qtype_obj not in _DEQUANTIZE_FUNCTIONS:
        out = gguf.quants.dequantize(raw.cpu().numpy(), qtype_obj)
        out = torch.from_numpy(out)
        return out.to(dtype) if dtype is not None else out
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype_obj]
    dequantize_blocks = _DEQUANTIZE_FUNCTIONS[qtype_obj]
    rows = raw.reshape((-1, raw.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def _maybe_cast_bias(bias, target_dtype):
    if bias is None or not torch.is_tensor(bias) or target_dtype is None:
        return bias
    if bias.dtype == target_dtype:
        return bias
    if isinstance(bias, GGUFSourceTensor):
        tensor_type = getattr(bias, "tensor_type", None)
        tensor_shape = getattr(bias, "tensor_shape", bias.shape)
        if _is_gguf_qtype(tensor_type):
            return _gguf_dequantize_tensor(bias, tensor_type, tensor_shape, dtype=target_dtype)
    return bias.to(target_dtype)


def _to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def _const_like(ref, values, dtype):
    device = ref.device if torch.is_tensor(ref) else None
    count = len(values)
    if count == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if count == 1:
        return torch.full((1,), values[0], device=device, dtype=dtype)
    step = values[1] - values[0]
    if all(values[idx] - values[idx - 1] == step for idx in range(1, count)):
        end = values[0] + step * count
        return torch.arange(values[0], end, step, device=device, dtype=dtype)
    raise ValueError("Unsupported constant pattern for GGUF dequantization.")


def _split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return d * qs + m


def _dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape((n_blocks, -1))
    return d * qs + m


def _dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape((n_blocks, -1))
    qs = qs.to(torch.int8) - 8
    return d * qs


QK_K = 256
K_SCALE_SIZE = 12


def _get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def _dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, list(range(8)), torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> _const_like(d, list(range(8)), torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


if gguf is not None:
    _DEQUANTIZE_FUNCTIONS = {
        gguf.GGMLQuantizationType.Q8_0: _dequantize_blocks_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: _dequantize_blocks_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: _dequantize_blocks_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: _dequantize_blocks_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: _dequantize_blocks_Q4_0,
        gguf.GGMLQuantizationType.Q6_K: _dequantize_blocks_Q6_K,
        gguf.GGMLQuantizationType.Q5_K: _dequantize_blocks_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: _dequantize_blocks_Q4_K,
        gguf.GGMLQuantizationType.Q3_K: _dequantize_blocks_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: _dequantize_blocks_Q2_K,
    }
else:
    _DEQUANTIZE_FUNCTIONS = {}


class GGUFWeightTensor(QTensor):
    @staticmethod
    def create(raw_tensor, size, stride, dtype, device=None, requires_grad=False, tensor_type=None, tensor_shape=None):
        if tensor_type is None:
            tensor_type = getattr(raw_tensor, "tensor_type", None)
        if tensor_shape is None:
            tensor_shape = getattr(raw_tensor, "tensor_shape", None) or size
        device = raw_tensor.device if device is None else device
        if raw_tensor.device != device:
            raw_tensor = raw_tensor.to(device)
        return GGUFWeightTensor(
            qtype=_GGUF_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            raw=raw_tensor,
            tensor_type=tensor_type,
            tensor_shape=tensor_shape,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, raw, tensor_type, tensor_shape, dtype, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=raw.device,
            requires_grad=requires_grad,
        )

    def __init__(self, qtype, axis, size, stride, raw, tensor_type, tensor_shape, dtype, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = raw
        self._tensor_type = tensor_type
        self._tensor_shape = torch.Size(tensor_shape)
        self._gguf_default_dtype = dtype

    def __repr__(self):
        cls_name = self.__class__.__name__
        try:
            shape = tuple(self.shape)
        except Exception:
            shape = "<?>"
        try:
            dtype = str(self.dtype).replace("torch.", "")
        except Exception:
            dtype = "<?>"
        try:
            device = str(self.device)
        except Exception:
            device = "<?>"
        qtype = getattr(self, "_qtype", None)
        qtype_name = getattr(qtype, "name", None) or str(qtype) if qtype is not None else "<?>"
        tensor_type = _gguf_qtype_name(getattr(self, "_tensor_type", None)) or "<?>"
        return (
            f"{cls_name}(shape={shape}, dtype={dtype}, device={device}, "
            f"qtype={qtype_name}, tensor_type={tensor_type})"
        )

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        raw = self._data if self._data.device == device else self._data.to(device)
        return _gguf_dequantize_tensor(raw, self._tensor_type, self._tensor_shape, dtype=dtype)

    def linear(self, input, bias=None):
        if torch.is_tensor(input):
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=input.dtype)
            target_device = input.device
        else:
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.dtype)
            target_device = self.device
        weight = self.dequantize(dtype=target_dtype, device=target_device)
        if torch.is_tensor(input) and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        bias = _maybe_cast_bias(bias, weight.dtype)
        return torch.nn.functional.linear(input, weight, bias)

    def get_quantized_subtensors(self):
        return [("data", self._data)]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        data = sub_map.get("data", None)
        if data is not None:
            old_data = self._data
            if torch.is_tensor(old_data):
                try:
                    torch.utils.swap_tensors(old_data, data)
                    self._data = old_data
                except Exception:
                    self._data = data
            else:
                self._data = data
            if hasattr(self, "_ggml_raw_cpu"):
                self._ggml_raw_cpu = None

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "tensor_type": _gguf_qtype_name(self._tensor_type) or "",
            "tensor_shape": str(list(self._tensor_shape)),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_str = meta.get("dtype", "torch.float16")
        if dtype_str.startswith("torch."):
            dtype_name = dtype_str.split(".", 1)[1]
            dtype = getattr(torch, dtype_name, torch.float16)
        else:
            dtype = getattr(torch, dtype_str, torch.float16)
        tensor_shape = ast.literal_eval(meta.get("tensor_shape", str(list(size))))
        tensor_type = None
        if gguf is not None and meta.get("tensor_type"):
            tensor_type = getattr(gguf.GGMLQuantizationType, meta["tensor_type"], None)
        return GGUFWeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            raw=inner_tensors["_data"],
            tensor_type=tensor_type,
            tensor_shape=tensor_shape,
            dtype=dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, GGUFWeightTensor):
                return weight.linear(input, bias=bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        kwargs = kwargs or {}
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, GGUFWeightTensor):
                return weight.linear(input, bias=bias)
        if op is torch.ops.aten.detach:
            t = args[0]
            return GGUFWeightTensor.create(
                raw_tensor=op(t._data),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
                tensor_type=getattr(t, "_tensor_type", None),
                tensor_shape=getattr(t, "_tensor_shape", None),
            )
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_data = op(t._data, device=device, **(kwargs or {}))
            return GGUFWeightTensor.create(
                raw_tensor=out_data,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
                tensor_type=getattr(t, "_tensor_type", None),
                tensor_shape=getattr(t, "_tensor_shape", None),
            )
        return _gguf_qfallback(op, *args, **(kwargs or {}))


class QLinearGGUF(QModuleMixin, torch.nn.Linear):
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
        self._gguf_default_dtype = dtype

    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
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

    def set_default_dtype(self, dtype):
        self._gguf_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _GGUF_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if isinstance(qweight, GGUFWeightTensor):
            return qweight.linear(input, bias=self.bias)
        return torch.nn.functional.linear(input, qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _GGUF_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_raw = state_dict.pop(weight_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_raw is None:
            missing_keys.append(weight_key)

        target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.weight.dtype)
        if weight_raw is not None:
            gguf_weight = GGUFWeightTensor.create(
                raw_tensor=weight_raw,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight_raw.device,
                requires_grad=False,
            )
            self.weight = torch.nn.Parameter(gguf_weight, requires_grad=False)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

        if torch.is_tensor(weight_raw):
            scale_device = weight_raw.device
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


def _collect_gguf_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if not _is_gguf_qtype(getattr(tensor, "tensor_type", None)):
            continue
        specs.append({"name": key[:-7], "tensor": tensor})
    return specs


def detect(state_dict, verboseLevel=1):
    if gguf is None:
        return {"matched": False, "kind": "none", "details": {"error": "gguf not installed"}}
    specs = _collect_gguf_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs][:8]
    return {"matched": True, "kind": "gguf", "details": {"count": len(specs), "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if gguf is None:
        return {"state_dict": state_dict, "quant_map": {}}
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_gguf_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    _set_default_dtype_from_loader(default_dtype)
    quant_map = {spec["name"]: {"weights": _GGUF_QTYPE_NAME, "activations": "none"} for spec in specs}
    return {"state_dict": state_dict, "quant_map": quant_map}


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    if default_dtype is None or model is None or not quantization_map:
        return quantization_map or {}, []
    _set_default_dtype_from_loader(default_dtype)
    quantized = set(quantization_map.keys())
    for name, module in model.named_modules():
        if name in quantized and isinstance(module, torch.nn.Linear):
            module._router_default_dtype = default_dtype
    return quantization_map or {}, []
