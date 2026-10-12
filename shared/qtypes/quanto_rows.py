# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Row projection for Quanto INT2/INT4 without unpacking the full weight."""
import torch
from torch._subclasses.fake_tensor import is_fake
from optimum.quanto.tensor.weights.qbits import WeightQBitsTensor
from optimum.quanto.tensor.weights.tinygemm.qbits import TinyGemmWeightQBitsTensor
from optimum.quanto.tensor.weights.awq.qbits import AWQWeightQBitsTensor
from mmgp import offload


class _QuantoBitsRowProjection:
    @staticmethod
    def supports(module, x):
        weight = getattr(module, "weight", None)
        return (type(weight) in (WeightQBitsTensor, TinyGemmWeightQBitsTensor, AWQWeightQBitsTensor)
                and weight.axis == 0 and weight.dtype == x.dtype and not is_fake(x)
                and (type(weight) is not TinyGemmWeightQBitsTensor or weight.is_cuda)
                and getattr(module, "activation_qtype", None) is None)

    @staticmethod
    def key(module):
        return None

    @staticmethod
    def rows(module, x, start, stop):
        weight = module.weight
        n, k = weight.shape
        if isinstance(weight, (TinyGemmWeightQBitsTensor, AWQWeightQBitsTensor)):
            first, last = start // 128 * 128, min((stop + 127) // 128 * 128, n)
            if isinstance(weight, TinyGemmWeightQBitsTensor):
                out = torch._weight_int4pack_mm(x, weight._data._data[first // 8:last // 8], weight._group_size,
                                                weight._scale_shift[:, first:last].contiguous())
            else:
                out = torch.ops.quanto.gemm_f16i4_awq(x, weight._data._data[first // 4:last // 4],
                    weight._scale[:, first:last].contiguous(), weight._shift[:, first:last].contiguous(),
                    rows=x.shape[0], out_cols=last - first, in_cols=k, bits=4, group_size=weight._group_size)
            out = out[:, start - first:stop - first]
        else:
            groups = k // weight._group_size if weight._group_size else 1
            packed = weight._data._data
            indices = torch.arange(start * groups, stop * groups, device=x.device)
            # Quanto interleaves distant groups in each byte, along dimension zero.
            data = packed.index_select(0, indices % packed.shape[0])
            data = (data >> ((indices // packed.shape[0]) * weight.qtype.bits).unsqueeze(1)) & ((1 << weight.qtype.bits) - 1)
            scale = weight._scale[start * groups:stop * groups]
            shift = weight._shift[start * groups:stop * groups]
            if not shift.dtype.is_floating_point:
                data = data.to(torch.int8) - shift.to(torch.int8)
            dense = scale * data
            if shift.dtype.is_floating_point:
                dense.sub_(shift)
            out = x @ dense.reshape(stop - start, k).t()
        if module.bias is not None:
            bias = module.bias[start:stop]
            return out.add_(bias) if bias.dtype == out.dtype else out + bias
        return out


_QUANTO_BITS_ROW_PROJECTION = _QuantoBitsRowProjection()
offload.register_row_projection(_QUANTO_BITS_ROW_PROJECTION)
