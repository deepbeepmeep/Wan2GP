# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.models.normalization import RMSNorm


try:
    # from .dcformer import DCMHAttention
    from .customer_attention_processor import (
        Attention,
        CustomLiteLAProcessor2_0,
        CustomerAttnProcessor2_0,
    )
except ImportError:
    # from dcformer import DCMHAttention
    from customer_attention_processor import (
        Attention,
        CustomLiteLAProcessor2_0,
        CustomerAttnProcessor2_0,
    )


logger = logging.get_logger(__name__)


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=None,
        act=None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        if norm is not None:
            self.norm = RMSNorm(out_dim, elementwise_affine=False)
        else:
            self.norm = None
        if act is not None:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)

        return x


class LinearTransformerBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        use_adaln_single=True,
        cross_attention_dim=None,
        added_kv_proj_dim=None,
        context_pre_only=False,
        mlp_ratio=4.0,
        add_cross_attention=False,
        add_cross_attention_dim=None,
        qk_norm=None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm=qk_norm,
            processor=CustomLiteLAProcessor2_0(),
        )

        self.add_cross_attention = add_cross_attention
        self.context_pre_only = context_pre_only

        if add_cross_attention and add_cross_attention_dim is not None:
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=add_cross_attention_dim,
                added_kv_proj_dim=add_cross_attention_dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=CustomerAttnProcessor2_0(),
            )

        self.norm2 = RMSNorm(dim, 1e-06, elementwise_affine=False)

        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
        )
        self.use_adaln_single = use_adaln_single
        if use_adaln_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        temb: torch.FloatTensor = None,
    ):

        N = hidden_states.shape[0]

        # step 1: AdaLN single
        if self.use_adaln_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # step 2: attention
        if not self.add_cross_attention:
            attn_output, encoder_hidden_states = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
        else:
            attn_output, _ = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=None,
            )

        if self.use_adaln_single:
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        if self.add_cross_attention:
            attn_output = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
            hidden_states = attn_output + hidden_states

        # step 3: add norm
        norm_hidden_states = self.norm2(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # step 4: feed forward
        ff_output = self.ff(norm_hidden_states)
        if self.use_adaln_single:
            ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states
