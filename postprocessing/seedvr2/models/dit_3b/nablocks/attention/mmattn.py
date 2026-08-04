# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from itertools import accumulate
from typing import Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple
from shared.attention import pay_attention

from .....common.cache import Cache
from .....common.half_precision_fixes import safe_pad_operation

from ... import na
from ...attention import FlashAttentionVarlen
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import _apply_rotary_inplace, get_na_rope
from ...window import get_window_op


def _pack_window_batch(vid_list: list[torch.Tensor], txt: torch.Tensor, vid_lens: list[int], max_len: int) -> torch.Tensor:
    vid = vid_list.pop()
    batch = vid.new_zeros(len(vid_lens), max_len, *vid.shape[1:])
    offset = 0
    txt_len = txt.shape[0]
    for row, vid_len in enumerate(vid_lens):
        batch[row, :vid_len].copy_(vid[offset:offset + vid_len])
        batch[row, vid_len:vid_len + txt_len].copy_(txt)
        offset += vid_len
    return batch


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.heads = heads
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode, window_batch_size=kwargs.pop("attention_window_batch_size", 0))

    def forward(
        self,
        vid_list: list[torch.FloatTensor],  # l c
        txt_list: list[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_qkv, txt_qkv = self.proj_qkv.forward_disposable(vid_list, txt_list)
        vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        vid_v, txt_v = vid_v.clone(), txt_v.clone()
        vid_qkv = txt_qkv = None

        vid_q_list, txt_q_list = [vid_q], [txt_q]
        vid_q = txt_q = None
        vid_q, txt_q = self.norm_q.forward_disposable(vid_q_list, txt_q_list)
        vid_k_list, txt_k_list = [vid_k], [txt_k]
        vid_k = txt_k = None
        vid_k, txt_k = self.norm_k.forward_disposable(vid_k_list, txt_k_list)

        if self.rope:
            if self.rope.mm:
                vid_qk_list, txt_qk_list = [vid_q, vid_k], [txt_q, txt_k]
                vid_q = vid_k = txt_q = txt_k = None
                vid_q, vid_k, txt_q, txt_k = self.rope(vid_qk_list, vid_shape, txt_qk_list, txt_shape, cache)
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)

        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        q_list = [vid_q, txt_q]
        vid_q = txt_q = None
        q = concat(q_list)
        k_list = [vid_k, txt_k]
        vid_k = txt_k = None
        k = concat(k_list)
        v_list = [vid_v, txt_v]
        vid_v = txt_v = None
        v = concat(v_list)
        qkv_list = [q, k, v]
        q = k = v = None
        attn = self.attn(
            qkv_list,
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max()),
            q_lens=cache("mm_attention_lens", lambda: all_len.tolist()),
            k_lens=cache("mm_attention_lens", lambda: all_len.tolist()),
        )

        attn = rearrange(attn, "l h d -> l (h d)")
        attn_list = [attn]
        attn = None
        vid_out, txt_out = unconcat(attn_list)
        vid_out_list, txt_out_list = [vid_out], [txt_out]
        vid_out = txt_out = None
        vid_out, txt_out = self.proj_out.forward_disposable(vid_out_list, txt_out_list)
        return vid_out, txt_out


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(*args, attention_mode=attention_mode, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid_list: list[torch.FloatTensor],  # l c
        txt_list: list[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_win = window_partition(vid_list)
        txt_qkv = self.proj_qkv.forward_txt_disposable(txt_list)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        txt_v = txt_v.clone()
        txt_qkv = None
        txt_q_list = [txt_q]
        txt_q = None
        txt_q = self.norm_q.forward_txt_disposable(txt_q_list)
        txt_k_list = [txt_k]
        txt_k = None
        txt_k = self.norm_k.forward_txt_disposable(txt_k_list)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        vid_lens = cache_win("vid_lens_py", lambda: vid_len_win.tolist())
        offsets = cache_win("vid_offsets_py", lambda: list(accumulate(vid_lens, initial=0)))
        txt_len_py = cache_win("txt_len_py", lambda: int(txt_len[0].item()))
        txt_shape_repeat = cache_win("txt_shape_repeat", lambda: txt_shape.repeat_interleave(window_count, dim=0))
        vid_cos, vid_sin, txt_cos, txt_sin = self.rope.get_cos_sin(window_shape, txt_shape, cache_win, txt_q.device, txt_shape_repeat)
        txt_q_list, txt_k_list = [txt_q], [txt_k]
        txt_q = txt_k = None
        txt_q = _apply_rotary_inplace(txt_q_list, txt_cos, txt_sin)
        txt_k = _apply_rotary_inplace(txt_k_list, txt_cos, txt_sin)

        txt_windows = txt_q.new_empty(txt_len_py, len(vid_lens), self.heads, self.head_dim)
        window_batch_size = self.attn.window_batch_size or len(vid_lens)
        for start in range(0, len(vid_lens), window_batch_size):
            stop = min(start + window_batch_size, len(vid_lens))
            chunk_vid_lens = vid_lens[start:stop]
            chunk_all_lens = [length + txt_len_py for length in chunk_vid_lens]
            vid_start, vid_stop = offsets[start], offsets[stop]
            vid_qkv = self.proj_qkv.forward_vid_disposable([vid_win[vid_start:vid_stop]])
            vid_qkv = rearrange(vid_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
            vid_q, vid_k, vid_v = vid_qkv.unbind(1)
            vid_v = vid_v.clone()
            vid_qkv = None
            vid_q_list, vid_k_list = [vid_q], [vid_k]
            vid_q = vid_k = None
            vid_q = self.norm_q.forward_vid_disposable(vid_q_list)
            vid_k = self.norm_k.forward_vid_disposable(vid_k_list)
            vid_q_list, vid_k_list = [vid_q], [vid_k]
            vid_q = vid_k = None
            vid_q = _apply_rotary_inplace(vid_q_list, vid_cos[vid_start:vid_stop], vid_sin[vid_start:vid_stop])
            vid_k = _apply_rotary_inplace(vid_k_list, vid_cos[vid_start:vid_stop], vid_sin[vid_start:vid_stop])

            max_len = max(chunk_all_lens)
            vid_q_list = [vid_q]
            vid_q = None
            q_batch = _pack_window_batch(vid_q_list, txt_q, chunk_vid_lens, max_len)
            vid_k_list = [vid_k]
            vid_k = None
            k_batch = _pack_window_batch(vid_k_list, txt_k, chunk_vid_lens, max_len)
            vid_v_list = [vid_v]
            vid_v = None
            v_batch = _pack_window_batch(vid_v_list, txt_v, chunk_vid_lens, max_len)
            qkv_list = [q_batch, k_batch, v_batch]
            q_batch = k_batch = v_batch = None
            out = pay_attention(qkv_list, k_lens=None if min(chunk_all_lens) == max_len else chunk_all_lens, recycle_q=True)

            vid_attn = out.new_empty(vid_stop - vid_start, self.heads, self.head_dim)
            offset = 0
            for row, vid_len in enumerate(chunk_vid_lens):
                vid_attn[offset:offset + vid_len].copy_(out[row, :vid_len])
                txt_windows[:, start + row].copy_(out[row, vid_len:vid_len + txt_len_py])
                offset += vid_len
            out = None
            vid_attn_list = [vid_attn.flatten(1)]
            vid_attn = None
            vid_projected = self.proj_out.forward_vid_disposable(vid_attn_list)
            vid_win[vid_start:vid_stop].copy_(vid_projected)
            vid_projected = None

        txt_q = txt_k = txt_v = None
        txt_out = txt_windows.mean(1).flatten(1)
        txt_windows = None
        txt_out_list = [txt_out]
        txt_out = None
        txt_out = self.proj_out.forward_txt_disposable(txt_out_list)
        vid_win_list = [vid_win]
        vid_win = None
        return window_reverse(vid_win_list), txt_out
