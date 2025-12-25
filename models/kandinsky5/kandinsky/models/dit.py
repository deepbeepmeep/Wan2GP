import torch
from torch import nn

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
)
from .utils import fractal_flatten, fractal_unflatten


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto", text_token_padding=False):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim, attention_engine, text_token_padding)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def _modulate_inplace(self, x, shift, scale, gate):
        shift = shift.to(dtype=x.dtype)
        scale = scale.to(dtype=x.dtype)
        gate = gate.to(dtype=x.dtype)
        scale.add_(1.0)
        x.mul_(scale.unsqueeze(1))
        x.add_(shift.unsqueeze(1))
        return gate.unsqueeze(1)

    def forward(self, x, time_embed, rope, attention_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.self_attention_norm(x)
        gate = self._modulate_inplace(out, shift, scale, gate)
        out = self.self_attention(out, rope, attention_mask)
        x.addcmul_(out, gate)
        out = None

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.feed_forward_norm(x)
        gate = self._modulate_inplace(out, shift, scale, gate)
        out = self.feed_forward(out)
        x.addcmul_(out, gate)
        out = None
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto", text_token_padding=False):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)
        self.ffn_mult = max(int(round(ff_dim / model_dim)), 1)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, attention_engine)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, attention_engine,text_token_padding)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def _modulate_inplace(self, x, shift, scale, gate):
        shift = shift.to(dtype=x.dtype)
        scale = scale.to(dtype=x.dtype)
        gate = gate.to(dtype=x.dtype)
        scale.add_(1.0)
        x.mul_(scale.unsqueeze(1))
        x.add_(shift.unsqueeze(1))
        return gate.unsqueeze(1)

    def _apply_ffn_chunked(self, ffn_in: torch.Tensor) -> torch.Tensor:
        if self.ffn_mult <= 1:
            return self.feed_forward(ffn_in)
        seq_len = ffn_in.shape[1]
        ffn_flat = ffn_in.reshape(-1, ffn_in.shape[-1])
        chunk_size = max(seq_len // self.ffn_mult, 1)
        for chunk in torch.split(ffn_flat, chunk_size, dim=0):
            chunk[...] = self.feed_forward(chunk)
        return ffn_in

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params, attention_mask=None):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = self.self_attention_norm(visual_embed)
        gate = self._modulate_inplace(visual_out, shift, scale, gate)
        visual_out = self.self_attention(visual_out, rope, sparse_params)
        visual_embed.addcmul_(visual_out, gate)
        visual_out = None

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = self.cross_attention_norm(visual_embed)
        gate = self._modulate_inplace(visual_out, shift, scale, gate)
        visual_out = self.cross_attention(visual_out, text_embed, attention_mask)
        visual_embed.addcmul_(visual_out, gate)
        visual_out = None

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = self.feed_forward_norm(visual_embed)
        gate = self._modulate_inplace(visual_out, shift, scale, gate)
        visual_out = self._apply_ffn_chunked(visual_out)
        visual_embed.addcmul_(visual_out, gate)
        visual_out = None
        return visual_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_engine="auto",
        instruct_type=None,
        text_token_padding=False
    ):
        super().__init__()
        self.instruct_type = instruct_type
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.text_token_padding = text_token_padding

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond or instruct_type=='channel' else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine, text_token_padding)
                for _ in range(num_text_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine, text_token_padding)
                for _ in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        self._interrupt_check = None

    def before_text_transformer_blocks(self, text_embed, time, pooled_text_embed, x,
                                       text_rope_pos):
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    def before_visual_transformer_blocks(self, visual_embed, visual_rope_pos, scale_factor,
                                         sparse_params):
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape,
                                                    block_mask=to_fractal)
        return visual_embed.unsqueeze(0), visual_shape, to_fractal, visual_rope

    def after_blocks(self, visual_embed, visual_shape, to_fractal, text_embed, time_embed):
        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)
        return x

    def _check_interrupt(self):
        if self._interrupt_check is None:
            return False
        return bool(self._interrupt_check())

    @staticmethod
    def _normalize_list(value, count):
        if isinstance(value, (list, tuple)):
            if len(value) != count:
                raise ValueError(f"Expected {count} items, got {len(value)}.")
            return list(value)
        return [value] * count

    def _forward_single(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor,
        sparse_params,
        attention_mask,
    ):
        text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos
        )
        x = None
        pooled_text_embed = None

        for text_transformer_block in self.text_transformer_blocks:
            text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
            if self._check_interrupt():
                return None
        text_rope = None

        visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params
        )
        visual_rope_pos = None

        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(
                visual_embed, text_embed, time_embed, visual_rope, sparse_params, attention_mask
            )
            if self._check_interrupt():
                return None
        visual_rope = None

        return self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    def _forward_joint(
        self,
        x_list,
        text_embed_list,
        pooled_text_embed_list,
        time_list,
        visual_rope_pos_list,
        text_rope_pos_list,
        scale_factor_list,
        sparse_params_list,
        attention_mask_list,
    ):
        count = len(x_list)
        text_embed_list = self._normalize_list(text_embed_list, count)
        pooled_text_embed_list = self._normalize_list(pooled_text_embed_list, count)
        time_list = self._normalize_list(time_list, count)
        visual_rope_pos_list = self._normalize_list(visual_rope_pos_list, count)
        text_rope_pos_list = self._normalize_list(text_rope_pos_list, count)
        scale_factor_list = self._normalize_list(scale_factor_list, count)
        sparse_params_list = self._normalize_list(sparse_params_list, count)
        attention_mask_list = self._normalize_list(attention_mask_list, count)

        text_embed_out = [None] * count
        time_embed_out = [None] * count
        text_rope_out = [None] * count
        visual_embed_out = [None] * count

        for idx in range(count):
            text_embed_out[idx], time_embed_out[idx], text_rope_out[idx], visual_embed_out[idx] = (
                self.before_text_transformer_blocks(
                    text_embed_list[idx],
                    time_list[idx],
                    pooled_text_embed_list[idx],
                    x_list[idx],
                    text_rope_pos_list[idx],
                )
            )
            x_list[idx] = None
            pooled_text_embed_list[idx] = None

        for text_transformer_block in self.text_transformer_blocks:
            for idx in range(count):
                text_embed_out[idx] = text_transformer_block(
                    text_embed_out[idx], time_embed_out[idx], text_rope_out[idx], attention_mask_list[idx]
                )
                if self._check_interrupt():
                    return None
        for idx in range(count):
            text_rope_out[idx] = None

        visual_shape_list = [None] * count
        to_fractal_list = [None] * count
        visual_rope_out = [None] * count

        for idx in range(count):
            visual_embed_out[idx], visual_shape_list[idx], to_fractal_list[idx], visual_rope_out[idx] = (
                self.before_visual_transformer_blocks(
                    visual_embed_out[idx],
                    visual_rope_pos_list[idx],
                    scale_factor_list[idx],
                    sparse_params_list[idx],
                )
            )
            visual_rope_pos_list[idx] = None

        for visual_transformer_block in self.visual_transformer_blocks:
            for idx in range(count):
                visual_embed_out[idx] = visual_transformer_block(
                    visual_embed_out[idx],
                    text_embed_out[idx],
                    time_embed_out[idx],
                    visual_rope_out[idx],
                    sparse_params_list[idx],
                    attention_mask_list[idx],
                )
                if self._check_interrupt():
                    return None
        for idx in range(count):
            visual_rope_out[idx] = None

        outputs = []
        for idx in range(count):
            outputs.append(
                self.after_blocks(
                    visual_embed_out[idx],
                    visual_shape_list[idx],
                    to_fractal_list[idx],
                    text_embed_out[idx],
                    time_embed_out[idx],
                )
            )
        return outputs

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None,
    ):
        if isinstance(x, (list, tuple)):
            return self._forward_joint(
                list(x),
                text_embed,
                pooled_text_embed,
                time,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                sparse_params,
                attention_mask,
            )

        return self._forward_single(
            x,
            text_embed,
            pooled_text_embed,
            time,
            visual_rope_pos,
            text_rope_pos,
            scale_factor,
            sparse_params,
            attention_mask,
        )


def get_dit(conf, text_token_padding=False):
    dit = DiffusionTransformer3D(**conf, text_token_padding=text_token_padding)
    return dit
