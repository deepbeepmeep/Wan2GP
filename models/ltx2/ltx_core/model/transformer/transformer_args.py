from dataclasses import dataclass, replace
import os

import torch

from .adaln import AdaLayerNormSingle
from .modality import Modality
from .rope import (
    LTXRopeType,
    RopeCache,
    build_rope_cache,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
)
from .text_projection import PixArtAlphaTextProjection


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor | RopeCache | tuple[torch.Tensor, torch.Tensor]
    cross_positional_embeddings: torch.Tensor | RopeCache | tuple[torch.Tensor, torch.Tensor] | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool


def _can_broadcast_timesteps(frame_indices: torch.Tensor | None, frame_count: int) -> bool:
    if frame_indices is None or frame_indices.ndim != 2:
        return False
    token_count = frame_indices.shape[1]
    if frame_count <= 0 or token_count % frame_count != 0:
        return False
    tokens_per_frame = token_count // frame_count
    if tokens_per_frame == 0:
        return False
    try:
        frame_view = frame_indices.reshape(frame_indices.shape[0], frame_count, tokens_per_frame)
    except RuntimeError:
        return False
    expected = torch.arange(frame_count, device=frame_indices.device).view(1, frame_count, 1)
    return torch.equal(frame_view, expected.expand(frame_indices.shape[0], -1, tokens_per_frame))


def _maybe_compress_timesteps(
    timestep: torch.Tensor, frame_indices: torch.Tensor | None, batch_size: int
) -> torch.Tensor:
    if frame_indices is None or timestep.ndim != 2:
        return timestep
    token_count = frame_indices.shape[1]
    if timestep.shape[1] != token_count:
        return timestep
    frame_count = int(frame_indices.max().item()) + 1
    if not _can_broadcast_timesteps(frame_indices, frame_count):
        return timestep
    tokens_per_frame = token_count // frame_count
    if tokens_per_frame <= 1:
        return timestep
    return timestep.reshape(batch_size, frame_count, tokens_per_frame)[:, :, 0]


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""

        timestep = timestep * self.timestep_scale_multiplier
        timestep = _maybe_compress_timesteps(timestep, frame_indices, batch_size)
        timestep, embedded_timestep = self.adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        if frame_indices is None or _can_broadcast_timesteps(frame_indices, timestep.shape[1]):
            return timestep, embedded_timestep

        gather_index = frame_indices.unsqueeze(-1)
        timestep = timestep.gather(1, gather_index.expand(-1, -1, timestep.shape[-1]))
        embedded_timestep = embedded_timestep.gather(1, gather_index.expand(-1, -1, embedded_timestep.shape[-1]))
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare context for transformer blocks."""
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = context.view(batch_size, -1, x.shape[-1])

        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
        rope_axes: tuple[int, ...] | None = None,
        rope_max_pos: list[int] | None = None,
    ) -> RopeCache | tuple[torch.Tensor, torch.Tensor]:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        return build_rope_cache(
            positions=positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            rope_axes=rope_axes,
            frame_indices=frame_indices,
            rope_max_pos=rope_max_pos,
            freq_grid_generator=freq_grid_generator,
        )

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        latent = modality.latent
        latent_dtype = latent.dtype
        weight = getattr(self.patchify_proj, "weight", None)
        weight_dtype = getattr(weight, "dtype", None)
        if weight_dtype is not None and latent_dtype != weight_dtype:
            latent = latent.to(weight_dtype)
            latent_dtype = latent.dtype
        if os.environ.get("WAN2GP_GGUF_TRACE_LATENT", "") in ("1", "true", "yes") and not hasattr(
            self, "_gguf_latent_logged"
        ):
            self._gguf_latent_logged = True
            print(f"[GGUF][patchify_proj] latent={latent_dtype} weight={weight_dtype}")
        x = self.patchify_proj(latent)
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps,
            x.shape[0],
            latent_dtype,
            frame_indices=modality.frame_indices,
        )
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, latent_dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=latent_dtype,
            frame_indices=modality.frame_indices,
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.audio_cross_attention_dim,
            max_pos=self.simple_preprocessor.max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
            frame_indices=modality.frame_indices,
            rope_axes=(0,),
            rope_max_pos=[self.cross_pe_max_pos],
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
            frame_indices=modality.frame_indices,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier
        timestep = _maybe_compress_timesteps(timestep, frame_indices, batch_size)

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        if frame_indices is None or _can_broadcast_timesteps(frame_indices, scale_shift_timestep.shape[1]):
            return scale_shift_timestep, gate_noise_timestep

        gather_index = frame_indices.unsqueeze(-1)
        scale_shift_timestep = scale_shift_timestep.gather(
            1, gather_index.expand(-1, -1, scale_shift_timestep.shape[-1])
        )
        gate_noise_timestep = gate_noise_timestep.gather(
            1, gather_index.expand(-1, -1, gate_noise_timestep.shape[-1])
        )

        return scale_shift_timestep, gate_noise_timestep
