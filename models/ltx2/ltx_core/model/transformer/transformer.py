# Original Code by LightTricks (https://github.com/Lightricks/LTX-2)
# VRAM Optimizations by DeepBeepMeep (c) 2026. Please quote DeepBeepMeep / WanGP if reused

from dataclasses import dataclass, replace

import torch

from ...guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from .adaln import adaln_embedding_coefficient
from .attention import Attention, AttentionCallable, AttentionFunction
from .feed_forward import FeedForward
from .rope import LTXRopeType
from .transformer_args import TransformerArgs
from ...utils import rms_norm


def _reshape_hidden_states(hidden_states: torch.Tensor, frames: int) -> torch.Tensor:
    return hidden_states.reshape(hidden_states.shape[0], frames, -1, hidden_states.shape[-1])


def _restore_hidden_states_shape(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])


def _apply_scale_shift(hidden_states: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    if scale.shape[1] == hidden_states.shape[1]:
        if in_place:
            hidden_states.mul_(1 + scale).add_(shift)
            return hidden_states
        return hidden_states * (1 + scale) + shift

    hidden_states_reshaped = _reshape_hidden_states(hidden_states, scale.shape[1])
    if in_place:
        hidden_states_reshaped.mul_(1 + scale.unsqueeze(2)).add_(shift.unsqueeze(2))
        return _restore_hidden_states_shape(hidden_states_reshaped)
    return _restore_hidden_states_shape(hidden_states_reshaped * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2))


def _apply_gate(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.shape[1] == hidden_states.shape[1]:
        hidden_states.mul_(gate)
        return hidden_states
    hidden_states = _reshape_hidden_states(hidden_states, gate.shape[1])
    hidden_states.mul_(gate.unsqueeze(2))
    return _restore_hidden_states_shape(hidden_states)


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(adaln_embedding_coefficient(video.cross_attention_adaln), video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(adaln_embedding_coefficient(audio.cross_attention_adaln), audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )
        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps
        self.ff_chunk_min_tokens = 1024

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def _apply_ffn_chunked(self, ffn: FeedForward, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.ff_chunk_min_tokens:
            return ffn(x)
        chunk_size = max(int(x.shape[1] / max(ffn.mult, 1)), 1)
        x_flat = x.view(-1, x.shape[-1])
        for chunk in torch.split(x_flat, chunk_size):
            chunk[...] = ffn(chunk)
        return x

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        nag: dict | None = None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        if not cross_attention_adaln:
            return attn([rms_norm(x, eps=self.norm_eps)], context_list=[context], mask=context_mask, NAG=nag)
        q_shift, q_scale, q_gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
        return apply_cross_attention_adaln(
            x,
            context,
            attn,
            q_shift,
            q_scale,
            q_gate,
            prompt_scale_shift_table,
            prompt_timestep,
            context_mask,
            nag=nag,
            norm_eps=self.norm_eps,
        )

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and audio.enabled and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and video.enabled and vx.numel() > 0)
        run_a2v = run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx)
        run_v2a = run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            vshift_msa = vshift_msa.to(vx.dtype)
            vscale_msa = vscale_msa.to(vx.dtype)
            vgate_msa = vgate_msa.to(vx.dtype)
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps)
                norm_vx = _apply_scale_shift(norm_vx, vscale_msa, vshift_msa)
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                x_list = [norm_vx]
                del norm_vx
                attn_out = self.attn1(x_list, pe=video.positional_embeddings)
                attn_out = _apply_gate(attn_out, vgate_msa)
                attn_out.mul_(v_mask)
                vx.add_(attn_out)
                attn_out = None
            attn_out = self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                nag=video.nag,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            vx.add_(attn_out)
            attn_out = None
            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            ashift_msa = ashift_msa.to(ax.dtype)
            ascale_msa = ascale_msa.to(ax.dtype)
            agate_msa = agate_msa.to(ax.dtype)

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps)
                norm_ax = _apply_scale_shift(norm_ax, ascale_msa, ashift_msa)
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                x_list = [norm_ax]
                del norm_ax
                attn_out = self.audio_attn1(x_list, pe=audio.positional_embeddings)
                attn_out = _apply_gate(attn_out, agate_msa)
                attn_out.mul_(a_mask)
                ax.add_(attn_out)
                attn_out = None
            attn_out = self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                nag=audio.nag if audio.nag is not None and audio.nag.get("enable_audio_text_nag", False) else None,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            ax.add_(attn_out)
            attn_out = None
            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            scale_ca_audio_hidden_states_a2v = scale_ca_audio_hidden_states_a2v.to(ax.dtype)
            shift_ca_audio_hidden_states_a2v = shift_ca_audio_hidden_states_a2v.to(ax.dtype)
            scale_ca_audio_hidden_states_v2a = scale_ca_audio_hidden_states_v2a.to(ax.dtype)
            shift_ca_audio_hidden_states_v2a = shift_ca_audio_hidden_states_v2a.to(ax.dtype)
            gate_out_v2a = gate_out_v2a.to(ax.dtype)

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )
            scale_ca_video_hidden_states_a2v = scale_ca_video_hidden_states_a2v.to(vx.dtype)
            shift_ca_video_hidden_states_a2v = shift_ca_video_hidden_states_a2v.to(vx.dtype)
            scale_ca_video_hidden_states_v2a = scale_ca_video_hidden_states_v2a.to(vx.dtype)
            shift_ca_video_hidden_states_v2a = shift_ca_video_hidden_states_v2a.to(vx.dtype)
            gate_out_a2v = gate_out_a2v.to(vx.dtype)

            if run_a2v:
                vx_scaled = _apply_scale_shift(
                    vx_norm3.clone(),
                    scale_ca_video_hidden_states_a2v,
                    shift_ca_video_hidden_states_a2v,
                )
                ax_scaled = _apply_scale_shift(
                    ax_norm3.clone(),
                    scale_ca_audio_hidden_states_a2v,
                    shift_ca_audio_hidden_states_a2v,
                )
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                x_list, context_list  = [vx_scaled], [ax_scaled]
                del vx_scaled, ax_scaled
                attn_out = self.audio_to_video_attn(
                    x_list,
                    context_list=context_list,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                )
                attn_out = _apply_gate(attn_out, gate_out_a2v)
                attn_out.mul_(a2v_mask)
                vx.add_(attn_out)
                attn_out = vx_scaled = ax_scaled = None

            if run_v2a:
                ax_scaled = _apply_scale_shift(
                    ax_norm3,
                    scale_ca_audio_hidden_states_v2a,
                    shift_ca_audio_hidden_states_v2a,
                )
                vx_scaled = _apply_scale_shift(
                    vx_norm3,
                    scale_ca_video_hidden_states_v2a,
                    shift_ca_video_hidden_states_v2a,
                )
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                x_list, context_list = [ax_scaled], [vx_scaled]
                del ax_scaled, vx_scaled
                attn_out = self.video_to_audio_attn(
                    x_list,
                    context_list=context_list,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                attn_out = _apply_gate(attn_out, gate_out_v2a)
                attn_out.mul_(v2a_mask)
                ax.add_(attn_out)
                attn_out = ax_scaled = vx_scaled = None

            del gate_out_a2v, gate_out_v2a
            del (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
            )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vshift_mlp = vshift_mlp.to(vx.dtype)
            vscale_mlp = vscale_mlp.to(vx.dtype)
            vgate_mlp = vgate_mlp.to(vx.dtype)
            vx_scaled = rms_norm(vx, eps=self.norm_eps)
            vx_scaled = _apply_scale_shift(vx_scaled, vscale_mlp, vshift_mlp)
            ff_out = self._apply_ffn_chunked(self.ff, vx_scaled)
            ff_out = _apply_gate(ff_out, vgate_mlp)
            vx.add_(ff_out)
            ff_out = vx_scaled = None

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ashift_mlp = ashift_mlp.to(ax.dtype)
            ascale_mlp = ascale_mlp.to(ax.dtype)
            agate_mlp = agate_mlp.to(ax.dtype)
            ax_scaled = rms_norm(ax, eps=self.norm_eps)
            ax_scaled = _apply_scale_shift(ax_scaled, ascale_mlp, ashift_mlp)
            ff_out = self.audio_ff(ax_scaled)
            ff_out = _apply_gate(ff_out, agate_mlp)
            ax.add_(ff_out)
            ff_out = ax_scaled = None

            del ashift_mlp, ascale_mlp, agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor | None,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
    nag: dict | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    if prompt_scale_shift_table is not None and prompt_timestep is not None:
        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        # Context is reused across blocks in LTX 2.3 prompt AdaLN, so this call must stay out-of-place.
        context = _apply_scale_shift(context, scale_kv, shift_kv, in_place=False)
    attn_input = _apply_scale_shift(rms_norm(x, eps=norm_eps), q_scale.squeeze(2), q_shift.squeeze(2))
    out = attn([attn_input], context_list=[context], mask=context_mask, NAG=nag)
    return _apply_gate(out, q_gate.squeeze(2))
