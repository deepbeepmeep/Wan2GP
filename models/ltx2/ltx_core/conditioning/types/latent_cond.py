import torch

from ..exceptions import ConditioningError
from ..item import ConditioningItem
from ...tools import LatentTools
from ...types import AudioLatentShape, LatentState


class VideoConditionByLatentIndex(ConditioningItem):
    """
    Conditions video generation by injecting latents at a specific latent frame index.
    Replaces tokens in the latent state at positions corresponding to latent_idx,
    and sets denoise strength according to the strength parameter.
    """

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        cond_batch, cond_channels, _, cond_height, cond_width = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_height, cond_width) != (tgt_batch, tgt_channels, tgt_height, tgt_width):
            raise ConditioningError(
                f"Can't apply image conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_height}, {tgt_width}). Make sure "
                "the image and latent have the same spatial shape."
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.latent_idx)
        )
        stop_token = start_token + tokens.shape[1]

        latent_state = latent_state.clone()

        latent_state.latent[:, start_token:stop_token] = tokens
        latent_state.clean_latent[:, start_token:stop_token] = tokens
        latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength

        return latent_state


class AudioConditionByLatent(ConditioningItem):
    """
    Conditions audio generation by injecting a full latent sequence.
    Replaces tokens in the latent state with the provided audio latents,
    and sets denoise strength according to the strength parameter.
    """

    def __init__(self, latent: torch.Tensor, strength: float):
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        if not isinstance(latent_tools.target_shape, AudioLatentShape):
            raise ConditioningError("Audio conditioning requires an audio latent target shape.")

        cond_batch, cond_channels, cond_frames, cond_bins = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_bins = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_frames, cond_bins) != (tgt_batch, tgt_channels, tgt_frames, tgt_bins):
            raise ConditioningError(
                f"Can't apply audio conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_bins})."
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        latent_state = latent_state.clone()
        latent_state.latent[:, : tokens.shape[1]] = tokens
        latent_state.clean_latent[:, : tokens.shape[1]] = tokens
        latent_state.denoise_mask[:, : tokens.shape[1]] = 1.0 - self.strength

        return latent_state


class AudioConditionByLatentPrefix(ConditioningItem):
    """
    Conditions audio generation by freezing only an initial latent prefix.
    The provided latents remain clean while the remaining target tokens stay noisy.
    """

    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        if not isinstance(latent_tools.target_shape, AudioLatentShape):
            raise ConditioningError("Audio prefix conditioning requires an audio latent target shape.")

        cond_batch, cond_channels, cond_frames, cond_bins = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_bins = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_bins) != (tgt_batch, tgt_channels, tgt_bins) or cond_frames > tgt_frames:
            raise ConditioningError(
                f"Can't apply audio prefix conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, <= {tgt_frames}, {tgt_bins})."
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        latent_state = latent_state.clone()
        latent_state.latent[:, : tokens.shape[1]] = tokens
        latent_state.clean_latent[:, : tokens.shape[1]] = tokens
        latent_state.denoise_mask[:, : tokens.shape[1]] = 0.0

        return latent_state


class AudioConditionByReferenceLatent(ConditioningItem):
    """
    Prepends clean reference-audio tokens to the audio latent state.
    The prepended tokens use negative temporal positions and a zero denoise mask so
    they act as a fixed identity reference during denoising.
    """

    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        if not isinstance(latent_tools.target_shape, AudioLatentShape):
            raise ConditioningError("Audio reference conditioning requires an audio latent target shape.")

        cond_batch, cond_channels, _, cond_bins = self.latent.shape
        tgt_batch, tgt_channels, _, tgt_bins = latent_tools.target_shape.to_torch_shape()
        if (cond_batch, cond_channels, cond_bins) != (tgt_batch, tgt_channels, tgt_bins):
            raise ConditioningError(
                f"Can't apply audio reference conditioning item to latent with shape {latent_tools.target_shape}, "
                f"expected shape is ({tgt_batch}, {tgt_channels}, *, {tgt_bins})."
            )

        ref_shape = AudioLatentShape.from_torch_shape(self.latent.shape)
        tokens = latent_tools.patchifier.patchify(self.latent)
        positions = latent_tools.patchifier.get_patch_grid_bounds(ref_shape, device=tokens.device).to(dtype=torch.float32)
        time_per_latent = float(latent_tools.patchifier.hop_length)
        time_per_latent *= float(latent_tools.patchifier.audio_latent_downsample_factor)
        time_per_latent /= float(latent_tools.patchifier.sample_rate)
        ref_duration = positions[:, :, -1:, 1]
        positions = positions - ref_duration - time_per_latent
        ref_mask = torch.zeros((tokens.shape[0], tokens.shape[1], 1), device=tokens.device, dtype=torch.float32)

        latent_state = latent_state.clone()
        return LatentState(
            latent=torch.cat([tokens, latent_state.latent], dim=1),
            denoise_mask=torch.cat([ref_mask, latent_state.denoise_mask], dim=1),
            positions=torch.cat([positions.to(latent_state.positions.dtype), latent_state.positions], dim=2),
            clean_latent=torch.cat([tokens, latent_state.clean_latent], dim=1),
        )
