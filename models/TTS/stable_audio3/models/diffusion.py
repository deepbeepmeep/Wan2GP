import torch
from torch import nn
import typing as tp

from .conditioners import MultiConditioner
from .dit import DiffusionTransformer
from models.TTS.stable_audio3.inference.distribution_shift import FluxDistributionShift, DistributionShift, LogSNRShift, IdentityDistributionShift

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: nn.Module,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
            distribution_shift_options = None,
            sampling_distribution_shift_options = None,
            mask_padding_attention: bool = False,
            use_effective_length_for_schedule: bool = False,
            pretransform: tp.Optional[nn.Module] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            local_add_cond_ids: tp.List[str] = [],
            modular_local_cond_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.local_add_cond_ids = local_add_cond_ids
        self.modular_local_cond_ids = modular_local_cond_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length
        self.mask_padding_attention = mask_padding_attention
        self.use_effective_length_for_schedule = use_effective_length_for_schedule

        self.dist_shift = None
        if distribution_shift_options is not None:
            self.dist_shift = self._create_dist_shift(distribution_shift_options)

        # Sampling dist_shift: separate config for inference-time schedule
        if sampling_distribution_shift_options is not None:
            self.sampling_dist_shift = self._create_dist_shift(sampling_distribution_shift_options)
        else:
            # Default: seq_len-invariant LogSNR shift matching legacy log_snr_sampling=True
            self.sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)

    @staticmethod
    def _create_dist_shift(options: dict):
        """Create a distribution shift object from config options."""
        dist_shift_type = options.get("type", "full")
        dist_shift_kwargs = {k: v for k, v in options.items() if k != "type"}
        if dist_shift_type == "none":
            return IdentityDistributionShift()
        elif dist_shift_type == "flux":
            return FluxDistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "full":
            return DistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "logsnr":
            return LogSNRShift(**dist_shift_kwargs)
        else:
            raise ValueError(f"Unknown distribution shift type: {dist_shift_type}. Expected 'none', 'flux', 'full', or 'logsnr'.")     

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        local_add_cond = None
        modular_local_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.local_add_cond_ids) > 0:
            # Concatenate all local conditioning inputs over the channel dimension
            # Assumes that the local conditioning inputs are of shape (batch, channels, seq)
            local_add_cond = torch.cat([conditioning_tensors[key][0] for key in self.local_add_cond_ids], dim=1)

        if len(self.modular_local_cond_ids) > 0:
            # Keep modular local conditioning as a dict of tensors (not concatenated)
            # Each tensor is of shape (batch, channels, seq)
            modular_local_cond = {}
            for key in self.modular_local_cond_ids:
                if key in conditioning_tensors:
                    modular_local_cond[key] = conditioning_tensors[key][0]
            # Only set if we have any conditioning
            if len(modular_local_cond) == 0:
                modular_local_cond = None

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


class DiTWrapper(nn.Module):
    def __init__(
        self,
        diffusion_objective: str,
        *args,
        **kwargs
    ):
        super().__init__()

        self.diffusion_objective = diffusion_objective

        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                local_add_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            local_add_cond=local_add_cond,
            **kwargs)


