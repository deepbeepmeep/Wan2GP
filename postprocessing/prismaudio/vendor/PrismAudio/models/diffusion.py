import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import numpy as np
import typing as tp

from .blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
#from .mmdit import MMAudio
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform

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

class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()

class DiffusionModelWrapper(nn.Module):
    def __init__(
                self,
                model: DiffusionModel,
                io_channels,
                sample_size,
                sample_rate,
                min_input_length,
                pretransform: tp.Optional[Pretransform] = None,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.model = model

        if pretransform is not None:
            self.pretransform = pretransform
        else:
            self.pretransform = None

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            zero_init: bool = False,
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            add_cond_ids: tp.List[str] = [],
            sync_cond_ids: tp.List[str] = [],
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
        self.prepend_cond_ids = prepend_cond_ids
        self.add_cond_ids = add_cond_ids
        self.sync_cond_ids = sync_cond_ids
        self.min_input_length = min_input_length
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        if zero_init is True:
            self.conditioner.apply(_basic_init)
            self.model.model.initialize_weights()


    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        add_input = None
        sync_input = None

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
                    # cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)
            # import ipdb
            # ipdb.set_trace()
            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.add_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            add_input = []

            for key in self.add_cond_ids:
                add_in = conditioning_tensors[key][0]

                # Add sequence dimension if it's not there
                if len(add_in.shape) == 2:
                    add_in = add_in.unsqueeze(1)
                # add_in = add_in.transpose(1,2)
                # add_in = F.interpolate(add_in, (194, ), mode='linear', align_corners=False)
                # add_in = add_in.transpose(1,2)
                add_input.append(add_in)

            add_input = torch.cat(add_input, dim=2)

        if len(self.sync_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            sync_input = []

            for key in self.sync_cond_ids:
                sync_in = conditioning_tensors[key][0]

                # Add sequence dimension if it's not there
                if len(sync_in.shape) == 2:
                    sync_in = sync_in.unsqueeze(1)
                sync_input.append(sync_in)

            sync_input = torch.cat(sync_input, dim=2)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]
                if len(global_cond_input.shape) == 2:
                    global_cond_input = global_cond_input.unsqueeze(1)
                global_conds.append(global_cond_input)

            # # Concatenate over the channel dimension
            # if global_conds[0].shape[-1] == 768:
            #     global_cond = torch.cat(global_conds, dim=-1)
            # else:
            #     global_cond = sum(global_conds)
            global_cond = sum(global_conds)
            # global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                if len(prepend_cond_input.shape) == 2:
                    prepend_cond_input = prepend_cond_input.unsqueeze(1)
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
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask,
                "add_cond": add_input,
                "sync_cond": sync_input
                }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        from ..inference.generation import generate_diffusion_cond
        return generate_diffusion_cond(self, *args, **kwargs)

class UNetCFG1DWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=True, supports_input_concat=True)

        from .adp import UNetCFG1d
        self.model = UNetCFG1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                input_concat_cond=None,
                global_cond=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                **kwargs):
        p = Profiler()

        p.tick("start")

        channels_list = None
        if input_concat_cond is not None:
            channels_list = [input_concat_cond]

        outputs = self.model(
            x,
            t,
            embedding=cross_attn_cond,
            embedding_mask=cross_attn_mask,
            features=global_cond,
            channels_list=channels_list,
            embedding_scale=cfg_scale,
            embedding_mask_proba=cfg_dropout_prob,
            batch_cfg=batch_cfg,
            rescale_cfg=rescale_cfg,
            negative_embedding=negative_cross_attn_cond,
            negative_embedding_mask=negative_cross_attn_mask,
            **kwargs)

        p.tick("UNetCFG1D forward")

        #print(f"Profiler: {p}")
        return outputs

class UNet1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=True, supports_input_concat=True)

        from .adp import UNet1d
        self.model = UNet1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                input_concat_cond=None,
                global_cond=None,
                cross_attn_cond=None,
                cross_attn_mask=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                **kwargs):

        channels_list = None
        if input_concat_cond is not None:

            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            channels_list = [input_concat_cond]

        outputs = self.model(
            x,
            t,
            features=global_cond,
            channels_list=channels_list,
            **kwargs)

        return outputs

class UNet1DUncondWrapper(DiffusionModel):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs
    ):
        super().__init__()

        from .adp import UNet1d
        self.model = UNet1d(in_channels=in_channels, *args, **kwargs)

        self.io_channels = in_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

class DAU1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=False, supports_input_concat=True)

        self.model = DiffusionAttnUnet1D(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                input_concat_cond=None,
                cross_attn_cond=None,
                cross_attn_mask=None,
                global_cond=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                **kwargs):

        return self.model(x, t, cond = input_concat_cond)

class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self,
        io_channels = 2,
        depth=14,
        n_attn_layers = 6,
        channels = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        cond_noise_aug = False,
        kernel_size = 5,
        learned_resample = False,
        strides = [2] * 13,
        conv_bias = True,
        use_snake = False
    ):
        super().__init__()

        self.cond_noise_aug = cond_noise_aug

        self.io_channels = io_channels

        if self.cond_noise_aug:
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers

        strides = [1] + strides

        block = nn.Identity()

        conv_block = partial(ResConvBlock, kernel_size=kernel_size, conv_bias = conv_bias, use_snake=use_snake)

        for i in range(depth, 0, -1):
            c = channels[i - 1]
            stride = strides[i-1]
            if stride > 2 and not learned_resample:
                raise ValueError("Must have stride 2 without learned resampling")

            if i > 1:
                c_prev = channels[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if (learned_resample or stride == 1) else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                )
            else:
                cond_embed_dim = 16 if not self.cond_noise_aug else 32
                block = nn.Sequential(
                    conv_block((io_channels + cond_dim) + cond_embed_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t, cond=None, cond_aug_scale=None):

        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)

        inputs = [x, timestep_embed]

        if cond is not None:
            if cond.shape[2] != x.shape[2]:
                cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)

            if self.cond_noise_aug:
                # Get a random number between 0 and 1, uniformly sampled
                if cond_aug_scale is None:
                    aug_level = self.rng.draw(cond.shape[0])[:, 0].to(cond)
                else:
                    aug_level = torch.tensor([cond_aug_scale]).repeat([cond.shape[0]]).to(cond)

                # Add noise to the conditioning signal
                cond = cond + torch.randn_like(cond) * aug_level[:, None, None]

                # Get embedding for noise cond level, reusing timestamp_embed
                aug_level_embed = expand_to_planes(self.timestep_embed(aug_level[:, None]), x.shape)

                inputs.append(aug_level_embed)

            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        return outputs

class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = DiffusionTransformer(*args, **kwargs)
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
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
            **kwargs)

class MMDiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = MMAudio(*args, **kwargs)

        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param *= 0.5

    def forward(self,
                x,
                t,
                clip_f,
                sync_f,
                text_f,
                inpaint_masked_input=None,
                t5_features=None,
                metaclip_global_text_features=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        # breakpoint()
        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        return self.model(
            latent=x,
            t=t,
            clip_f=clip_f,
            sync_f=sync_f,
            text_f=text_f,
            inpaint_masked_input=inpaint_masked_input,
            t5_features=t5_features,
            metaclip_global_text_features=metaclip_global_text_features,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            **kwargs)

class MMConditionedDiffusionModelWrapper(ConditionedDiffusionModel):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            add_cond_ids: tp.List[str] = [],
            mm_cond_ids: tp.List[str] = [],
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
        self.prepend_cond_ids = prepend_cond_ids
        self.add_cond_ids = add_cond_ids
        self.min_input_length = min_input_length
        self.mm_cond_ids = mm_cond_ids
        
        assert len(self.cross_attn_cond_ids) == 0, "cross_attn_cond_ids is not supported for MMDiTWrapper"
        assert len(self.global_cond_ids) == 0, "global_cond_ids is not supported for MMDiTWrapper"
        assert len(self.input_concat_ids) == 0, "input_concat_ids is not supported for MMDiTWrapper"
        assert len(self.prepend_cond_ids) == 0, "prepend_cond_ids is not supported for MMDiTWrapper"
        assert len(self.add_cond_ids) == 0, "add_cond_ids is not supported for MMDiTWrapper"
        assert len(self.mm_cond_ids) > 0, "mm_cond_ids must be specified for MMDiTWrapper"
        assert "metaclip_features" in self.mm_cond_ids, "clip_f must be specified in mm_cond_ids for MMDiTWrapper"
        assert "sync_features" in self.mm_cond_ids, "sync_features must be specified in mm_cond_ids for MMDiTWrapper"
        assert "metaclip_text_features" in self.mm_cond_ids, "metaclip_text_features must be specified in mm_cond_ids for MMDiTWrapper"
        # assert len(self.mm_cond_ids) == 3, "mm_cond_ids must be clip_f sync_f text_f for MMDiTWrapper"

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        assert negative == False, "negative conditioning is not supported for MMDiTWrapper"
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        add_input = None
        inpaint_masked_input = None
        t5_features = None
        metaclip_global_text_features = None
        clip_f = conditioning_tensors["metaclip_features"]
        sync_f = conditioning_tensors["sync_features"]
        text_f = conditioning_tensors["metaclip_text_features"]
        if 'inpaint_masked_input' in conditioning_tensors.keys():
            inpaint_masked_input = conditioning_tensors["inpaint_masked_input"]
        if 't5_features' in conditioning_tensors.keys():
            t5_features = conditioning_tensors["t5_features"]
        if 'metaclip_global_text_features' in conditioning_tensors.keys():
            metaclip_global_text_features = conditioning_tensors["metaclip_global_text_features"]
        return {
            "clip_f": clip_f,
            "sync_f": sync_f,
            "text_f": text_f,
            "inpaint_masked_input": inpaint_masked_input,
            "t5_features": t5_features,
            "metaclip_global_text_features": metaclip_global_text_features
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        # breakpoint()
        # print(kwargs)
        return self.model(x=x, t=t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        from ..inference.generation import generate_diffusion_cond
        return generate_diffusion_cond(self, *args, **kwargs)

class DiTUncondWrapper(DiffusionModel):
    def __init__(
        self,
        io_channels,
        *args,
        **kwargs
    ):
        super().__init__()

        self.model = DiffusionTransformer(io_channels=io_channels, *args, **kwargs)

        self.io_channels = io_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

def create_diffusion_uncond_from_config(config: tp.Dict[str, tp.Any]):
    diffusion_uncond_config = config["model"]

    model_type = diffusion_uncond_config.get('type', None)

    diffusion_config = diffusion_uncond_config.get('config', {})

    assert model_type is not None, "Must specify model type in config"

    pretransform = diffusion_uncond_config.get("pretransform", None)

    sample_size = config.get("sample_size", None)
    assert sample_size is not None, "Must specify sample size in config"

    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "Must specify sample rate in config"

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if model_type == 'DAU1d':

        model = DiffusionAttnUnet1D(
            **diffusion_config
        )
    
    elif model_type == "adp_uncond_1d":

        model = UNet1DUncondWrapper(
            **diffusion_config
        )

    elif model_type == "dit":
        model = DiTUncondWrapper(
            **diffusion_config
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

    return DiffusionModelWrapper(model,
                                io_channels=model.io_channels,
                                sample_size=sample_size,
                                sample_rate=sample_rate,
                                pretransform=pretransform,
                                min_input_length=min_input_length)

def create_diffusion_infill_from_config(config: tp.Dict[str, tp.Any]):
    diffusion_uncond_config = config["model"]


    diffusion_config = diffusion_uncond_config.get('diffusion', {})
    model_type = diffusion_config.get('type', None)
    model_config = diffusion_config.get("config",{})
    assert model_type is not None, "Must specify model type in config"

    pretransform = diffusion_uncond_config.get("pretransform", None)

    sample_size = config.get("sample_size", None)
    assert sample_size is not None, "Must specify sample size in config"

    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "Must specify sample rate in config"

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if model_type == 'DAU1d':

        model = DiffusionAttnUnet1D(
            **model_config
        )
    
    elif model_type == "adp_uncond_1d":

        model = UNet1DUncondWrapper(
            io_channels = io_channels,
            **model_config
        )
    elif model_type == "dit":
        model = DiTUncondWrapper(
            **model_config
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

    return DiffusionModelWrapper(model,
                                io_channels=model.io_channels,
                                sample_size=sample_size,
                                sample_rate=sample_rate,
                                pretransform=pretransform,
                                min_input_length=min_input_length)

def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'adp_cfg_1d':
        diffusion_model = UNetCFG1DWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'adp_1d':
        diffusion_model = UNet1DCondWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'mmdit':
        diffusion_model = MMDiTWrapper(**diffusion_model_config)

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    add_cond_ids = diffusion_config.get('add_cond_ids', [])
    sync_cond_ids = diffusion_config.get('sync_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])
    mm_cond_ids = diffusion_config.get('mm_cond_ids', [])
    zero_init = diffusion_config.get('zero_init', False)
    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "mm_diffusion_cond":
        wrapper_fn = MMConditionedDiffusionModelWrapper
        extra_kwargs["diffusion_objective"] = diffusion_objective
        extra_kwargs["mm_cond_ids"] = mm_cond_ids

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint" or model_type == 'diffusion_infill':
        wrapper_fn = ConditionedDiffusionModelWrapper
        extra_kwargs["diffusion_objective"] = diffusion_objective

    elif model_type == "diffusion_prior":
        prior_type = model_config.get("prior_type", None)
        assert prior_type is not None, "Must specify prior_type in diffusion prior model config"

        if prior_type == "mono_stereo":
            from .diffusion_prior import MonoToStereoDiffusionPrior
            wrapper_fn = MonoToStereoDiffusionPrior
            
    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        add_cond_ids=add_cond_ids,
        sync_cond_ids=sync_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        zero_init=zero_init,
        **extra_kwargs
    )
