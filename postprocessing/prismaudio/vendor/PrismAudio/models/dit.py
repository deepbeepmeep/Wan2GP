import typing as tp
import math
import torch
# from beartype.typing import Tuple
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from .mmmodules.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from .blocks import FourierFeatures
from .transformer import ContinuousTransformer
from .utils import mask_from_frac_lengths, resample

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        cond_ctx_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        add_token_dim=0,
        sync_token_dim=0,
        use_mlp=False,
        use_zero_init=False,
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256
        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type
        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.use_mlp = use_mlp
        if cond_token_dim > 0:
            # Conditioning tokens
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )
        if add_token_dim > 0:
            # Conditioning tokens
            add_embed_dim = add_token_dim if not project_cond_tokens else embed_dim
            self.to_add_embed = nn.Sequential(
                nn.Linear(add_token_dim, add_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(add_embed_dim, add_embed_dim, bias=False)
            )
        else:
            add_embed_dim = 0
        
        if sync_token_dim > 0:
            # Conditioning tokens
            sync_embed_dim = sync_token_dim if not project_cond_tokens else embed_dim
            self.to_sync_embed = nn.Sequential(
                nn.Linear(sync_token_dim, sync_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(sync_embed_dim, sync_embed_dim, bias=False)
            )
        else:
            sync_embed_dim = 0


        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.empty_clip_feat = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)
        self.global_cond_type = global_cond_type
        print("######################")
        print(f'global type: {global_cond_type}')
        print("######################")
        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        nn.init.zeros_(self.postprocess_conv.weight)


    def initialize_weights(self):
        print("######################")
        print(f'Fine! You are using zero initialization!')
        print("######################")
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            # if isinstance(module, nn.Conv1d):
            #     if module.bias is not None:
            #         nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.to_timestep_embed[0].weight, std=0.02)
        nn.init.normal_(self.to_timestep_embed[2].weight, std=0.02)

        # Zero-out output layers:
        if self.global_cond_type == "adaLN":
            for block in self.transformer.layers:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        add_cond=None,
        add_masks=None,
        sync_cond=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)
            
        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # reshape from (b, n, c) to (b, c, n)
            if input_concat_cond.shape[1] != x.shape[1]:
                input_concat_cond = input_concat_cond.transpose(1,2)
            # Interpolate input_concat_cond to the same length as x
            # if input_concat_cond.shape[1] != x.shape[2]:
            # input_concat_cond = input_concat_cond.transpose(1,2)
            input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')
            # input_concat_cond = input_concat_cond.transpose(1,2)
            # if len(global_embed.shape) == 2:
            #     global_embed = global_embed.unsqueeze(1)
            # global_embed = global_embed + input_concat_cond
            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)
        # import ipdb
        # ipdb.set_trace()
        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if self.timestep_cond_type == "global":
            if global_embed is not None:
                if len(global_embed.shape) == 3:
                    timestep_embed = timestep_embed.unsqueeze(1)
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                if len(global_embed.shape) == 2:
                    prepend_inputs = global_embed.unsqueeze(1)
                else:
                    prepend_inputs = global_embed
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                if len(global_embed.shape) == 2:
                    prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                else:
                    prepend_inputs = torch.cat([prepend_inputs, global_embed], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            b, seq_len, c = x.shape
            
            # 计算需要填充的数量
            pad_amount = (self.patch_size - seq_len % self.patch_size) % self.patch_size
            
            if pad_amount > 0:
                # 在时间维度上进行填充
                x = F.pad(x, (0, 0, 0, pad_amount), mode='constant', value=0)
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if add_cond is not None:
            # Interpolate add_cond to the same length as x
            # if self.use_mlp:
            add_cond = self.to_add_embed(add_cond)
            if add_cond.shape[1] != x.shape[1]:
                add_cond = add_cond.transpose(1,2)
                add_cond = F.interpolate(add_cond, (x.shape[1], ), mode='linear', align_corners=False)
                add_cond = add_cond.transpose(1,2)
                # add_cond = resample(add_cond, x)

        if sync_cond is not None:
            sync_cond = self.to_sync_embed(sync_cond)

        if self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, add_cond=add_cond, sync_cond=sync_cond, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)
            # 移除之前添加的填充
            if pad_amount > 0:
                output = output[:, :, :seq_len]

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self, 
        x, 
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        add_cond=None,
        sync_cond=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        **kwargs):

        assert causal == False, "Causal mode is not supported for DiffusionTransformer"
        bsz, a, b = x.shape
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if add_cond is not None:
            add_cond = add_cond.to(model_dtype)
        
        if sync_cond is not None:
            sync_cond = sync_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()
        

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)
            
            if add_cond is not None:
                null_embed = torch.zeros_like(add_cond, device=add_cond.device)
                dropout_mask = torch.bernoulli(torch.full((add_cond.shape[0], 1, 1), cfg_dropout_prob, device=add_cond.device)).to(torch.bool)
                add_cond = torch.where(dropout_mask, null_embed, add_cond)
            
            if sync_cond is not None:
                null_embed = torch.zeros_like(sync_cond, device=sync_cond.device)
                dropout_mask = torch.bernoulli(torch.full((sync_cond.shape[0], 1, 1), cfg_dropout_prob, device=sync_cond.device)).to(torch.bool)
                sync_cond = torch.where(dropout_mask, null_embed, sync_cond)

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None or add_cond is not None):
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)
            if global_embed is not None and global_embed.shape[0] == bsz:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            elif global_embed is not None:
                batch_global_cond = global_embed
            else:
                batch_global_cond = None

            if input_concat_cond is not None and input_concat_cond.shape[0] == bsz:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            elif input_concat_cond is not None:
                batch_input_concat_cond = input_concat_cond
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None and cross_attn_cond.shape[0] == bsz:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
            elif cross_attn_cond is not None:
                batch_cond = cross_attn_cond
            else:
                batch_cond = None

            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None and prepend_cond.shape[0] == bsz:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
            elif prepend_cond is not None:
                batch_prepend_cond = prepend_cond
            else:
                batch_prepend_cond = None

            batch_add_cond = None
            
            # Handle CFG for cross-attention conditioning
            if add_cond is not None and add_cond.shape[0] == bsz:

                null_embed = torch.zeros_like(add_cond, device=add_cond.device)

                
                batch_add_cond = torch.cat([add_cond, null_embed], dim=0)
            elif add_cond is not None:
                batch_add_cond = add_cond
            else:
                batch_add_cond = None
            
            batch_sync_cond = None
            
            # Handle CFG for cross-attention conditioning
            if sync_cond is not None and sync_cond.shape[0] == bsz:

                null_embed = torch.zeros_like(sync_cond, device=sync_cond.device)

                
                batch_sync_cond = torch.cat([sync_cond, null_embed], dim=0)
            elif sync_cond is not None:
                batch_sync_cond = sync_cond
            else:
                batch_sync_cond = None

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                input_concat_cond=batch_input_concat_cond, 
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                add_cond = batch_add_cond,
                sync_cond = batch_sync_cond,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
            
            if return_info:
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                add_cond=add_cond,
                sync_cond=sync_cond,
                mask=mask,
                return_info=return_info,
                **kwargs
            )