from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.amp as amp

import numpy as np 
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from safetensors.torch import load_file

from .lora_utils import create_lora_network
from .attention import Attention, MultiHeadCrossAttention
from .blocks import TimestepEmbedder, CaptionEmbedder, PatchEmbed3D, FeedForwardSwiGLU, FinalLayer_FP32, LayerNorm_FP32, modulate_fp32


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))
        self.ffn_mult = self.ffn.ffn_mult
        self.ffn_chunk_min = 128

    def _apply_ffn_chunked(self, ffn_in: torch.Tensor) -> torch.Tensor:
        _, seq_len, dim = ffn_in.shape
        if seq_len < self.ffn_chunk_min:
            return self.ffn(ffn_in)
        ffn_in_flat = ffn_in.reshape(-1, dim)
        chunk_size = max(int(seq_len // self.ffn_mult), 1)
        if chunk_size >= ffn_in_flat.shape[0]:
            return self.ffn(ffn_in)
        for ffn_chunk in torch.split(ffn_in_flat, chunk_size, dim=0):
            ffn_chunk[...] = self.ffn(ffn_chunk)
        return ffn_in

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1) # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv)
        
        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        # cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self._apply_ffn_chunked(x_m)
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoTransformer3DModel(
    ModelMixin, ConfigMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict = {}
        self.active_loras = []
        self._interrupt_check = None
    
    def load_lora(self, lora_path, lora_key, multiplier=1.0, lora_network_dim=128, lora_network_alpha=64):
        lora_network_state_dict_loaded = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict_loaded,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )
        
        lora_network.load_state_dict(lora_network_state_dict_loaded, strict=True)
        
        self.lora_dict[lora_key] = lora_network

    def enable_loras(self, lora_key_list=[]):
        self.disable_all_loras()
    
        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        
        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)
    
        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, 'org_forward'):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)
    
    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output
            
            return org_output + total_lora_output
        
        return multi_lora_forward
    
    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")
    
    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, 'org_forward'):
                module.forward = module.org_forward
                delattr(module, 'org_forward')
        
        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")
        
        self.active_loras.clear()

    def enable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = True
    
    def disable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = False    

    def forward(
        self, 
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask=None, 
        num_cond_latents=0,
        return_kv=False, 
        kv_cache_dict={},
        skip_crs_attn=False, 
        offload_kv_cache=False
    ):
        x_list = hidden_states if isinstance(hidden_states, list) else [hidden_states]
        joint_pass = isinstance(hidden_states, list)

        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states] * len(x_list)
        if not isinstance(encoder_attention_mask, list):
            encoder_attention_mask = [encoder_attention_mask] * len(x_list)
        if not isinstance(timestep, list):
            timestep = [timestep] * len(x_list)
        if not isinstance(num_cond_latents, list):
            num_cond_latents = [num_cond_latents] * len(x_list)
        if not isinstance(kv_cache_dict, list):
            kv_cache_dict = [kv_cache_dict] * len(x_list)

        dtype = self.x_embedder.proj.weight.dtype
        t_list = []
        enc_list = []
        y_seqlens_list = []
        latent_shapes = []
        for idx, (x, step, enc, mask) in enumerate(zip(x_list, timestep, encoder_hidden_states, encoder_attention_mask)):
            B, _, T, H, W = x.shape
            N_t = T // self.patch_size[0]
            N_h = H // self.patch_size[1]
            N_w = W // self.patch_size[2]

            assert self.patch_size[0] == 1, "Currently, 3D x_embedder should not compress the temporal dimension."

            if len(step.shape) == 1:
                step = step.unsqueeze(1).expand(-1, N_t)

            x = x.to(dtype)
            step = step.to(dtype)
            enc = enc.to(dtype)

            x = self.x_embedder(x)

            with amp.autocast(device_type='cuda', dtype=torch.float32):
                t = self.t_embedder(step.float().flatten(), torch.float32).reshape(B, N_t, -1)
            t = t.to(torch.float32)

            enc = self.y_embedder(enc)

            if self.text_tokens_zero_pad and mask is not None:
                enc = enc * mask[:, None, :, None]

            if mask is not None:
                if mask.dim() > 2:
                    mask = mask.squeeze(1).squeeze(1)
                y_seqlens = mask.sum(dim=1).to(torch.int64).tolist()
            else:
                y_seqlens = [enc.shape[2]] * enc.shape[0]

            enc = enc.squeeze(1)

            x_list[idx] = x
            t_list.append(t)
            enc_list.append(enc)
            y_seqlens_list.append(y_seqlens)
            latent_shapes.append((N_t, N_h, N_w))

        kv_cache_dict_ret = [dict() for _ in x_list] if return_kv else None
        for block_idx, block in enumerate(self.blocks):
            if self._interrupt_check is not None and self._interrupt_check():
                return [None] * len(x_list) if joint_pass else None
            for i in range(len(x_list)):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    block_outputs = self._gradient_checkpointing_func(
                        block, x_list[i], enc_list[i], t_list[i], y_seqlens_list[i],
                        latent_shapes[i], num_cond_latents[i], return_kv, kv_cache_dict[i].get(block_idx, None) if kv_cache_dict[i] is not None else None, skip_crs_attn
                    )
                else:
                    block_outputs = block(
                        x_list[i], enc_list[i], t_list[i], y_seqlens_list[i],
                        latent_shapes[i], num_cond_latents[i], return_kv, kv_cache_dict[i].get(block_idx, None) if kv_cache_dict[i] is not None else None, skip_crs_attn
                    )
                if return_kv:
                    x_list[i], kv_cache = block_outputs
                    if offload_kv_cache:
                        kv_cache_dict_ret[i][block_idx] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                    else:
                        kv_cache_dict_ret[i][block_idx] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
                else:
                    x_list[i] = block_outputs

                if self._interrupt_check is not None and self._interrupt_check():
                    return [None] * len(x_list) if joint_pass else None

        outputs = []
        for x, t, latent_shape in zip(x_list, t_list, latent_shapes):
            x = self.final_layer(x, t, latent_shape)
            x = self.unpatchify(x, *latent_shape)
            outputs.append(x.to(torch.float32))

        if return_kv:
            return (outputs if joint_pass else outputs[0]), (kv_cache_dict_ret if joint_pass else kv_cache_dict_ret[0])
        return outputs if joint_pass else outputs[0]
    

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x
