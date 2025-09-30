import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import AttentionModuleMixin
from .attention import WanSparseAttnProcessor 
from .attn_mask import MaskMap

def setup_radial_attention(
    pipe,
    height,
    width,
    num_frames,
    dense_layers=0,
    dense_timesteps=0,
    decay_factor=1.0,
    sparsity_type="radial",
    use_sage_attention=False,
):

    num_frames = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_size = int(height // mod_value) * int(width // mod_value)
    
    AttnModule = WanSparseAttnProcessor
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.mask_map = MaskMap(video_token_num=frame_size * num_frames, num_frame=num_frames)
    AttnModule.decay_factor = decay_factor
    AttnModule.sparse_type = sparsity_type
    AttnModule.use_sage_attention = use_sage_attention
    
    print(f"Replacing Wan attention with {sparsity_type} attention")
    print(f"video token num: {AttnModule.mask_map.video_token_num}, num frames: {num_frames}")
    print(f"dense layers: {dense_layers}, dense timesteps: {dense_timesteps}, decay factor: {decay_factor}")
    
    for layer_idx, m in enumerate(pipe.transformer.blocks):
        m.attn1.processor.layer_idx = layer_idx
        
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, AttentionModuleMixin) and hasattr(m.processor, 'layer_idx'):
            layer_idx = m.processor.layer_idx
            m.set_processor(AttnModule(layer_idx))