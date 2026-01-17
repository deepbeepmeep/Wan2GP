import torch
import torch.nn as nn

from einops import rearrange
from diffusers import ConfigMixin, ModelMixin


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768,  
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=True,
        enable_compile=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()
        self.flops = 0.0
        self.enable_compile = enable_compile

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        B1, _ = audio_embeds.shape
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        if not self.enable_compile:
            self.flops += B1 * self.input_dim * self.intermediate_dim * 2

        B1_vf, _ = audio_embeds_vf.shape
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        if not self.enable_compile:
            self.flops += B1_vf * self.input_dim_vf * self.intermediate_dim * 2 

        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        B2, _ = audio_embeds_c.shape
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))
        if not self.enable_compile:
            self.flops += B2 * self.intermediate_dim * self.intermediate_dim * 2 

        # third projection
        B3, _ = audio_embeds_c.shape
        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)
        if not self.enable_compile:
            self.flops += B3 * self.intermediate_dim * (self.context_tokens * self.output_dim) * 2 

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens