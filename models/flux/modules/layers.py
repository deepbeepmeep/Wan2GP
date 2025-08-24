import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope

def get_linear_split_map():
    hidden_size = 3072
    split_linear_modules_map =  {
                                "qkv" : {"mapped_modules" : ["q", "k", "v"] , "split_sizes": [hidden_size, hidden_size, hidden_size]},
                                "linear1" : {"mapped_modules" : ["linear1_attn_q", "linear1_attn_k", "linear1_attn_v", "linear1_mlp"] , "split_sizes":  [hidden_size, hidden_size, hidden_size, 7*hidden_size- 3*hidden_size]}
                                }
    return split_linear_modules_map


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale



class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if k != None:
            return self.key_norm(k).to(v)
        else: 
            return self.query_norm(q).to(v)
        # q = self.query_norm(q)
        # k = self.key_norm(k)
        # return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        raise Exception("not implemented")

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0):
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )


def split_mlp(mlp, x, divide = 8):
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    chunk_size = int(x.shape[0]/divide)
    chunk_size = int(x_shape[1]/divide)
    x_chunks = torch.split(x, chunk_size)
    for i, x_chunk  in enumerate(x_chunks):
        mlp_chunk = mlp[0](x_chunk)
        mlp_chunk = mlp[1](mlp_chunk)
        x_chunk[...] = mlp[2](mlp_chunk)
    return x.reshape(x_shape)      

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, chroma_modulation = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.chroma_modulation = chroma_modulation
        if not chroma_modulation:
            self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        if not chroma_modulation:
            self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        if self.chroma_modulation:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated.mul_(1 + img_mod1.scale)
        img_modulated.add_(img_mod1.shift)

        shape = (*img_modulated.shape[:2], self.num_heads, int(img_modulated.shape[-1] / self.num_heads) )
        img_q = self.img_attn.q(img_modulated).view(*shape).transpose(1,2)
        img_k = self.img_attn.k(img_modulated).view(*shape).transpose(1,2) 
        img_v = self.img_attn.v(img_modulated).view(*shape).transpose(1,2)
        del img_modulated


        img_q= self.img_attn.norm(img_q, None, img_v)
        img_k = self.img_attn.norm(None, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated.mul_(1 + txt_mod1.scale)
        txt_modulated.add_(txt_mod1.shift)

        shape = (*txt_modulated.shape[:2], self.num_heads, int(txt_modulated.shape[-1] / self.num_heads) )
        txt_q = self.txt_attn.q(txt_modulated).view(*shape).transpose(1,2)
        txt_k = self.txt_attn.k(txt_modulated).view(*shape).transpose(1,2) 
        txt_v = self.txt_attn.v(txt_modulated).view(*shape).transpose(1,2)
        del txt_modulated


        txt_q = self.txt_attn.norm(txt_q, None, txt_v)
        txt_k = self.txt_attn.norm(None, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        qkv_list = [q, k, v]
        del q, k, v
        attn = attention(qkv_list, pe=pe)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img.addcmul_(self.img_attn.proj(img_attn), img_mod1.gate)
        mod_img = self.img_norm2(img)
        mod_img.mul_(1 + img_mod2.scale)
        mod_img.add_(img_mod2.shift)
        mod_img = split_mlp(self.img_mlp, mod_img)
        # mod_img = self.img_mlp(mod_img)
        img.addcmul_( mod_img, img_mod2.gate)
        mod_img = None

        # calculate the txt blocks
        txt.addcmul_(self.txt_attn.proj(txt_attn), txt_mod1.gate)
        txt.addcmul_(self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift), txt_mod2.gate)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        chroma_modulation = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.chroma_modulation = chroma_modulation
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        if not chroma_modulation:
            self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        if self.chroma_modulation:
            mod = vec
        else:
            mod, _ = self.modulation(vec)
        x_mod = self.pre_norm(x)
        x_mod.mul_(1 + mod.scale)
        x_mod.add_(mod.shift)

        ##### More spagheti VRAM optimizations done by DeepBeepMeep !
        # I am sure you are a nice person and as you copy this code, you will give me proper credits:
        # Please link to https://github.com/deepbeepmeep/Wan2GP and @deepbeepmeep on twitter  

        # x_mod = (1 + mod.scale) * x + mod.shift

        shape = (*x_mod.shape[:2], self.num_heads, int(x_mod.shape[-1] / self.num_heads) )
        q = self.linear1_attn_q(x_mod).view(*shape).transpose(1,2)
        k = self.linear1_attn_k(x_mod).view(*shape).transpose(1,2)
        v = self.linear1_attn_v(x_mod).view(*shape).transpose(1,2)

        q = self.norm(q, None, v)
        k = self.norm(None, k, v)

        # compute attention
        qkv_list = [q, k, v]
        del q, k, v
        attn = attention(qkv_list, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer

        x_mod_shape = x_mod.shape
        x_mod = x_mod.view(-1, x_mod.shape[-1])
        chunk_size = int(x_mod_shape[1]/6)
        x_chunks = torch.split(x_mod, chunk_size)
        attn = attn.view(-1, attn.shape[-1])
        attn_chunks =torch.split(attn, chunk_size)
        for x_chunk, attn_chunk in zip(x_chunks, attn_chunks):
            mlp_chunk = self.linear1_mlp(x_chunk)
            mlp_chunk = self.mlp_act(mlp_chunk)
            attn_mlp_chunk = torch.cat((attn_chunk, mlp_chunk), -1)
            del attn_chunk, mlp_chunk 
            x_chunk[...] = self.linear2(attn_mlp_chunk)
            del attn_mlp_chunk
        x_mod = x_mod.view(x_mod_shape)
        x.addcmul_(x_mod, mod.gate)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, chroma_modulation = False):
        super().__init__()
        self.chroma_modulation = chroma_modulation
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if not chroma_modulation:        
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if self.chroma_modulation:
            shift, scale = vec
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)            
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = torch.addcmul(shift[:, None, :], 1 + scale[:, None, :], self.norm_final(x))
        x = self.linear(x)
        return x


class DistilledGuidance(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range( n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)


    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x