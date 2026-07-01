import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class DiscreteBottleneck(Bottleneck):
    def __init__(self, num_quantizers, codebook_size, tokens_id):
        super().__init__(is_discrete=True)

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.tokens_id = tokens_id

    def decode_tokens(self, codes, **kwargs):
        raise NotImplementedError
    
class TanhBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)
        self.tanh = nn.Tanh()

    def encode(self, x, return_info=False):
        info = {}

        x = torch.tanh(x)

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def compute_mean_kernel(x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

def compute_mmd(latents):
    latents_reshaped = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
    noise = torch.randn_like(latents_reshaped)

    latents_kernel = compute_mean_kernel(latents_reshaped, latents_reshaped)
    noise_kernel = compute_mean_kernel(noise, noise)
    latents_noise_kernel = compute_mean_kernel(latents_reshaped, noise)
    
    mmd = latents_kernel + noise_kernel - 2 * latents_noise_kernel
    return mmd.mean()

class WassersteinBottleneck(Bottleneck):
    def __init__(self, noise_augment_dim: int = 0, bypass_mmd: bool = False):
        super().__init__(is_discrete=False)

        self.noise_augment_dim = noise_augment_dim
        self.bypass_mmd = bypass_mmd
    
    def encode(self, x, return_info=False):
        info = {}

        if self.training and return_info:
            if self.bypass_mmd:
                mmd = torch.tensor(0.0)
            else:
                mmd = compute_mmd(x)
                
            info["mmd"] = mmd
        
        if return_info:
            return x, info
        
        return x

    def decode(self, x):

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x

class L2Bottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)
    
    def encode(self, x, return_info=False):
        info = {}

        x = F.normalize(x, dim=1)

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return F.normalize(x, dim=1)
        
class RVQBottleneck(DiscreteBottleneck):
    def __init__(self, **quantizer_kwargs):
        super().__init__(num_quantizers = quantizer_kwargs["num_quantizers"], codebook_size = quantizer_kwargs["codebook_size"], tokens_id = "quantizer_indices")
        from vector_quantize_pytorch import ResidualVQ
        self.quantizer = ResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["num_quantizers"]

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        x = rearrange(x, "b c n -> b n c")
        x, indices, loss = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices
        info["quantizer_loss"] = loss.mean()

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return x
    
    def decode_tokens(self, codes, **kwargs):
        latents = self.quantizer.get_outputs_from_indices(codes)

        return self.decode(latents, **kwargs)
    
class RVQVAEBottleneck(DiscreteBottleneck):
    def __init__(self, **quantizer_kwargs):
        super().__init__(num_quantizers = quantizer_kwargs["num_quantizers"], codebook_size = quantizer_kwargs["codebook_size"], tokens_id = "quantizer_indices")
        from vector_quantize_pytorch import ResidualVQ
        self.quantizer = ResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["num_quantizers"]

    def encode(self, x, return_info=False):
        info = {}

        x, kl = vae_sample(*x.chunk(2, dim=1))

        info["kl"] = kl

        x = rearrange(x, "b c n -> b n c")
        x, indices, loss = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices
        info["quantizer_loss"] = loss.mean()

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):
        return x
    
    def decode_tokens(self, codes, **kwargs):
        latents = self.quantizer.get_outputs_from_indices(codes)

        return self.decode(latents, **kwargs)

class DACRVQBottleneck(DiscreteBottleneck):
    def __init__(self, quantize_on_decode=False, noise_augment_dim=0, **quantizer_kwargs):
        super().__init__(num_quantizers = quantizer_kwargs["n_codebooks"], codebook_size = quantizer_kwargs["codebook_size"], tokens_id = "codes")
        from dac.nn.quantize import ResidualVectorQuantize as DACResidualVQ
        self.quantizer = DACResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["n_codebooks"]
        self.quantize_on_decode = quantize_on_decode
        self.noise_augment_dim = noise_augment_dim

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        info["pre_quantizer"] = x

        if self.quantize_on_decode:
            return x, info if return_info else x

        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(x, **kwargs)

        output = {
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

        output["vq/commitment_loss"] /= self.num_quantizers
        output["vq/codebook_loss"] /= self.num_quantizers

        info.update(output)

        if return_info:
            return output["z"], info
        
        return output["z"]
    
    def decode(self, x):

        if self.quantize_on_decode:
            x = self.quantizer(x)[0]

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x
    
    def decode_tokens(self, codes, **kwargs):
        latents, _, _ = self.quantizer.from_codes(codes)

        return self.decode(latents, **kwargs)

class DACRVQVAEBottleneck(DiscreteBottleneck):
    def __init__(self, quantize_on_decode=False, **quantizer_kwargs):
        super().__init__(num_quantizers = quantizer_kwargs["n_codebooks"], codebook_size = quantizer_kwargs["codebook_size"], tokens_id = "codes")
        from dac.nn.quantize import ResidualVectorQuantize as DACResidualVQ
        self.quantizer = DACResidualVQ(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["n_codebooks"]
        self.quantize_on_decode = quantize_on_decode

    def encode(self, x, return_info=False, n_quantizers: int = None):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["pre_quantizer"] = x
        info["kl"] = kl

        if self.quantize_on_decode:
            return x, info if return_info else x

        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(x, n_quantizers=n_quantizers)

        output = {
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

        output["vq/commitment_loss"] /= self.num_quantizers
        output["vq/codebook_loss"] /= self.num_quantizers

        info.update(output)

        if return_info:
            return output["z"], info
        
        return output["z"]
    
    def decode(self, x):

        if self.quantize_on_decode:
            x = self.quantizer(x)[0]

        return x

    def decode_tokens(self, codes, **kwargs):
        latents, _, _ = self.quantizer.from_codes(codes)

        return self.decode(latents, **kwargs)
    
class FSQBottleneck(DiscreteBottleneck):
    def __init__(self, noise_augment_dim=0, **kwargs):
        super().__init__(num_quantizers = kwargs.get("num_codebooks", 1), codebook_size = np.prod(kwargs["levels"]), tokens_id = "quantizer_indices")

        self.noise_augment_dim = noise_augment_dim

        from vector_quantize_pytorch import FSQ
        self.quantizer = FSQ(**kwargs, allowed_dtypes=[torch.float16, torch.float32, torch.float64])

    def encode(self, x, return_info=False):
        info = {}

        orig_dtype = x.dtype
        x = x.float()

        x = rearrange(x, "b c n -> b n c")
        x, indices = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        x = x.to(orig_dtype)

        # Reorder indices to match the expected format
        indices = rearrange(indices, "b n q -> b q n")

        info["quantizer_indices"] = indices

        if return_info:
            return x, info
        else:
            return x
        
    def decode(self, x):

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x
    
    def decode_tokens(self, tokens, **kwargs):
        latents = self.quantizer.indices_to_codes(tokens)

        return self.decode(latents, **kwargs)
