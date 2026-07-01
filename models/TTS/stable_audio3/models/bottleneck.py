import torch
from torch import nn

class SoftNormBottleneck(nn.Module):
    def __init__(self, dim = 32, noise_augment_dim=0, noise_regularize = False, auto_scale = False, freeze = False):
        super().__init__()

        self.noise_augment_dim = noise_augment_dim
        self.scaling_factor = nn.Parameter(torch.ones(1,dim,1))
        self.bias = nn.Parameter(torch.zeros(1,dim,1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1,noise_augment_dim,1))
        self.noise_regularize = noise_regularize
        self.freeze = freeze
        if self.freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            running_std = torch.ones(1)
            self.register_parameter("running_std", nn.Parameter(running_std, requires_grad=False))

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        x = x * self.scaling_factor + self.bias

        if self.training and hasattr(self,"running_std") and not self.freeze:
            # Update running std
            self.running_std.data = (self.running_std.data * 0.999 + x.std().detach() * 0.001).clamp(min = 1e-4)

        if hasattr(self, "running_std"):
            x = x / self.running_std.to(device=x.device, dtype=x.dtype)

        if self.training and return_info:
            var = (x.std(dim=-1) ** 2).clip(min = 1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-1)
            loss = (mean * mean + var - logvar - 1).mean()
            var = (x.std(dim=-2) ** 2).clip(min = 1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-2)
            loss = loss + 0.4 * (mean * mean + var - logvar - 1).mean()
            info["softnorm_loss"] = loss

        if return_info:
            return x, info

        return x

    def decode(self, x, **kwargs):
        if hasattr(self, "running_std"):
            running_std = self.running_std.to(device=x.device, dtype=x.dtype)
            x = x * running_std

        if self.noise_regularize:
            if hasattr(self, "running_std"):
                scaling = self.running_std.to(device=x.device, dtype=x.dtype)
            else:
                scaling = x.std(dim = -1).unsqueeze(-1)
            if self.training:
                scale = 5e-2
            else:
                scale = 1e-3
            noise = torch.randn_like(x) * scaling * scale
            x = x + noise

        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x

