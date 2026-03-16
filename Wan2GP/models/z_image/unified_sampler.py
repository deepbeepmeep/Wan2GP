import torch
from torch import nn
from typing import Callable, Union


class Linear:
    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1


class UnifiedSampler(torch.nn.Module):
    """
    UCGM-S: https://arxiv.org/abs/2505.07447
    Credit to https://github.com/LINs-lab/UCGM/blob/main/methodes/unigen.py
    """

    def __init__(self):
        super().__init__()

        transport = Linear()
        self.alpha_in, self.gamma_in = transport.alpha_in, transport.gamma_in
        self.alpha_to, self.gamma_to = transport.alpha_to, transport.gamma_to

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
            self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
            self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def forward(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        tt = tt.flatten()
        dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        q = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        q = q if self.integ_st == 1 else 1 - q
        F_t = (-1) ** (1 - self.integ_st) * model(x_t, t=q, tt=tt, **model_kwargs)
        t = torch.abs(t)
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c
