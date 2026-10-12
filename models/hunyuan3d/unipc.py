# UniPC multistep sampler (Zhao et al. 2023), the path Hunyuan3D 2.1 paint uses of diffusers' UniPCMultistepScheduler
# (diffusers/schedulers/scheduling_unipc_multistep.py, Apache-2.0, Copyright 2025 TSAIL Team and The HuggingFace Team):
# scaled-linear betas rescaled to zero terminal SNR, "trailing" timesteps, v-prediction converted to x0, B(h) = expm1 (bh2), order 2
# with lower order at the end, corrector at every step after the first. Ported for WanGP with the same arithmetic and devices: the
# sigmas stay on the CPU and the solver coefficients are built on the sample's device, as upstream.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch


class UniPCSampler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, solver_order=2, steps_offset=1, **_):
        self.num_train_timesteps, self.solver_order = num_train_timesteps, solver_order
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32, device="cpu") ** 2
        betas = self._rescale_zero_terminal_snr(betas)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        alphas_cumprod[-1] = 2 ** -24  # close to 0 without being 0 so that the first sigma is finite (as diffusers)
        self.alphas_cumprod = alphas_cumprod

    @staticmethod
    def _rescale_zero_terminal_snr(betas):
        alphas_bar_sqrt = torch.cumprod(1.0 - betas, dim=0).sqrt()
        first, last = alphas_bar_sqrt[0].clone(), alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= last
        alphas_bar_sqrt *= first / (first - last)
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
        return 1 - alphas

    def set_timesteps(self, steps, device):
        timesteps = np.arange(self.num_train_timesteps, 0, -self.num_train_timesteps / steps).round().copy().astype(np.int64) - 1
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0]]).astype(np.float32))
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums, self.last_sample, self.step_index, self.this_order = 0, None, 0, None
        return self.timesteps

    @staticmethod
    def _alpha_sigma(sigma):
        alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
        return alpha_t, sigma * alpha_t

    def _lambda(self, index):
        alpha, sigma = self._alpha_sigma(self.sigmas[index])
        return torch.log(alpha) - torch.log(sigma)

    def _coefficients(self, order, rks, h, device):
        rks = torch.tensor(rks, device=device)
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i, B_h = 1, torch.expm1(hh)
        R, b = [], []
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        return torch.stack(R), torch.tensor(b, device=device), h_phi_1, B_h

    def _predictor(self, x, order):
        """multistep_uni_p_bh_update (predict_x0, bh2)."""
        m0, s = self.model_outputs[-1], self.step_index
        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[s + 1])
        _, sigma_s0 = self._alpha_sigma(self.sigmas[s])
        lambda_s0 = self._lambda(s)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - lambda_s0
        rks, D1s = [], []
        for i in range(1, order):
            rk = (self._lambda(s - i) - lambda_s0) / h
            rks.append(rk)
            D1s.append((self.model_outputs[-(i + 1)] - m0) / rk)
        rks.append(1.0)
        R, b, h_phi_1, B_h = self._coefficients(order, rks, h, x.device)
        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=x.device) if order == 2 else torch.linalg.solve(R[:-1, :-1], b[:-1]).to(x.device).to(x.dtype)
            x_t = x_t - alpha_t * B_h * torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            x_t = x_t - alpha_t * B_h * 0
        return x_t.to(x.dtype)

    def _corrector(self, model_t, x, x_t, order):
        """multistep_uni_c_bh_update (predict_x0, bh2)."""
        m0, s = self.model_outputs[-1], self.step_index
        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[s])
        _, sigma_s0 = self._alpha_sigma(self.sigmas[s - 1])
        lambda_s0 = self._lambda(s - 1)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - lambda_s0
        rks, D1s = [], []
        for i in range(1, order):
            rk = (self._lambda(s - (i + 1)) - lambda_s0) / h
            rks.append(rk)
            D1s.append((self.model_outputs[-(i + 1)] - m0) / rk)
        rks.append(1.0)
        R, b, h_phi_1, B_h = self._coefficients(order, rks, h, x_t.device)
        D1s = torch.stack(D1s, dim=1) if D1s else None
        rhos_c = torch.tensor([0.5], dtype=x.dtype, device=x_t.device) if order == 1 else torch.linalg.solve(R, b).to(x_t.device).to(x.dtype)
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
        return (x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * (model_t - m0))).to(x.dtype)

    def step(self, model_output, sample):
        """One UniPC step from v-prediction model_output at the current timestep; returns the previous sample."""
        alpha_t, sigma_t = self._alpha_sigma(self.sigmas[self.step_index])
        x0_pred = alpha_t * sample - sigma_t * model_output
        if self.step_index > 0 and self.last_sample is not None:
            sample = self._corrector(x0_pred, self.last_sample, sample, self.this_order)
        self.model_outputs = self.model_outputs[1:] + [x0_pred]
        self.this_order = min(min(self.solver_order, len(self.timesteps) - self.step_index), self.lower_order_nums + 1)
        self.last_sample = sample
        prev_sample = self._predictor(sample, self.this_order)
        self.lower_order_nums = min(self.lower_order_nums + 1, self.solver_order)
        self.step_index += 1
        return prev_sample
