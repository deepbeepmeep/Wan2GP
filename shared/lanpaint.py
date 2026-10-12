# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# LanPaint inpainting sampler, implemented from the method described in Zheng et al., "LanPaint: Training-Free Diffusion
# Inpainting with Asymptotically Exact and Fast Conditional Sampling", arXiv:2502.03491: before each sampler step, a few
# Langevin iterations driven by the Bidirectional Guided (BiG) score (Eq. 12-13) refine the inpainted region together with
# the known region, using Fast Langevin Dynamics (Eq. 15) solved exactly as a stochastic harmonic oscillator (Algorithms 1-4).
import cmath
import torch
import torch.nn.functional as F


def _phi(z):
    return (cmath.exp(z) - 1) / z if abs(z) > 1e-6 else 1 + z / 2 + z * z / 6


def _oscillator(a, g, f):
    """Exact Gaussian transition of dz = q dt, dq = Gamma (-q - A (z - m)) dt + Gamma sqrt(2) dW over f * tau (paper Eq. 111-118),
    with a = A tau and g = Gamma tau. Units: u = (z - m) sqrt(A) has unit stationary variance and p = q tau sqrt(A).
    Returns the mean coefficients (uu, up, pu, pp) and the noise factors (su, c, sp): u' = uu u + up p + su n1, p' = pu u + pp p + c n1 + sp n2."""
    d = 1 - 4 * a / g
    d = d if abs(d) > 1e-8 else 1e-8  # critical damping is the continuous limit of both regimes
    root = cmath.sqrt(d)  # complex when underdamped: the same closed form stays exact
    l1, l2 = -f * g * (1 - root) / 2, -f * g * (1 + root) / 2
    e1, e2 = cmath.exp(l1), cmath.exp(l2)
    dl = l1 - l2
    uu, up = (l1 * e2 - l2 * e1) / dl, f * (e1 - e2) / dl
    pu, pp = -g * a * f * (e1 - e2) / dl, (l1 * e1 - l2 * e2) / dl
    var_u = (2 * f * a / d * (_phi(2 * l1) - 2 * _phi(l1 + l2) + _phi(2 * l2))).real
    cov = (a / d * (e1 - e2) ** 2).real
    var_p = (2 * a / (f * d) * (l1 * (e1 * e1 - 1) / 2 - 2 * l1 * l2 * (e1 * e2 - 1) / (l1 + l2) + l2 * (e2 * e2 - 1) / 2)).real
    su = var_u ** 0.5
    c = cov / su
    return uu.real, up.real, pu.real, pp.real, su, c, max(var_p - c * c, 0.0) ** 0.5


class LanPaint:
    """Inpainting refinement called before a sampler step. latent_mask is 1 where content is generated and 0 where the
    latent_image is kept; sigma is the flow-matching noise level in [0, 1] (variance exploding sigma when neither IS_FLUX nor
    IS_FLOW is set, the model then predicting the noise). Lambda is the BiG guidance scale, StepSize the Langevin step,
    Friction the momentum damping and Beta scales the step of the known region. cfg_BIG is the guidance scale of the
    prediction used by the BiG term (forced to 1 for Flux style models). The exact oscillator solution covers every
    damping regime and the update is elementwise, so overdamped_fallback, height, width and vae_scale_factor are accepted
    for compatibility only."""

    def __init__(self, NSteps=5, Friction=15, Lambda=8, Beta=1, StepSize=0.15, IS_FLUX=True, IS_FLOW=False, overdamped_fallback=False, force_cfg_big_for_flux=True, MaxTotalStepSize=None):
        self.n_steps, self.friction, self.lam, self.beta, self.step_size = NSteps, Friction, Lambda, Beta, StepSize
        self.flow, self.force_big, self.max_total = IS_FLUX or IS_FLOW, IS_FLUX and force_cfg_big_for_flux, MaxTotalStepSize

    def __call__(self, denoise, cfg_predictions, true_cfg_scale, cfg_BIG, x, latent_image, noise, sigma, latent_mask, n_steps=None, height=720, width=1280, vae_scale_factor=8):
        steps = self.n_steps if n_steps is None else n_steps
        eta = self.step_size if self.max_total is None else min(self.step_size, self.max_total / steps)
        big_scale, lam = 1.0 if self.force_big else cfg_BIG, self.lam
        # Per region, a = A dtau and g = Gamma dtau with Gamma = Friction^2 A and dtau proportional to 1 / A (Eq. 126-127,
        # Algorithm 4): the known region has A = (1 + Lambda) / (1 - alpha_bar), its stationary std shrinks by sqrt(1 + Lambda).
        a_y = eta * (1 + lam) * self.beta
        regions = (eta, self.friction ** 2 * eta, 1.0), (a_y, self.friction ** 2 * a_y, (1 + lam) ** -0.5)
        f32 = dict(device=x.device, dtype=torch.float32)
        s = sigma.to(**f32).reshape(-1, *(1,) * (x.dim() - 1))
        signal = 1 - s if self.flow else 1.0
        mask = latent_mask.to(**f32)

        def pick(value_x, value_y):
            return mask * (value_x - value_y) + value_y

        clean = latent_image.to(**f32) * signal
        r = torch.lerp(clean + s * noise.to(**f32), x.to(**f32), mask)  # the known region restarts from its pre-diffused value

        def target(r):  # mean m = C / A of the oscillator, from the BiG score at r (Algorithm 4, Ccoef)
            cond, uncond = denoise(r, true_cfg_scale)
            if cond is None:
                return None
            v = cfg_predictions(cond, uncond, true_cfg_scale, sigma).to(**f32)
            v_big = v if big_scale == true_cfg_scale else cfg_predictions(cond, uncond, big_scale, sigma).to(**f32)
            del cond, uncond
            # flow matching: r = (1 - s) x0 + s eps and v = eps - x0; variance exploding: r = x0 + s eps and v = eps
            x0, eps = (r - s * v, r + (1 - s) * v_big) if self.flow else (r - s * v, v_big)
            return torch.lerp(clean + (lam / (1 + lam) * s) * eps, signal * x0, mask)

        def oscillate(r, p, m, f):
            cx, cy = (_oscillator(a, g, f) for a, g, _ in regions)
            uu, up, pu, pp = (pick(vx, vy) for vx, vy in zip(cx[:4], cy[:4]))
            su, c, sp = (pick(vx, vy * regions[1][2]) * s for vx, vy in zip(cx[4:], cy[4:]))
            u, n1 = r - m, torch.randn_like(r)
            return m + uu * u + up * p + su * n1, pu * u + pp * p + c * n1 + sp * torch.randn_like(r)

        m = None
        for _ in range(steps):
            if m is None:  # first order step from the input state, momentum drawn from its stationary law (Algorithms 1-2)
                m = target(r)
                if m is None:
                    return None
                p = pick(*((a * g) ** 0.5 * k for a, g, k in regions)) * s * torch.randn_like(r)
                r, p = oscillate(r, p, m, 1.0)
            else:  # second order Strang splitting, one model evaluation at the midpoint (Algorithm 3)
                r, p = oscillate(r, p, m, 0.5)
                m_new = target(r)
                if m_new is None:
                    return None
                p += pick(*(a * g for a, g, _ in regions)) * (m_new - m)
                r, p = oscillate(r, p, m, 0.5)
                m = m_new
        return r


def _pack_latents(latents):
    b, c, _, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack_latents(latents, height, width, vae_scale_factor=8):
    b, _, c = latents.shape
    h, w = int(height) // (vae_scale_factor * 2), int(width) // (vae_scale_factor * 2)
    return latents.reshape(b, h, w, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c // 4, 1, h * 2, w * 2)


def blend_images_with_mask(image1, image2, mask, blend_overlap=1):
    """image2 inside the (N, 1, H, W) mask and image1 outside; blend_overlap > 1 grows the mask by r = blend_overlap // 2 pixels and
    feathers it with a Gaussian kernel of radius r (standard deviation r / 2), so the seam falls on the kept side."""
    radius = blend_overlap // 2
    if radius:
        grid = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
        kernel = torch.exp(-(grid[:, None] ** 2 + grid[None] ** 2) / (radius * radius / 2))
        mask = F.conv2d(F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius), (kernel / kernel.sum())[None, None], padding=radius)
    return image1 * (1 - mask) + image2 * mask
