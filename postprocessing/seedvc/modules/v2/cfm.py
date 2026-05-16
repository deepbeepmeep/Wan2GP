import torch
from tqdm import tqdm

class CFM(torch.nn.Module):
    def __init__(
        self,
        estimator: torch.nn.Module,
    ):
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator = estimator
        self.in_channels = estimator.in_channels
        self.criterion = torch.nn.L1Loss()

    @torch.inference_mode()
    def inference(self,
                  mu: torch.Tensor,
                  x_lens: torch.Tensor,
                  prompt: torch.Tensor,
                  style: torch.Tensor,
                  n_timesteps=10,
                  temperature=1.0,
                  inference_cfg_rate=[0.5, 0.5],
                  random_voice=False,
                  ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            inference_cfg_rate (float, optional): Classifier-Free Guidance inference introduced in VoiceBox. Defaults to 0.5.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate, random_voice)
    def solve_euler(self, x, x_lens, prompt, mu, style, t_span, inference_cfg_rate=[0.5, 0.5], random_voice=False,):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            inference_cfg_rate (float, optional): Classifier-Free Guidance inference introduced in VoiceBox. Defaults to 0.5.
            sway_sampling (bool, optional): Sway sampling. Defaults to False.
            amo_sampling (bool, optional): AMO sampling. Defaults to False.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            if random_voice:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = ((1.0 + inference_cfg_rate[0]) * cond_txt - inference_cfg_rate[0] * uncond)
            elif all(i == 0 for i in inference_cfg_rate):
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
            elif inference_cfg_rate[0] == 0:
                # Classifier-Free Guidance inference introduced in VoiceBox
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, mu], dim=0),
                )
                cond_txt_spk, cond_txt = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = ((1.0 + inference_cfg_rate[1]) * cond_txt_spk - inference_cfg_rate[1] * cond_txt)
            elif inference_cfg_rate[1] == 0:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt_spk, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = ((1.0 + inference_cfg_rate[0]) * cond_txt_spk - inference_cfg_rate[0] * uncond)
            else:
                # Multi-condition Classifier-Free Guidance inference introduced in MegaTTS3
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt_spk, cond_txt, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2], cfg_dphi_dt[2:3]
                dphi_dt = (1.0 + inference_cfg_rate[0] + inference_cfg_rate[1]) * cond_txt_spk - \
                    inference_cfg_rate[0] * uncond - inference_cfg_rate[1] * cond_txt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return x

    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(), style, mu)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss
