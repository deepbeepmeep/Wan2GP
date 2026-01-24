import math

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

from .configuration_heartcodec import HeartCodecConfig
from .models.flow_matching import FlowMatching
from .models.sq_codec import ScalarModel


class HeartCodec(PreTrainedModel):
    config_class = HeartCodecConfig

    def __init__(self, config: HeartCodecConfig):
        super().__init__(config)

        self.config = config

        self.flow_matching = FlowMatching(
            dim=config.dim,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            norm_type=config.norm_type,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
        )
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )
        self.post_init()

        self.sample_rate = config.sample_rate

    @torch.inference_mode()
    def detokenize(
        self,
        codes,
        duration=29.76,
        num_steps=10,
        disable_progress=False,
        guidance_scale=1.25,
        abort_signal=None,
    ):
        if abort_signal and abort_signal():
            return None
        codes = codes.unsqueeze(0).to(self.device)
        first_latent = torch.randn(
            codes.shape[0], int(duration * 25), 256, dtype=self.dtype
        ).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0
        min_samples = int(duration * 12.5)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples * 2
        codes_len = codes.shape[-1]
        target_len = int(
            (codes_len - first_latent_codes_length) / 12.5 * self.sample_rate
        )

        if codes_len < min_samples:
            while codes.shape[-1] < min_samples:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, 0:min_samples]
        codes_len = codes.shape[-1]
        if (codes_len - ovlp_frames) % hop_samples > 0:
            len_codes = (
                math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples
                + ovlp_samples
            )
            while codes.shape[-1] < len_codes:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, 0:len_codes]
        latent_length = int(duration * 25)
        latent_list = []
        restore = False
        if self.flow_matching.vq_embed.layers[0]._codebook.embed.device.type=="cpu":
            self.flow_matching.vq_embed.layers.to("cuda")
            restore = True

        for sinx in range(0, codes.shape[-1] - hop_samples + 1, hop_samples):
            if abort_signal and abort_signal():
                return None
            codes_input = [codes[:, :, sinx : sinx + min_samples]]
            if sinx == 0 or ovlp_frames == 0:
                incontext_length = first_latent_length
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    first_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                    abort_signal=abort_signal,
                )
                if latents is None:
                    return None
                latent_list.append(latents)
            else:
                true_latent = latent_list[-1][:, -ovlp_frames:, :]
                len_add_to_latent = latent_length - true_latent.shape[1]
                incontext_length = true_latent.shape[1]
                true_latent = torch.cat(
                    [
                        true_latent,
                        torch.randn(
                            true_latent.shape[0],
                            len_add_to_latent,
                            true_latent.shape[-1],
                            dtype=true_latent.dtype,
                            device=true_latent.device,
                        ),
                    ],
                    1,
                )
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    true_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                    abort_signal=abort_signal,
                )
                if latents is None:
                    return None
                latent_list.append(latents)

        if restore:
            self.flow_matching.vq_embed.layers.to("cpu")

        latent_list[0] = latent_list[0][:, first_latent_length:, :]
        min_samples = int(duration * self.sample_rate)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples

        output = None
        for latent in latent_list:
            if abort_signal and abort_signal():
                return None
            latent = latent.reshape(
                latent.shape[0], latent.shape[1], 2, latent.shape[2] // 2
            ).permute(0, 2, 1, 3)
            latent = latent.reshape(
                latent.shape[0] * 2, latent.shape[2], latent.shape[3]
            )
            cur_output = (
                self.scalar_model.decode(latent.transpose(1, 2)).squeeze(0).squeeze(1)
            )

            cur_output = cur_output[:, 0:min_samples].detach().cpu()
            if cur_output.dim() == 3:
                cur_output = cur_output[0]

            if output is None:
                output = cur_output
            else:
                if ovlp_samples == 0:
                    output = torch.cat([output, cur_output], -1)
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = (
                        output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:]
                        + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    )
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
        output = output[:, 0:target_len]
        return output
