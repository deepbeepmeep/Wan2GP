# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from tqdm import tqdm
from transformers import AutoTokenizer
from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress, set_phase_status, GenerationAborted
from shared.utils.text_encoder_cache import TextEncoderCache
from .transformer import MOSSImageTransformer, rope_embed
from .text_encoder import TextEncoder, Qwen3_5TextConfig, encode_prompt
from .vae import MOSSImageVAE


def sampling_schedule(steps, shift, device):
    # The released nearest-grid schedule is constructed in CPU FP32.
    lower = shift / (1000 + shift - 1)
    values = torch.linspace(1, lower, steps, device='cpu')
    sigmas = shift * values / (1 + (shift - 1) * values)
    training = torch.linspace(1, 0.001, 1000, device='cpu').flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=999)
    lower_index = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower_index]).abs() <= (training[upper] - sigmas).abs(), lower_index, upper)
    return torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1, device='cpu'))).to(device)


class MOSSImagePipeline:
    def __init__(self, transformer, text_encoder, tokenizer, vae):
        self.transformer, self.text_encoder, self.tokenizer, self.vae = transformer, text_encoder, tokenizer, vae
        self.text_encoder_cache = TextEncoderCache()
        self.preview_mean = vae.bn.running_mean.detach().float().cpu().reshape(1, 128, 1, 1)
        self.preview_std = (vae.bn.running_var.detach().float().cpu() + vae.bn_eps).sqrt().reshape(1, 128, 1, 1)
        self._interrupt = False

    @property
    def _interrupt(self):
        return self.transformer._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.transformer._interrupt = self.text_encoder._interrupt = value

    def abort(self):
        self._interrupt = True

    def preview_latents(self, latents):
        z = latents * self.preview_std.to(latents) + self.preview_mean.to(latents)
        b, c, h, w = z.shape
        return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2*h, 2*w).transpose(0, 1)

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt='', n_prompt='', seed=0, sampling_steps=50, width=1024, height=1024, guide_scale=3.0,
                 flow_shift=5.0, batch_size=1, callback=None, VAE_tile_size=None, set_progress_status=None, **kwargs):
        device = runtime.execution_device(self.transformer)
        prompts = [n_prompt, input_prompt] if guide_scale != 1 else [input_prompt]
        prompts = ['\n'.join(line for line in p.splitlines() if not line.lstrip().startswith('#')) for p in prompts]
        set_phase_status('Preparing Text Conditioning')
        texts = self.text_encoder_cache.encode(lambda items: [encode_prompt(self.text_encoder, self.tokenizer, p) for p in items], prompts, device=device)
        set_phase_status('Preparing Timesteps')
        schedule = sampling_schedule(sampling_steps, flow_shift, device)
        h, w = height // 16, width // 16
        config = self.transformer.config
        ropes = [rope_embed(h, w, text.shape[1], device, config.dim // config.num_heads, config.rope_base).bfloat16() for text in texts]
        tiled, tile_size = VAE_tile_size if VAE_tile_size is not None else (True, 1024)
        latents = torch.empty((batch_size, 128, h, w), device='cpu', dtype=torch.float32)
        for image_no in range(batch_size):
            generator = torch.Generator(device='cpu').manual_seed(seed + image_no)
            torch.randn((1, 128, h, w), device='cpu', dtype=torch.float32, generator=generator, out=latents[image_no:image_no + 1])
        latents = latents.to(device)
        if callback is not None:
            callback(-1, None, True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for step in tqdm(range(len(schedule) - 1), desc='Denoising', mininterval=1/3):
                if self._interrupt:
                    raise GenerationAborted
                predictions = self.transformer(latents, schedule[step] * 1000, texts, ropes)
                if self._interrupt:
                    raise GenerationAborted
                velocity = predictions[0] + guide_scale * (predictions[1] - predictions[0]) if len(texts) == 2 else predictions[0]
                latents = latents + (schedule[step + 1] - schedule[step]) * velocity
                del predictions, velocity
                if callback is not None:
                    callback(step, self.preview_latents(latents), False)
        set_phase_status('VAE Decoding')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pixels = self.vae(latents.to(runtime.execution_device(self.vae), dtype=torch.bfloat16), tile_size=tile_size if tiled else 0)
        return pixels.transpose(0, 1)


def load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized, dtype):
    config_path = Path(__file__).with_name('moss_image.json')
    config = json.loads(config_path.read_text())
    folder = model_def['text_encoder_folder']
    assets = {name: fl.locate_file(f'{folder}/{name}') for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    encoder_config = Qwen3_5TextConfig.from_json_file(assets['config.json'])
    tokenizer = AutoTokenizer.from_pretrained(str(Path(assets['tokenizer_config.json']).parent), local_files_only=True, extra_special_tokens={})
    with init_empty_weights(include_buffers=True):
        transformer = MOSSImageTransformer(**config)
        encoder = TextEncoder(encoder_config)
        vae = MOSSImageVAE()
    with torch.device('cpu'):
        encoder.rotary_emb.inv_freq, encoder.rotary_emb.attention_scaling = encoder.rotary_emb.compute_default_rope_parameters(encoder_config, device='cpu')
    encoder.rotary_emb.original_inv_freq = encoder.rotary_emb.inv_freq.clone()
    encoder.rotary_emb._lock_dtype = torch.float32
    offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16, fused_split_map=transformer.split_linear_modules_map)
    offload.load_model_data(encoder, text_encoder_filename, writable_tensors=False, default_dtype=torch.bfloat16, fused_split_map=encoder.split_linear_modules_map)
    offload.load_model_data(vae, fl.locate_file('moss_image/flux2_small_decoder_bf16.safetensors'), writable_tensors=False, default_dtype=None)
    transformer.eval()
    encoder.eval()
    vae.eval()
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, str(config_path))
    return MOSSImagePipeline(transformer, encoder, tokenizer, vae)
