"""Unified AuK speech generation and instruction-based editing for WanGP."""

import math
import json
from pathlib import Path

import torch
import torchaudio
import librosa
from mmgp import offload
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperFeatureExtractor

from .encoder import Qwen2_5OmniFeatureEncoder
from .transformer import AuKModel
from .vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig


FLASH_TIMESTEPS = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]


class AuKVAE(BigVGANFlowVAE):
    @classmethod
    def from_config(cls, config):
        return cls(BigVGANFlowVAEConfig.from_dict(config['vae']))

    def forward(self, tensor, encode=False):
        if encode:
            latent, lengths = self.encoding_and_normalization(tensor)
            return latent[:, :lengths[0]]
        return self.inference_from_latents(self.denormalize(tensor).transpose(1, 2))


class AuKPipeline:
    sample_rate = 24000
    latent_rate = 50

    def __init__(self, model_path, encoder_path, vae_path, tokenizer_folder, config_path, flash=False, dtype=torch.bfloat16):
        self.device = torch.device('cuda')
        dtype = torch.bfloat16
        self.dtype = dtype
        self.flash = flash
        self._interrupt = False
        self._early_stop = False
        self.transformer = offload.fast_load_transformers_model(model_path, modelClass=AuKModel, defaultConfigPath=config_path, default_dtype=dtype, writable_tensors=False)
        encoder_config = json.loads((Path(tokenizer_folder) / 'config.json').read_text(encoding='utf-8'))
        with init_empty_weights():
            self.text_encoder = Qwen2_5OmniFeatureEncoder.from_config(encoder_config)
        self.text_encoder._config = encoder_config
        offload.load_model_data(self.text_encoder, encoder_path, default_dtype=dtype, writable_tensors=False)
        self.text_encoder.eval().requires_grad_(False)
        self.vae = offload.fast_load_transformers_model(vae_path, modelClass=AuKVAE, defaultConfigPath=config_path, default_dtype=torch.float32, writable_tensors=False)
        self.vae._lock_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder, local_files_only=True)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_folder, local_files_only=True)
        self.tokenizer.chat_template = json.loads((Path(tokenizer_folder) / 'chat_template.json').read_text(encoding='utf-8'))['chat_template']
        self.transformer.transformer._interrupt_check = self._should_abort
        self.text_encoder._interrupt_check = self._should_abort
        self.model = self.transformer

        def check_abort(_module, _args, _output):
            if self._should_abort():
                raise InterruptedError('AuK generation interrupted')

        self._abort_handles = [module.register_forward_hook(check_abort) for module in self.vae.modules() if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))]
        self._abort_handles.extend(layer.register_forward_hook(check_abort) for layer in self.text_encoder.audio_tower.layers)

    def get_trans_lora(self):
        return self.transformer, None

    def request_early_stop(self):
        self._early_stop = True

    def _should_abort(self):
        return self._interrupt or self._early_stop

    @torch.inference_mode()
    def generate(self, input_prompt, audio_guide=None, duration_seconds=30, sampling_steps=32, guide_scale=2.0, seed=1, batch_size=1, callback=None, audio_prompt_type='', joint_pass=True, **kwargs):
        self._interrupt = self._early_stop = False
        torch.manual_seed(seed)
        prompt = input_prompt
        audio = None
        if 'A' in audio_prompt_type:
            waveform, _ = librosa.load(audio_guide, sr=self.sample_rate, mono=True, duration=duration_seconds)
            duration_seconds = min(duration_seconds, len(waveform) / self.sample_rate)
            audio = torch.from_numpy(waveform).unsqueeze(0)
        content = [{'type': 'text', 'text': prompt + ('|<no_prompt_audio>|' if audio is None else '')}]
        audio_inputs = {}
        if audio is not None:
            content.append({'type': 'audio', 'audio': audio_guide})
            waveform = torchaudio.functional.resample(audio, self.sample_rate, 16000).squeeze(0).numpy()
            audio_inputs = self.feature_extractor([waveform], sampling_rate=16000, padding='max_length', return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
        messages = [{'role': 'user', 'content': content}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if audio is not None:
            feature_length = (audio_inputs['feature_attention_mask'].sum().item() - 1) // 2 + 1
            audio_tokens = (feature_length - 2) // 2 + 1
            formatted = formatted.replace(self.tokenizer.audio_token, self.tokenizer.audio_token * audio_tokens)
        inputs = dict(self.tokenizer([formatted], return_tensors='pt'), **audio_inputs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        target_length = math.ceil(duration_seconds * self.latent_rate)
        steps = 4 if self.flash else sampling_steps
        strength = 0.0 if self.flash else guide_scale
        try:
            if callback is not None:
                callback(step_idx=-1, override_num_inference_steps=steps, denoising_extra='Encoding instruction and reference audio')
            with torch.autocast('cuda', dtype=self.dtype):
                embeddings = self.text_encoder(**inputs, layer_weights=self.transformer.layer_weights, layer_scale=self.transformer.layer_scale)
            reference = torch.zeros((1, 0, 64), device=self.device)
            if audio is not None:
                reference = self.vae(audio.to(self.device).unsqueeze(0), encode=True)
            reference = reference.expand(batch_size, -1, -1)
            embeddings = embeddings.expand(batch_size, -1, -1)
            latent = torch.randn((batch_size, target_length, 64), device=self.device, dtype=torch.float32)
            times = torch.tensor(FLASH_TIMESTEPS, device=self.device) if self.flash else torch.linspace(0, 1, steps + 1, device=self.device)
            if not self.flash:
                times = 1 - torch.cos(torch.pi / 2 * times)
            for index in tqdm(range(steps), desc='AuK', disable=False):
                if self._should_abort():
                    raise InterruptedError('AuK generation interrupted')
                with torch.autocast('cuda', dtype=self.dtype):
                    velocity = self.transformer(latent, embeddings, times[index], reference, strength, joint_pass)
                latent.add_(velocity.float() * (times[index + 1] - times[index]))
                if callback is not None:
                    callback(step_idx=index, override_num_inference_steps=steps, denoising_extra=f'{index + 1}/{steps} steps')
            if callback is not None:
                callback(step_idx=steps - 1, denoising_extra='Decoding audio')
            result = self.vae(latent).float().cpu()
            return {'x': result[..., :round(duration_seconds * self.sample_rate)], 'audio_sampling_rate': self.sample_rate}
        except InterruptedError:
            return None

    def release(self):
        for handle in self._abort_handles:
            handle.remove()
