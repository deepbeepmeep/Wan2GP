"""Lyrics/style/score to stereo music using WanGP's shared AR engine and MMGP."""

import json
from dataclasses import replace
from pathlib import Path

import torch
from mmgp import offload
from tqdm import tqdm
from transformers import Qwen3Config

from shared.llm_engines.nanovllm.token_generation import TokenGenerationEngine
from shared.utils.loras_mutipliers import update_loras_slists

from .modules import YuE2Config
from .protocol import ABC_END, MUSIC_END, CONTEXT, GenerationConfig, SongRequest, token_prefixes, negative_prefix, chunk_ranges
from .sampling import YuE2LogitsProcessor
from .tokenization_yue2 import YuE2TextTokenizer
from .transformer import YuE2AR, YuE2Acoustic
from .vae import YuE2VAE, YuE2VAEConfig
from .loras import YuE2LoraTarget, ar_lora_signature


class YuE2Pipeline:
    sample_rate = 48000
    frame_rate = 25

    def __init__(self, ar_weights, acoustic_weights, tokenizer_path, vae_weights, vae_config, dtype, vae_dtype, lm_decoder_engine, scoring_checkpoint=None, hum_weights=None, hum_encoder_weights=None):
        directory = Path(__file__).parent
        self._interrupt = False
        self._early_stop = False
        self.lm_decoder_engine = lm_decoder_engine
        self.scoring_checkpoint = scoring_checkpoint
        self.hum = None
        self.hum_encoder = None
        if hum_weights is not None:
            from .hum import HumProjections, HumEncoder
            with torch.device("meta"):
                self.hum = HumProjections()
                self.hum_encoder = HumEncoder()
            for model, filename in ((self.hum, hum_weights), (self.hum_encoder, hum_encoder_weights)):
                offload.load_model_data(model, filename, default_dtype=None, writable_tensors=False)
                model.eval().requires_grad_(False)
            self.hum.abort_fn = self._abort_requested

        dtype = vae_dtype = torch.bfloat16
        ar_config = Qwen3Config(**json.loads((directory / "yue2_ar.json").read_text()))
        nar_config = YuE2Config(**json.loads((directory / "yue2.json").read_text()))
        with torch.device("meta"):
            self.text_encoder = YuE2AR(ar_config)
            self.transformer = YuE2Acoustic(nar_config)
            self.vae = YuE2VAE(YuE2VAEConfig(**json.loads(Path(vae_config).read_text())))
        for model, filename, precision in ((self.text_encoder, ar_weights, dtype), (self.transformer, acoustic_weights, dtype), (self.vae, vae_weights, vae_dtype)):
            offload.load_model_data(model, filename, default_dtype=precision, writable_tensors=False)
            model.eval().requires_grad_(False)
        self.text_encoder.configure_engine(lm_decoder_engine, self._abort_requested)
        self.transformer.engine = lm_decoder_engine
        self.transformer.abort_fn = self._abort_requested
        self.vae._model_dtype = vae_dtype
        self.vae._offload_hooks = ["decode", "decode_tiled"]
        self.vae.get_VAE_tile_size = self.get_vae_tile_size
        self._vae_abort_handles = [block.register_forward_hook(self._abort_after_block) for block in self.vae.decoder.layers]
        self.tokenizer = YuE2TextTokenizer(tokenizer_path)
        self.engine = TokenGenerationEngine(self.text_encoder, ar_weights, self.tokenizer, enforce_eager=lm_decoder_engine == "legacy")
        self.lora_target = YuE2LoraTarget(self.text_encoder, self.transformer)
        self._ar_lora_signature = None
        print(f"[YuE2] AR LM engine: {lm_decoder_engine} (CUDA graphs: {'off' if lm_decoder_engine == 'legacy' else 'on'}; Triton decoder kernels: {'on' if lm_decoder_engine == 'vllm' else 'off'}; attention: {'FlashAttention 2' if lm_decoder_engine == 'vllm' else 'PyTorch SDPA'}).")
        print(f"[YuE2] Acoustic flow: PyTorch midpoint solver; attention: {'FlashAttention 2' if lm_decoder_engine == 'vllm' else 'PyTorch SDPA'}; CUDA graphs: off.")
        self.generation_config = GenerationConfig()
        self.last_plan = None
        self.last_latents = None
        self.last_truncated = {}

    @staticmethod
    def get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if vae_config == 1:
            return 0
        if vae_config == 3 or (vae_config == 0 and device_mem_capacity < 12000):
            return 256
        return 1024

    def _abort_requested(self):
        return self._interrupt

    def get_trans_lora(self):
        # Called before the shared loader replaces user/system adapter tensors.
        self.engine.release_runtime_allocations()
        self._ar_lora_signature = None
        self.lora_target.bind()
        # Profiling only creates LoRA slots. Initialize both stages even when
        # the shared loader skips loading because no adapters were selected.
        offload.activate_loras(self.lora_target, [], [])
        return self.lora_target, None

    def get_loras_transformer(self, _get_model_recursive_prop, model_def, model_mode, activated_loras, **kwargs):
        if self.hum is not None or model_mode != 3:
            return [], []
        url = model_def["yue2_lora_instrumental"]
        filenames = {"ar_lora_inst_v3abc.safetensors", "ar_lora_inst_v3abc.bf16.safetensors", "ar_lora_inst_v3abc_comfyui.safetensors"}
        filenames.add(url.split("|", 1)[0].rsplit("/", 1)[-1].lower())
        if any(Path(lora.split("|", 1)[0]).name.lower() in filenames for lora in activated_loras):
            print("[YuE2] Using the selected instrumental AR LoRA and its user multiplier.")
            return [], []
        return [url], [1.0]

    def request_early_stop(self):
        self._early_stop = True

    def _early_stop_requested(self):
        return self._early_stop

    def _abort_after_block(self, module, inputs, output):
        if self._interrupt:
            raise InterruptedError("YuE2 audio decoding interrupted")

    def _tokens(self, prefix, sampling, seed, phase, callback, negative=None, cfg_scale=1.0, direct=False):
        available = CONTEXT - max(len(prefix), len(negative) if negative is not None else 0)
        if phase == "semantic":
            if available < 5:
                raise ValueError("YuE2 lyrics/score leave no room for audio synthesis. Shorten the lyrics or score.")
            if sampling.max_tokens > available:
                print(f"[YuE2] Maximum audio duration limited by context: {sampling.max_tokens / self.frame_rate:.2f}s requested, up to {available / self.frame_rate:.2f}s available after lyrics/score. Generation can end earlier.")
                sampling = replace(sampling, max_tokens=available, min_tokens=min(sampling.min_tokens, available - 1))
        if sampling.max_tokens > available:
            raise ValueError(f"YuE2 {phase} prompt and generation budget exceed its {CONTEXT}-token context. Shorten the lyrics, score or duration.")
        # Trigger MMGP before the engine allocates persistent GPU state.
        self.text_encoder.model.embed_tokens(torch.tensor([prefix[0]], device="cuda"))
        signature = ar_lora_signature(self.text_encoder)
        if signature != self._ar_lora_signature:
            self.engine.release_runtime_allocations()
            self._ar_lora_signature = signature
        tokens, truncated = self.engine.generate_tokens(prefix, end_token=ABC_END if phase == "abc" else MUSIC_END, max_tokens=sampling.max_tokens, seed=seed, logits_processor=YuE2LogitsProcessor(sampling, phase, direct), negative=negative, cfg_scale=cfg_scale, callback=callback, abort_fn=self._abort_requested, stop_fn=self._early_stop_requested if phase == "semantic" else None, stop_min_tokens=sampling.min_tokens if self._early_stop else 0, progress_label="YuE2 score" if phase == "abc" else "YuE2 semantic audio", initial_cache_tokens=60 * self.frame_rate if phase == "semantic" else 0)
        self.last_truncated[phase] = truncated
        if self._early_stop and phase == "semantic":
            print(f"[YuE2] Early stop during {phase}: {len(tokens)} tokens retained.")
        elif truncated:
            print(f"YuE2 {phase} reached its token limit; output may be incomplete.")
        else:
            print(f"[YuE2] {phase} reached its end token after {len(tokens)} tokens" + (f" ({len(tokens) / self.frame_rate:.2f}s of audio)." if phase == "semantic" else "."))
        return tokens

    @torch.inference_mode()
    def generate(self, input_prompt, alt_prompt, seed, duration_seconds, sampling_steps, guide_scale, temperature, top_k, top_p, model_mode=0, custom_settings=None, VAE_tile_size=1024, callback=None, audio_prompt_type="", audio_guide=None, offloadobj=None, input_custom=None, loras_slists=None, **kwargs):
        self._interrupt = self._early_stop = False
        self.last_plan = self.last_latents = None
        self.last_truncated = {}
        mode = "melody" if self.hum is not None else ("full", "melody", "off", "full")[model_mode]
        carrier = None
        midi = None
        side_files = {}
        abc = ""
        if "A" not in audio_prompt_type and input_custom is not None:
            abc = Path(input_custom).read_text(encoding="utf-8-sig").strip()
            if not abc:
                raise ValueError("The ABC score file is empty.")
        maximum = int(duration_seconds * self.frame_rate)
        sampling = replace(self.generation_config.semantic, max_tokens=maximum, min_tokens=min(200, maximum - 1), temperature=temperature, top_k=top_k, top_p=top_p)
        try:
            if loras_slists is not None:
                self.lora_target.validate_ar_scaling(loras_slists)
            if self.hum is not None:
                from .hum import prosody_carrier
                import time
                self.engine.release_runtime_allocations()
                offloadobj.unload_all()
                def report_hum(title):
                    print(f"[YuE2] {title}", flush=True)
                    if callback is not None:
                        callback(step_idx=0, override_num_inference_steps=1, progress_title=title, denoising_extra=title)
                audio = prosody_carrier(audio_guide, self._abort_requested, report_hum)
                report_hum("Encoding Hum Carrier")
                last_update = 0.
                with tqdm(desc="Encoding Hum Carrier", unit="tile") as progress:
                    def report_encode(completed, total):
                        nonlocal last_update
                        progress.total = total
                        progress.update()
                        now = time.monotonic()
                        if callback is not None and (now - last_update >= 1 / 3 or completed == total):
                            last_update = now
                            callback(step_idx=completed - 1, override_num_inference_steps=total, progress_title="Encoding Hum Carrier", denoising_extra="Encoding Hum Carrier", progress_unit="tiles")
                        if self._interrupt:
                            raise InterruptedError("Hum encoding interrupted")
                    carrier = self.hum_encoder.encode(audio, VAE_tile_size, self._abort_requested, report_encode)
                del audio
            if "A" in audio_prompt_type:
                from .sheetsage2.scoring import score_audio
                self.engine.release_runtime_allocations()
                offloadobj.unload_all()
                abc, midi = score_audio(audio_guide, self.scoring_checkpoint, mode == "melody", callback, self._abort_requested)
            request = SongRequest(style=alt_prompt, lyrics=input_prompt, cot=mode, abc=abc or None, cfg_scale=guide_scale, seed=seed)
            if self.hum is not None and model_mode == 0:
                from .hum import open_hum_score
                partial = self.tokenizer.encode(open_hum_score(abc))
                request = replace(request, abc=None)
                continuation = self._tokens(token_prefixes(request, self.tokenizer) + partial, self.generation_config.abc, seed, "abc", callback)
                abc_ids = partial + continuation
                abc = self.tokenizer.decode(abc_ids)
                midi = None
                request = replace(request, abc=abc)
            elif mode == "off":
                abc_ids = []
            elif abc:
                abc_ids = self.tokenizer.encode(abc)
            else:
                abc_ids = self._tokens(token_prefixes(request, self.tokenizer), self.generation_config.abc, seed, "abc", callback)
                abc = self.tokenizer.decode(abc_ids)
            self.last_plan = {"abc": abc, "abc_ids": abc_ids, "request": request.to_dict()}
            if abc and custom_settings is not None and custom_settings.get("save_score", 0):
                from .score_export import score_side_files
                side_files = score_side_files(abc, midi)
            prefix = token_prefixes(request, self.tokenizer, abc_ids)
            negative = negative_prefix(request, self.tokenizer, abc_ids) if guide_scale != 1 else None
            codec = self._tokens(prefix, sampling, seed, "semantic", callback, negative, guide_scale, mode == "off")
            self.engine.release_runtime_allocations()
            if not codec:
                if self._early_stop:
                    return None
                raise RuntimeError("YuE2 generated no audio tokens.")
            chunks = chunk_ranges(len(codec), len(prefix))
            if loras_slists is not None:
                update_loras_slists(self.transformer, loras_slists, sampling_steps)
            noise = torch.randn((len(codec), 64), device="cpu", generator=torch.Generator(device="cpu").manual_seed(seed))
            latent_parts = []
            total = len(chunks) * sampling_steps
            with tqdm(total=total, desc="YuE2 acoustic synthesis", unit="step") as progress:
                for chunk_index, (start, end) in enumerate(chunks):
                    hum_features = None
                    if carrier is not None:
                        conditioning = torch.zeros((end - start, 64), device="cpu", dtype=carrier.dtype)
                        available = max(0, min(end, len(carrier)) - start)
                        conditioning[:available] = carrier[start:start + available]
                        hum_features = self.hum(conditioning)
                        del conditioning
                    ar_ids = prefix + codec[start:end] + [MUSIC_END]
                    cache = [self.text_encoder.condition(ar_ids)]
                    def report(step):
                        progress.update()
                        if callback is not None:
                            callback(step_idx=chunk_index * sampling_steps + step, override_num_inference_steps=total, denoising_extra="YuE2 acoustic synthesis")
                    latent_parts.append(self.transformer.synthesize(noise[start:end], cache, len(ar_ids), sampling_steps, report, hum_features=hum_features))
                    del hum_features
            latents = torch.cat(latent_parts)
            self.last_latents = latents
            latent = latents.T[None]
            tile = VAE_tile_size
            total = (len(latents) + tile - 1) // tile if tile else 1
            with tqdm(total=total, desc="YuE2 audio decoding", unit="tile") as progress:
                def report_decode(completed, total):
                    progress.update()
                    if callback is not None:
                        callback(step_idx=completed - 1, override_num_inference_steps=total, denoising_extra="YuE2 audio decoding", progress_unit="tiles")
                if tile:
                    audio = self.vae.decode_tiled(latent, core_frames=tile, halo_frames=16, on_progress=report_decode)
                else:
                    audio = self.vae.decode(latent).float().cpu()
                    report_decode(1, 1)
            if self._interrupt:
                return None
            if not torch.isfinite(audio).all():
                raise FloatingPointError("YuE2 decoded non-finite audio.")
            return {"x": audio[0].float().clamp_(-1, 1), "audio_sampling_rate": self.sample_rate, "side_files": side_files}
        except InterruptedError:
            if not self._interrupt:
                raise
            print("[YuE2] Generation aborted.")
            return None
        finally:
            self.engine.release_runtime_allocations()
            carrier = None

    def release(self):
        self.engine.close()
        self.last_latents = self.last_plan = None
