import json
import os
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .autoencoder_kl import AutoencoderKL
from .pipeline_z_image import ZImagePipeline
from .transformer_z_image import ZImageTransformer2DModel


logger = logging.get_logger(__name__)


def conv_state_dict(sd: dict) -> dict:
    if "x_embedder.weight" not in sd and "model.diffusion_model.x_embedder.weight" not in sd:
        return sd

    inverse_replace = {
        "final_layer.": "all_final_layer.2-1.",
        "x_embedder.": "all_x_embedder.2-1.",
        ".attention.out.bias": ".attention.to_out.0.bias",
        ".attention.k_norm.weight": ".attention.norm_k.weight",
        ".attention.q_norm.weight": ".attention.norm_q.weight",
        ".attention.out.weight": ".attention.to_out.0.weight",
    }

    out_sd: dict[str, torch.Tensor] = {}

    for key, tensor in sd.items():
        key = key.replace("model.diffusion_model.", "")
        
        if key.endswith(".attention.qkv.weight"):
            base = key[: -len(".attention.qkv.weight")]

            total_dim = tensor.shape[0]
            if total_dim % 3 != 0:
                raise ValueError(
                    f"{key}: qkv first dimension ({total_dim}) not divisible by 3"
                )
            d = total_dim // 3
            q, k_w, v = torch.split(tensor, d, dim=0)

            out_sd[base + ".attention.to_q.weight"] = q
            out_sd[base + ".attention.to_k.weight"] = k_w
            out_sd[base + ".attention.to_v.weight"] = v
            continue

        new_key = key
        for comfy_sub, orig_sub in inverse_replace.items():
            new_key = new_key.replace(comfy_sub, orig_sub)
        out_sd[new_key] = tensor

    to_add = {}
    for key, tensor in out_sd.items():
        if key.endswith(".attention.to_out.0.weight"):
            prefix = key[: -len(".attention.to_out.0.weight")]
            bias_key = prefix + ".attention.to_out.0.bias"
            if bias_key not in out_sd:
                to_add[bias_key] = torch.zeros(tensor.shape[0], dtype=tensor.dtype)

    out_sd.update(to_add)
    return out_sd


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        **kwargs,
    ):
        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if transformer_filename is None:
            raise ValueError("No transformer filename provided for Z-Image.")

        self.base_model_type = base_model_type

        # Transformer
        default_transformer_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json") 

        preprocess_sd = conv_state_dict

        transformer = offload.fast_load_transformers_model(transformer_filename, writable_tensors= False, modelClass=ZImageTransformer2DModel, defaultConfigPath= default_transformer_config, preprocess_sd=preprocess_sd)
        transformer.to(dtype)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, transformer_filename, dtype, default_transformer_config)

        # Text encoder

        # text_encoder = Qwen3ForCausalLM.from_pretrained(os.path.dirname(text_encoder_filename), trust_remote_code=True)
        # text_encoder.to(torch.bfloat16)
        # offload.save_model(text_encoder, "c:/temp/qwnen3_bf16_.safetensors")
        
        text_encoder = offload.fast_load_transformers_model( text_encoder_filename, writable_tensors=True, modelClass=Qwen3ForCausalLM,)

        # Tokenizer
        tokenizer_path = os.path.join(os.path.dirname(text_encoder_filename))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # VAE
        vae_filename = fl.locate_file("ZImageTurbo_VAE_bf16.safetensors")
        vae_config_path = os.path.join(os.path.dirname(vae_filename), "ZImageTurbo_VAE_bf16_config.json") 

        vae = offload.fast_load_transformers_model(
            vae_filename,
            writable_tensors=True,
            modelClass=AutoencoderKL,
            defaultConfigPath=vae_config_path,
            default_dtype=VAE_dtype,
        )

        # Scheduler
        with open(fl.locate_file("ZImageTurbo_scheduler_config.json"), "r", encoding="utf-8") as f:
            scheduler_config = json.load(f)

        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        self.pipeline = ZImagePipeline(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
        )
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 0.0,
        batch_size: int = 1,
        callback=None,
        max_sequence_length: int = 512,
        VAE_tile_size=None,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        **kwargs,
    ):
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is None or seed < 0:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling = VAE_tile_size > 0
                tile_size = max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            self.vae.use_tiling = tiling
            self.vae.tile_latent_min_height = tile_size
            self.vae.tile_latent_min_width = tile_size

        guide_scale = 0

        images = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt,
            num_inference_steps=sampling_steps,
            guidance_scale=guide_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=None,
            output_type="pt",
            return_dict=True,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            callback=callback,
            pipeline=self.pipeline,
        )

        if images is None:
            return None

        if not torch.is_tensor(images):
            images = torch.tensor(images)

        return images.transpose(0, 1)

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
