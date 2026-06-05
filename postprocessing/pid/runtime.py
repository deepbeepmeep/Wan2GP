import gc
import os
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights

from shared.attention import attention_config_shared_state
from postprocessing.pid.networks import PidNet
from postprocessing.pid.networks.pixeldit_official import get_pid_linear_split_map


PID_LEGACY_UPSAMPLING_VALUE = "pid4"
PID_FLUX_VAE_UPSAMPLING_VALUE = "flux_vae_pid4"
PID_FLUX2_VAE_UPSAMPLING_VALUE = "flux2_vae_pid4"
PID_FLUX_POST_UPSAMPLING_VALUE = "flux_pid4"
PID_FLUX2_POST_UPSAMPLING_VALUE = "flux2_pid4"
PID_VAE_UPSAMPLING_VALUES = (PID_FLUX_VAE_UPSAMPLING_VALUE, PID_FLUX2_VAE_UPSAMPLING_VALUE, PID_LEGACY_UPSAMPLING_VALUE)
PID_POST_UPSAMPLING_VALUES = (PID_FLUX_POST_UPSAMPLING_VALUE, PID_FLUX2_POST_UPSAMPLING_VALUE)
PID_UPSAMPLING_VALUES = PID_VAE_UPSAMPLING_VALUES + PID_POST_UPSAMPLING_VALUES
PID_CHECKPOINT_TYPES = ("2k", "2kto4k")
PID_TILING_THRESHOLD_AUTO = 0
PID_TILING_THRESHOLD_2K = 1
PID_TILING_THRESHOLD_4K = 2
PID_TILING_THRESHOLD_DEFAULT = PID_TILING_THRESHOLD_AUTO
PID_TILING_THRESHOLD_CHOICES = [("Auto", PID_TILING_THRESHOLD_AUTO), ("2k", PID_TILING_THRESHOLD_2K), ("4k", PID_TILING_THRESHOLD_4K)]
PID_TILING_AUTO_4K_MIN_VRAM_GB = 12
PID_TEXT_ENCODER_FOLDER = "gemma-2-2b-it"
PID_TEXT_ENCODER_FILENAME = "gemma-2-2b-it_quanto_bf16_int8.safetensors"
PID_TEXT_ENCODER_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    PID_TEXT_ENCODER_FILENAME,
]

_STUDENT_T_LIST = (0.999, 0.866, 0.634, 0.342, 0.0)
_MODEL_MAX_LENGTH = 300
_FM_TIMESCALE = 1000.0
PID_TILE_UPSAMPLING = True
PID_TILE_2K_MAX_OUTPUT_PIXELS = 2048 * 2048
PID_TILE_MIN_OUTPUT_PIXELS = 5120 * 2880
PID_TILE_2K_INPUT_SIZE = 512
PID_TILE_4K_INPUT_SIZE = 1024
PID_TILE_INPUT_SIZE = PID_TILE_2K_INPUT_SIZE
PID_TILE_OVERLAP = 0.25
PID_UPSAMPLER_BUDGET = 1500
_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def is_pid_upsampling(spatial_upsampling):
    return str(spatial_upsampling or "").strip().lower() in PID_UPSAMPLING_VALUES


def is_pid_vae_upsampling(spatial_upsampling):
    return str(spatial_upsampling or "").strip().lower() in PID_VAE_UPSAMPLING_VALUES


def is_pid_post_upsampling(spatial_upsampling):
    return str(spatial_upsampling or "").strip().lower() in PID_POST_UPSAMPLING_VALUES


def normalize_pid_backbone(backbone):
    text = str(backbone or "").strip().lower()
    if text in ("z_image", "zimage", "zimage_turbo", "z_image_turbo"):
        return "z_image"
    if text == "flux2":
        return "flux2"
    return "flux"


def pid_backbone_for_upsampling(spatial_upsampling, default="flux"):
    text = str(spatial_upsampling or "").strip().lower()
    if text in (PID_FLUX2_VAE_UPSAMPLING_VALUE, PID_FLUX2_POST_UPSAMPLING_VALUE):
        return "flux2"
    if text in (PID_FLUX_VAE_UPSAMPLING_VALUE, PID_FLUX_POST_UPSAMPLING_VALUE):
        return "flux"
    return normalize_pid_backbone(default)


def pid_vae_upsampling_value(backbone):
    return PID_FLUX2_VAE_UPSAMPLING_VALUE if normalize_pid_backbone(backbone) == "flux2" else PID_FLUX_VAE_UPSAMPLING_VALUE


def pid_vae_upsampling_choice(backbone):
    return ("Flux2 VAE PiD Upsampler", PID_FLUX2_VAE_UPSAMPLING_VALUE) if normalize_pid_backbone(backbone) == "flux2" else ("Flux VAE PiD Upsampler", PID_FLUX_VAE_UPSAMPLING_VALUE)


def pid_post_upsampling_choices(include_name=True):
    prefix = "" if include_name else ""
    return [
        (f"{prefix}Flux PiD Upsampler", PID_FLUX_POST_UPSAMPLING_VALUE),
        (f"{prefix}Flux2 PiD Upsampler", PID_FLUX2_POST_UPSAMPLING_VALUE),
    ]


def pid_checkpoint_family(backbone):
    backbone = normalize_pid_backbone(backbone)
    return "flux" if backbone == "z_image" else backbone


def normalize_pid_tiling_threshold(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = PID_TILING_THRESHOLD_DEFAULT
    return value if value in (PID_TILING_THRESHOLD_AUTO, PID_TILING_THRESHOLD_2K, PID_TILING_THRESHOLD_4K) else PID_TILING_THRESHOLD_DEFAULT


def resolve_pid_tiling_threshold(value):
    value = normalize_pid_tiling_threshold(value)
    if value != PID_TILING_THRESHOLD_AUTO:
        return value
    try:
        vram_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
    except Exception:
        vram_gb = 0.0
    return PID_TILING_THRESHOLD_4K if vram_gb >= PID_TILING_AUTO_4K_MIN_VRAM_GB else PID_TILING_THRESHOLD_2K


def pid_tiling_threshold_label(value):
    value = normalize_pid_tiling_threshold(value)
    return {PID_TILING_THRESHOLD_AUTO: "Auto", PID_TILING_THRESHOLD_2K: "2k", PID_TILING_THRESHOLD_4K: "4k"}[value]


def normalize_pid_checkpoint_types(ckpt_types=None):
    if ckpt_types is None:
        return PID_CHECKPOINT_TYPES
    if isinstance(ckpt_types, str):
        ckpt_types = (ckpt_types,)
    out = tuple(ckpt_type for ckpt_type in ckpt_types if ckpt_type in PID_CHECKPOINT_TYPES)
    return out or ("2k",)


def pid_checkpoint_types_for_tiling_threshold(tiling_threshold):
    return ("2k",) if resolve_pid_tiling_threshold(tiling_threshold) == PID_TILING_THRESHOLD_2K else PID_CHECKPOINT_TYPES


def pid_checkpoint_filename(backbone, ckpt_type="2k"):
    ckpt_type = ckpt_type if ckpt_type in PID_CHECKPOINT_TYPES else "2k"
    checkpoint_family = pid_checkpoint_family(backbone)
    if ckpt_type == "2kto4k":
        if checkpoint_family == "flux2":
            return "PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606_quanto_bf16_int8.safetensors"
        return f"PiD_res2kto4k_sr4x_official_{checkpoint_family}_distill_4step_quanto_bf16_int8.safetensors"
    return f"PiD_res2k_sr4x_official_{checkpoint_family}_distill_4step_quanto_bf16_int8.safetensors"


def select_pid_checkpoint_type(width, height):
    return "2kto4k" if max(int(width), int(height)) > 512 else "2k"


def _pid_latent_downscale(backbone):
    return 16 if normalize_pid_backbone(backbone) == "flux2" else 8


def _pid_tile_axis(length, scale, tile_input_size=PID_TILE_INPUT_SIZE):
    tile_size = min(int(length), max(scale, (int(tile_input_size) // scale) * scale))
    if int(length) <= tile_size:
        return [(0, int(length))]
    stride = max(scale, int(tile_size * (1.0 - PID_TILE_OVERLAP)) // scale * scale)
    starts = list(range(0, int(length) - tile_size + 1, stride))
    last_start = int(length) - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + tile_size) for start in starts]


def _pid_repo(backbone):
    backbone = normalize_pid_backbone(backbone)
    if backbone == "flux2":
        return "DeepBeepMeep/Flux2"
    return "DeepBeepMeep/Flux"


def pid_vae_filename(backbone):
    return "flux2_vae.safetensors" if normalize_pid_backbone(backbone) == "flux2" else "flux_vae.safetensors"


def _pid_vae_repo(backbone):
    return "DeepBeepMeep/Flux2" if normalize_pid_backbone(backbone) == "flux2" else "DeepBeepMeep/Flux"


def get_pid_download_def(backbone, ckpt_type=None, include_text_encoder=True, include_vae=True):
    ckpt_types = normalize_pid_checkpoint_types(ckpt_type)
    download_defs = [
        {
            "repoId": _pid_repo(backbone),
            "sourceFolderList": [""],
            "fileList": [[pid_checkpoint_filename(backbone, ckpt) for ckpt in ckpt_types]],
        }
    ]
    if include_text_encoder:
        download_defs.append(
            {
                "repoId": "DeepBeepMeep/Flux",
                "sourceFolderList": [PID_TEXT_ENCODER_FOLDER],
                "fileList": [PID_TEXT_ENCODER_FILES],
            }
        )
    if include_vae:
        download_defs.append(
            {
                "repoId": _pid_vae_repo(backbone),
                "sourceFolderList": [""],
                "fileList": [[pid_vae_filename(backbone)]],
            }
        )
    return download_defs


def _disable_broken_transformers_optional_imports():
    import transformers.utils.import_utils as import_utils

    import_utils._scipy_available = False
    import_utils._sklearn_available = False


def _load_gemma2_model_class():
    _disable_broken_transformers_optional_imports()
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Model

    return Gemma2Model


def _preprocess_gemma2_state_dict(state_dict):
    out = {}
    for key, tensor in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = tensor
        elif not key.startswith("lm_head."):
            out[key] = tensor
    return out


def _preprocess_pid_state_dict(state_dict):
    out = {}
    for key, tensor in state_dict.items():
        if key.startswith("net.") and not key.startswith("net_ema."):
            out[key[len("net.") :]] = tensor
        elif key.startswith(("net_ema.", "fake_score.", "discriminator.")):
            continue
        else:
            out[key] = tensor
    return out


def _build_pid_net(backbone):
    backbone = normalize_pid_backbone(backbone)
    net_kwargs = {
        "in_channels": 3,
        "num_groups": 24,
        "hidden_size": 1536,
        "pixel_hidden_size": 16,
        "pixel_attn_hidden_size": 1152,
        "pixel_num_groups": 16,
        "patch_depth": 14,
        "pixel_depth": 2,
        "patch_size": 16,
        "txt_embed_dim": 2304,
        "txt_max_length": _MODEL_MAX_LENGTH,
        "use_text_rope": True,
        "text_rope_theta": 10000.0,
        "repa_encoder_index": 6,
        "lq_inject_mode": "controlnet",
        "lq_in_channels": 0,
        "lq_latent_channels": 16,
        "lq_hidden_dim": 512,
        "lq_gate_type": "sigma_aware_per_token_per_dim",
        "lq_interval": 2,
        "zero_init_lq": True,
        "train_lq_proj_only": False,
        "sr_scale": 4,
        "pit_lq_inject": False,
        "pit_lq_gate_type": "sigma_aware_per_token_per_dim",
        "rope_mode": "ntk_aware",
        "rope_ref_h": 1024,
        "rope_ref_w": 1024,
    }
    if backbone == "flux2":
        net_kwargs.update({"lq_latent_channels": 128, "latent_spatial_down_factor": 16})
    return PidNet(**net_kwargs)


def _build_pid_vae(backbone):
    if normalize_pid_backbone(backbone) == "flux2":
        from models.flux.modules.autoencoder_flux2 import AutoencoderKLFlux2, AutoEncoderParamsFlux2

        return AutoencoderKLFlux2(AutoEncoderParamsFlux2())
    from models.flux.modules.autoencoder import AutoEncoder
    from models.flux.util import configs

    return AutoEncoder(configs["flux-dev"].ae_params)


def _apply_pid_offload_budgets(pipe, kwargs):
    budgets = kwargs.get("budgets")
    if not isinstance(budgets, dict):
        return
    for name in pipe:
        if name.startswith("pid_upsampler_"):
            budgets[name] = PID_UPSAMPLER_BUDGET


class PiDUpsampler:
    def __init__(self, backbone="flux", dtype=torch.bfloat16, ckpt_types=None):
        self.backbone = normalize_pid_backbone(backbone)
        self.dtype = dtype
        self.ckpt_types = normalize_pid_checkpoint_types(ckpt_types)
        self.upsampling_set = pid_vae_upsampling_value(self.backbone)

        from shared.utils import files_locator as fl

        text_encoder_path = fl.locate_file(os.path.join(PID_TEXT_ENCODER_FOLDER, PID_TEXT_ENCODER_FILENAME))
        text_encoder_config = fl.locate_file(os.path.join(PID_TEXT_ENCODER_FOLDER, "config.json"))
        tokenizer_path = os.path.dirname(fl.locate_file(os.path.join(PID_TEXT_ENCODER_FOLDER, "tokenizer_config.json")))
        vae_path = fl.locate_file(pid_vae_filename(self.backbone))

        _disable_broken_transformers_optional_imports()
        from mmgp import offload

        self.nets = {}
        for ckpt_type in self.ckpt_types:
            checkpoint_path = fl.locate_file(pid_checkpoint_filename(self.backbone, ckpt_type))
            with init_empty_weights(include_buffers=True):
                net = _build_pid_net(self.backbone)
            net.name = f"pid_upsampler_{ckpt_type}"
            offload.load_model_data(
                net,
                checkpoint_path,
                writable_tensors=False,
                preprocess_sd=_preprocess_pid_state_dict,
                default_dtype=dtype,
            )
            split_linear_modules_map = get_pid_linear_split_map(net.hidden_size, net.pixel_attn_hidden_size)
            net.split_linear_modules_map = split_linear_modules_map
            offload.split_linear_modules(net, split_linear_modules_map)
            net.eval().requires_grad_(False)
            self.nets[ckpt_type] = net

        self.text_encoder = offload.fast_load_transformers_model(
            text_encoder_path,
            modelClass=_load_gemma2_model_class(),
            defaultConfigPath=text_encoder_config,
            writable_tensors=False,
            preprocess_sd=_preprocess_gemma2_state_dict,
            default_dtype=dtype,
        )
        self.text_encoder.name = "pid_text_encoder"
        with init_empty_weights(include_buffers=True):
            self.vae = _build_pid_vae(self.backbone)
        self.vae.name = "pid_vae_encoder"
        offload.load_model_data(self.vae, vae_path, writable_tensors=False, default_dtype=dtype)
        self.vae.eval().requires_grad_(False)
        _disable_broken_transformers_optional_imports()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
        self.tokenizer.padding_side = "right"
        self._chi_prompt_str = "\n".join(_CHI_PROMPT)
        self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str))

    def pipe_modules(self):
        pipe = {f"pid_upsampler_{ckpt_type}": net for ckpt_type, net in self.nets.items()}
        pipe["pid_text_encoder"] = self.text_encoder
        pipe["pid_vae_encoder"] = self.vae
        return pipe

    @torch.inference_mode()
    def _encode_text_raw(self, captions, device):
        prompts = [self._chi_prompt_str + str(caption or "") for caption in captions]
        max_length = self._num_chi_tokens + _MODEL_MAX_LENGTH - 2
        caption_token = self.tokenizer(
            prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = caption_token["input_ids"].to(device)
        attention_mask = caption_token["attention_mask"].to(device)
        caption_token = None
        caption_embs = self.text_encoder(input_ids, attention_mask)[0]
        select_index = [0] + list(range(-_MODEL_MAX_LENGTH + 1, 0))
        caption_embs = caption_embs[:, select_index].to(dtype=self.dtype)
        del input_ids, attention_mask
        return caption_embs

    @staticmethod
    def _velocity_to_x0(x_t, net_output, t):
        original_dtype = x_t.dtype
        shape = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
        return (x_t.float() - t.float().view(*shape) * net_output.float()).to(original_dtype)

    @staticmethod
    def _apply_velocity_step_inplace(x, net_output, t_cur, t_next, generator):
        x.add_(net_output, alpha=-float(t_cur))
        del net_output
        if float(t_next) > 0:
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x.mul_(1.0 - float(t_next)).add_(eps, alpha=float(t_next))
            del eps
        return x

    def encode_lq_image(self, lq_image):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if lq_image.device.type == "cuda" else nullcontext()
        with autocast_ctx:
            return self.vae.encode(lq_image.to(dtype=self.dtype))

    def _decode_patch(self, lq_latent, caption_embs, degrade_sigma, image_h, image_w, seed, num_steps, ckpt_type, abort_callback=None, progress_callback=None, progress_start=0, progress_total=None, progress_final=True):
        device = caption_embs.device
        batch_size = caption_embs.shape[0]
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed if seed is not None and seed >= 0 else 0))
        x = torch.randn(batch_size, 3, image_h, image_w, device=device, dtype=self.dtype, generator=generator)

        full_t = torch.tensor(_STUDENT_T_LIST, device=device, dtype=torch.float32)
        if num_steps != len(_STUDENT_T_LIST) - 1:
            indices = torch.linspace(0, len(full_t) - 1, int(num_steps) + 1, device=device).round().long()
            full_t = full_t[indices]

        autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            for step_no, (t_cur, t_next) in enumerate(zip(full_t[:-1], full_t[1:])):
                if abort_callback is not None and abort_callback():
                    return None
                if progress_callback is not None:
                    progress_callback("PiD", int(progress_start) + step_no if progress_total is not None else step_no, progress_total or len(full_t) - 1)
                t_cur_batch = t_cur.expand(x.shape[0])
                net = self.nets[ckpt_type]
                v_pred = net(
                    x,
                    t_cur_batch * _FM_TIMESCALE,
                    caption_embs,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
                net.last_repa_tokens = None
                x = self._apply_velocity_step_inplace(x, v_pred, t_cur, t_next, generator)

        if progress_callback is not None and progress_final:
            progress_callback("PiD", len(full_t) - 1, len(full_t) - 1)
        del lq_latent, full_t
        return x.clamp(-1, 1)

    @staticmethod
    def _float_to_uint8(image):
        return image.clamp_(-1, 1).add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)

    @staticmethod
    def _tile_weight(tile_h, tile_w, top, left, bottom, right, full_h, full_w, device, dtype):
        weight_y = torch.ones(tile_h, device=device, dtype=dtype)
        weight_x = torch.ones(tile_w, device=device, dtype=dtype)
        overlap_y = max(1, int(round(tile_h * PID_TILE_OVERLAP)))
        overlap_x = max(1, int(round(tile_w * PID_TILE_OVERLAP)))
        if top > 0:
            weight_y[:overlap_y] = torch.linspace(0.0, 1.0, overlap_y, device=device, dtype=dtype)
        if bottom < full_h:
            weight_y[-overlap_y:] = torch.linspace(1.0, 0.0, overlap_y, device=device, dtype=dtype)
        if left > 0:
            weight_x[:overlap_x] = torch.linspace(0.0, 1.0, overlap_x, device=device, dtype=dtype)
        if right < full_w:
            weight_x[-overlap_x:] = torch.linspace(1.0, 0.0, overlap_x, device=device, dtype=dtype)
        return weight_y.view(1, 1, tile_h, 1) * weight_x.view(1, 1, 1, tile_w)

    def _previous_tile_weight(self, processed_tiles, out_top, out_bottom, out_left, out_right, full_h, full_w, device, dtype):
        tile_h, tile_w = out_bottom - out_top, out_right - out_left
        previous_weight = torch.zeros(1, 1, tile_h, tile_w, device=device, dtype=dtype)
        for prev_top, prev_bottom, prev_left, prev_right in processed_tiles:
            inter_top = max(out_top, prev_top)
            inter_bottom = min(out_bottom, prev_bottom)
            inter_left = max(out_left, prev_left)
            inter_right = min(out_right, prev_right)
            if inter_top >= inter_bottom or inter_left >= inter_right:
                continue
            prev_weight = self._tile_weight(prev_bottom - prev_top, prev_right - prev_left, prev_top, prev_left, prev_bottom, prev_right, full_h, full_w, device, dtype)
            previous_weight[:, :, inter_top - out_top:inter_bottom - out_top, inter_left - out_left:inter_right - out_left].add_(
                prev_weight[:, :, inter_top - prev_top:inter_bottom - prev_top, inter_left - prev_left:inter_right - prev_left]
            )
            del prev_weight
        return previous_weight

    def _direct_ckpt_type(self, lq_image, ckpt_type, tiling_threshold):
        if resolve_pid_tiling_threshold(tiling_threshold) == PID_TILING_THRESHOLD_2K:
            return "2k"
        ckpt_type = ckpt_type if ckpt_type in self.nets else select_pid_checkpoint_type(lq_image.shape[-1], lq_image.shape[-2])
        return ckpt_type if ckpt_type in self.nets else "2k"

    def _should_tile(self, lq_image, tiling_threshold):
        if not PID_TILE_UPSAMPLING:
            return False
        output_pixels = int(lq_image.shape[-2] * 4) * int(lq_image.shape[-1] * 4)
        threshold_pixels = PID_TILE_2K_MAX_OUTPUT_PIXELS if resolve_pid_tiling_threshold(tiling_threshold) == PID_TILING_THRESHOLD_2K else PID_TILE_MIN_OUTPUT_PIXELS
        return output_pixels > threshold_pixels

    @staticmethod
    def _tile_plan_processed_pixels(rows, cols):
        row_pixels = sum((bottom - top) * 4 for top, bottom in rows)
        col_pixels = sum((right - left) * 4 for left, right in cols)
        return int(row_pixels * col_pixels)

    def _tile_plan(self, lq_h, lq_w, latent_scale, tiling_threshold):
        resolved_threshold = resolve_pid_tiling_threshold(tiling_threshold)
        candidates = [("2k", PID_TILE_2K_INPUT_SIZE)]
        if resolved_threshold == PID_TILING_THRESHOLD_4K and "2kto4k" in self.nets:
            candidates.append(("2kto4k", PID_TILE_4K_INPUT_SIZE))
        plans = []
        for ckpt_type, tile_input_size in candidates:
            if ckpt_type not in self.nets:
                continue
            rows = _pid_tile_axis(lq_h, latent_scale, tile_input_size)
            cols = _pid_tile_axis(lq_w, latent_scale, tile_input_size)
            plans.append(
                {
                    "ckpt_type": ckpt_type,
                    "tile_input_size": tile_input_size,
                    "rows": rows,
                    "cols": cols,
                    "processed_pixels": self._tile_plan_processed_pixels(rows, cols),
                    "tile_count": len(rows) * len(cols),
                }
            )
        return min(plans, key=lambda plan: (plan["processed_pixels"], plan["tile_count"]))

    def _encode_tile_latents(self, lq_image, rows, cols, abort_callback=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback("PiD VAE Encode")
        total_tiles = len(rows) * len(cols)
        print(f"[PiD] VAE encoding {total_tiles} tiles before tiled denoising; latents kept in CPU RAM")
        tile_latents = []
        for top, bottom in rows:
            for left, right in cols:
                if abort_callback is not None and abort_callback():
                    return None
                tile_image = lq_image[:, :, top:bottom, left:right].contiguous()
                tile_latents.append(self.encode_lq_image(tile_image).to("cpu"))
                del tile_image
        return tile_latents

    def _decode_tiled(self, lq_image, lq_latent, caption_embs, degrade_sigma, seed, num_steps, tile_plan, abort_callback=None, progress_callback=None):
        device = lq_image.device
        batch_size, _, lq_h, lq_w = lq_image.shape
        latent_scale = _pid_latent_downscale(self.backbone)
        rows = tile_plan["rows"]
        cols = tile_plan["cols"]
        tile_ckpt_type = tile_plan["ckpt_type"]
        full_h, full_w = int(lq_h * 4), int(lq_w * 4)
        output = torch.zeros(batch_size, 3, full_h, full_w, device=device, dtype=torch.uint8)
        processed_tiles = []
        total_tiles = len(rows) * len(cols)
        denoise_steps = int(num_steps)
        total_denoise_steps = total_tiles * denoise_steps
        print(f"[PiD] tiled upsampling enabled: tiles={len(rows)}x{len(cols)}, overlap={PID_TILE_OVERLAP:g}, tile_input={tile_plan['tile_input_size']}px, ckpt={tile_ckpt_type}, processed_pixels={tile_plan['processed_pixels']}")
        tile_latents = self._encode_tile_latents(lq_image, rows, cols, abort_callback=abort_callback, progress_callback=progress_callback) if lq_latent is None else None
        if lq_latent is None and tile_latents is None:
            return None
        tile_no = 0
        for top, bottom in rows:
            for left, right in cols:
                if abort_callback is not None and abort_callback():
                    return None
                if tile_latents is not None:
                    tile_latent = tile_latents[tile_no]
                else:
                    latent_top, latent_bottom = top // latent_scale, min(lq_latent.shape[-2], -(-bottom // latent_scale))
                    latent_left, latent_right = left // latent_scale, min(lq_latent.shape[-1], -(-right // latent_scale))
                    tile_latent = lq_latent[:, :, latent_top:latent_bottom, latent_left:latent_right].contiguous()
                tile = self._decode_patch(tile_latent.to(device=device, dtype=self.dtype), caption_embs, degrade_sigma, (bottom - top) * 4, (right - left) * 4, None if seed is None else int(seed) + tile_no, num_steps, tile_ckpt_type, abort_callback=abort_callback, progress_callback=progress_callback, progress_start=tile_no * denoise_steps, progress_total=total_denoise_steps, progress_final=False)
                if tile is None:
                    return None
                if tile_latents is not None:
                    tile_latents[tile_no] = None
                out_top, out_bottom, out_left, out_right = top * 4, bottom * 4, left * 4, right * 4
                tile = self._float_to_uint8(tile)
                weight = self._tile_weight(tile.shape[-2], tile.shape[-1], out_top, out_left, out_bottom, out_right, full_h, full_w, device, self.dtype)
                previous_weight = self._previous_tile_weight(processed_tiles, out_top, out_bottom, out_left, out_right, full_h, full_w, device, self.dtype)
                region = output[:, :, out_top:out_bottom, out_left:out_right]
                blended = region.to(dtype=self.dtype).mul_(previous_weight).add_(tile.to(dtype=self.dtype).mul_(weight)).div_(previous_weight.add_(weight).clamp_min_(1e-6))
                region.copy_(blended.round_().clamp_(0, 255).to(torch.uint8))
                processed_tiles.append((out_top, out_bottom, out_left, out_right))
                del tile, tile_latent, weight, previous_weight, blended
                tile_no += 1
        del tile_latents
        return output

    @torch.inference_mode()
    def decode(self, lq_image, lq_latent=None, prompt="", seed=0, num_steps=4, ckpt_type=None, vae_encode=False, abort_callback=None, progress_callback=None, tiling_threshold=PID_TILING_THRESHOLD_DEFAULT):
        device = lq_image.device
        lq_image = lq_image.to(dtype=self.dtype)
        if lq_latent is not None:
            lq_latent = lq_latent.to(dtype=self.dtype)
        batch_size = lq_image.shape[0]
        resolved_threshold = resolve_pid_tiling_threshold(tiling_threshold)
        variant_label = "flux2" if normalize_pid_backbone(self.backbone) == "flux2" else "flux"
        encode_label = "VAE Encode" if vae_encode or lq_latent is None else "No VAE Encode"
        tiled = self._should_tile(lq_image, resolved_threshold)
        tile_plan = self._tile_plan(lq_image.shape[-2], lq_image.shape[-1], _pid_latent_downscale(self.backbone), resolved_threshold) if tiled else None
        ckpt_type = tile_plan["ckpt_type"] if tiled else self._direct_ckpt_type(lq_image, ckpt_type, resolved_threshold)
        threshold_label = pid_tiling_threshold_label(resolved_threshold)
        print(f"[PiD] variant={variant_label}, res={ckpt_type}, tiling_threshold={threshold_label}, conditioning={encode_label}, batch={batch_size}, input={int(lq_image.shape[-1])}x{int(lq_image.shape[-2])}, output={int(lq_image.shape[-1] * 4)}x{int(lq_image.shape[-2] * 4)}, tiled={tiled}")
        captions = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if len(captions) == 1 and batch_size > 1:
            captions = captions * batch_size
        caption_embs = self._encode_text_raw(captions, device)
        degrade_sigma = torch.zeros(batch_size, device=device, dtype=self.dtype)
        if tiled:
            x = self._decode_tiled(lq_image, lq_latent, caption_embs, degrade_sigma, seed, num_steps, tile_plan, abort_callback=abort_callback, progress_callback=progress_callback)
        else:
            if lq_latent is None:
                if progress_callback is not None:
                    progress_callback("PiD VAE Encode")
                lq_latent = self.encode_lq_image(lq_image)
            x = self._decode_patch(lq_latent.to(device=device, dtype=self.dtype), caption_embs, degrade_sigma, int(lq_image.shape[-2] * 4), int(lq_image.shape[-1] * 4), seed, num_steps, ckpt_type, abort_callback=abort_callback, progress_callback=progress_callback)
            if x is not None:
                x = self._float_to_uint8(x)
        del caption_embs, degrade_sigma
        return x


class PiDUpsamplerSession:
    def __init__(
        self,
        runtime,
        backbone,
        ckpt_type,
        *,
        init_pipe,
        profile,
        main_offloadobj=None,
        persistent_models=False,
        dtype=torch.bfloat16,
        tiling_threshold=PID_TILING_THRESHOLD_DEFAULT,
        attention_mode=None,
    ):
        self._runtime = runtime
        self.backbone = normalize_pid_backbone(backbone)
        self.tiling_threshold = normalize_pid_tiling_threshold(tiling_threshold)
        self.ckpt_types = pid_checkpoint_types_for_tiling_threshold(self.tiling_threshold)
        self.ckpt_type = ckpt_type if ckpt_type in self.ckpt_types else None
        self.init_pipe = init_pipe
        self.profile = profile
        self.main_offloadobj = main_offloadobj
        self.dtype = dtype
        self.persistent_models = bool(persistent_models)
        self.attention_mode = attention_mode

    def ensure_loaded(self):
        self._runtime.load(self.backbone, init_pipe=self.init_pipe, profile=self.profile, dtype=self.dtype, ckpt_types=self.ckpt_types)

    def unload_vram(self):
        self._runtime._unload_mmgp()

    def unload_main_model_vram(self):
        if self.main_offloadobj is not None:
            self.main_offloadobj.unload_all()

    def __getattr__(self, name):
        self.ensure_loaded()
        return getattr(self._runtime.upsampler, name)

    def decode_inputs(self, lq_image_ref, lq_latent_ref, **kwargs):
        lq_image = lq_image_ref[0]
        lq_latent = lq_latent_ref[0]
        lq_image_ref.clear()
        lq_latent_ref.clear()
        return self.decode(lq_image, lq_latent, **kwargs)

    def decode(self, *args, **kwargs):
        self.unload_main_model_vram()
        self.ensure_loaded()
        return self._runtime.decode(
            *args,
            ckpt_type=self.ckpt_type,
            persistent_models=self.persistent_models,
            tiling_threshold=self.tiling_threshold,
            attention_mode=self.attention_mode,
            **kwargs,
        )


class PiDRuntime:
    def __init__(self):
        self.upsampler = None
        self.offloadobj = None
        self.backbone = None
        self.profile = None
        self.ckpt_types = None
        self.dtype = torch.bfloat16

    def load(self, backbone, *, init_pipe, profile, dtype=torch.bfloat16, ckpt_types=None):
        backbone = normalize_pid_backbone(backbone)
        ckpt_types = normalize_pid_checkpoint_types(ckpt_types)
        if self.upsampler is not None and self.backbone == backbone and self.profile == profile and self.dtype == dtype and self.ckpt_types == ckpt_types:
            return

        self.release()
        from mmgp import offload

        self.upsampler = PiDUpsampler(backbone=backbone, dtype=dtype, ckpt_types=ckpt_types)
        pipe = self.upsampler.pipe_modules()
        kwargs = {}
        profile_no = init_pipe(pipe, kwargs, profile)
        _apply_pid_offload_budgets(pipe, kwargs)
        kwargs["pinnedMemory"] = False
        self.offloadobj = offload.profile(pipe, profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=-1, **kwargs)
        self.backbone = backbone
        self.profile = profile
        self.ckpt_types = ckpt_types
        self.dtype = dtype

    def session(self, backbone, ckpt_type, *, init_pipe, profile, main_offloadobj=None, persistent_models=False, dtype=torch.bfloat16, tiling_threshold=PID_TILING_THRESHOLD_DEFAULT, attention_mode=None):
        return PiDUpsamplerSession(
            self,
            backbone,
            ckpt_type,
            init_pipe=init_pipe,
            profile=profile,
            main_offloadobj=main_offloadobj,
            persistent_models=persistent_models,
            dtype=dtype,
            tiling_threshold=tiling_threshold,
            attention_mode=attention_mode,
        )

    def _unload_mmgp(self):
        if self.offloadobj is not None:
            self.offloadobj.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self):
        if self.offloadobj is not None:
            self.offloadobj.release()
            self.offloadobj = None
        self.upsampler = None
        self.backbone = None
        self.profile = None
        self.ckpt_types = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decode(self, *args, persistent_models=False, attention_mode=None, **kwargs):
        if self.upsampler is None:
            raise RuntimeError("PiD upsampler is not loaded.")
        try:
            with attention_config_shared_state(attention_mode, disable_sage_pre_ada=True):
                return self.upsampler.decode(*args, **kwargs)
        finally:
            if persistent_models:
                self._unload_mmgp()
            else:
                self.release()


_RUNTIME = PiDRuntime()


def get_pid_upsampler(backbone, ckpt_type, *, init_pipe, profile, main_offloadobj=None, persistent_models=False, dtype=torch.bfloat16, tiling_threshold=PID_TILING_THRESHOLD_DEFAULT, attention_mode=None):
    return _RUNTIME.session(
        backbone,
        ckpt_type,
        init_pipe=init_pipe,
        profile=profile,
        main_offloadobj=main_offloadobj,
        persistent_models=persistent_models,
        dtype=dtype,
        tiling_threshold=tiling_threshold,
        attention_mode=attention_mode,
    )


def release_models():
    _RUNTIME.release()
