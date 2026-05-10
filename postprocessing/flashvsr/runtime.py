##### Enjoy this spagheti VRAM optimizations done by DeepBeepMeep !
# I am sure you are a nice person and as you copy this code, you will give me officially proper credits:
# Please link to https://github.com/deepbeepmeep/Wan2GP and @deepbeepmeep on twitter  
from __future__ import annotations

import gc
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

from mmgp import offload
from models.wan.modules.vae import WanVAE
from .attention_backend import require_sparge_attention
from .tcdecoder import build_tcdecoder
from .utils import Causal_LQ4x_Proj
from .wan_video_dit import WanModel, precompute_freqs_cis_3d


FLASHVSR_VARIANT_TINY_LONG = "tiny-long"
FLASHVSR_VARIANT_TINY = "tiny"
FLASHVSR_VARIANT_FULL = "full"

FLASHVSR_TOPK_RATIO = 0.0  # 0 = auto area-scaled ratio; >0 = fixed sparse attention ratio.
FLASHVSR_FULL_MIN_AUTO_TOPK_RATIO = 1.5
FLASHVSR_KV_CACHE_WINDOWS = 1  # Stream cache windows kept between denoise chunks; each window is two latent frames.
FLASHVSR_CONTINUE_CACHE_FRAMES = 11
FLASHVSR_COTENANTS_MAP = {"lq_proj": ["transformer"]}

WAN_1_3B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


@dataclass
class FlashVSRPaths:
    transformer: str
    lq_proj: str
    posi_prompt: str
    tcdecoder: str | None = None
    vae: str | None = None


def _preprocess_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converter = WanModel.state_dict_converter()
    state_dict, _ = converter.from_civitai(state_dict)
    return state_dict


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def _next_conditioning_frame_count(frame_count: int) -> int:
    padded = max(25, frame_count + 4)
    remainder = padded % 8
    if remainder != 1:
        padded += (1 - remainder) % 8
    return padded


def _aligned_output_size(height: int, width: int, scale: float) -> tuple[int, int]:
    target_h = max(1, int(height * scale))
    target_w = max(1, int(width * scale))
    return max(128, math.ceil(target_h / 128) * 128), max(128, math.ceil(target_w / 128) * 128)


def _conditioning_sizes(sample: torch.Tensor, scale: float) -> tuple[int, int, int, int]:
    _, frames, height, width = sample.shape
    output_height = max(1, int(height * scale))
    output_width = max(1, int(width * scale))
    padded_output_height, padded_output_width = _aligned_output_size(height, width, scale)
    pad_h = padded_output_height - output_height
    pad_w = padded_output_width - output_width
    if pad_h or pad_w:
        print(f"[FlashVSR] Edge padding output canvas {output_width}x{output_height} -> {padded_output_width}x{padded_output_height}; final crop restores {output_width}x{output_height}")
    return output_height, output_width, padded_output_height, padded_output_width


def _prepare_conditioning_range(sample: torch.Tensor, start: int, end: int, output_height: int, output_width: int, padded_output_height: int, padded_output_width: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    frames = int(sample.shape[1])
    pad_h = padded_output_height - output_height
    pad_w = padded_output_width - output_width
    frame_indices = [min(max(frame_idx, 0), frames - 1) for frame_idx in range(start, end)]
    lq = sample[:, frame_indices]
    if lq.dtype == torch.uint8:
        lq = lq.float().div_(127.5).sub_(1.0)
    else:
        lq = lq.detach().float().clamp_(-1.0, 1.0)
    lq = F.interpolate(lq.permute(1, 0, 2, 3).contiguous(), size=(output_height, output_width), mode="bicubic", align_corners=False)
    if pad_h or pad_w:
        lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="replicate")
    return lq.clamp_(-1.0, 1.0).to(dtype=dtype).permute(1, 0, 2, 3).contiguous()


def _pad_conditioning_frames(lq_video: torch.Tensor, target_frames: int) -> torch.Tensor:
    missing = target_frames - lq_video.shape[2]
    if missing <= 0:
        return lq_video[:, :, :target_frames]
    tail = lq_video[:, :, -1:].repeat(1, 1, missing, 1, 1)
    return torch.cat([lq_video, tail], dim=2)


def _crop_output_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] == (height, width):
        return frames
    return frames[..., :height, :width].contiguous()


def _decoded_frames_to_cpu(frames: torch.Tensor, frame_count: int, height: int, width: int) -> torch.Tensor:
    frames = frames.detach()[0, :, :frame_count, :height, :width]
    if frames.device.type == "cpu" and frames.dtype == torch.float32 and frames.is_contiguous():
        return frames
    frames_cpu = torch.empty(tuple(frames.shape), dtype=torch.float32, device="cpu")
    frames_cpu.copy_(frames)
    return frames_cpu


def _nested_tensors_to(value: Any, device: torch.device | str, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype or value.dtype)
    if isinstance(value, list):
        return [_nested_tensors_to(item, device, dtype) for item in value]
    return value


def _tcdecoder_mem_halo_latents(tcdecoder: torch.nn.Module) -> int:
    radius = 0.0
    jump = 1.0
    decoder = tcdecoder.taehv.decoder if hasattr(tcdecoder, "taehv") else tcdecoder.decoder
    for module in decoder:
        if isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else int(module.kernel_size)
            radius += ((kernel - 1) / 2) * jump
        elif module.__class__.__name__ == "MemBlock":
            for submodule in module.conv:
                if isinstance(submodule, torch.nn.Conv2d):
                    kernel = submodule.kernel_size[0] if isinstance(submodule.kernel_size, tuple) else int(submodule.kernel_size)
                    radius += ((kernel - 1) / 2) * jump
        elif isinstance(module, torch.nn.Upsample):
            scale = module.scale_factor[0] if isinstance(module.scale_factor, tuple) else module.scale_factor
            jump /= float(scale or 1)
    return max(1, int(math.ceil(radius)))


def _report_progress(progress_callback, phase: str, current_step: int | None = None, total_steps: int | None = None) -> None:
    if callable(progress_callback):
        progress_callback(phase, current_step, total_steps)


def _abort_requested(abort_callback) -> bool:
    return callable(abort_callback) and abort_callback()


def _apply_continue_cache(frames: torch.Tensor, continue_cache: Any) -> torch.Tensor:
    if not isinstance(continue_cache, dict):
        return frames
    tail = continue_cache.get("tail_frames")
    if not torch.is_tensor(tail) or tail.ndim != 4:
        return frames
    if tail.shape[0] != frames.shape[0] or tail.shape[-2:] != frames.shape[-2:]:
        return frames
    overlap = min(int(tail.shape[1]), int(frames.shape[1]))
    if overlap <= 0:
        return frames
    if frames.dtype == torch.uint8:
        if tail.dtype != torch.uint8:
            tail = tail.float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        frames[:, :overlap].copy_(tail[:, -overlap:].to(device=frames.device))
        return frames
    if tail.dtype == torch.uint8:
        tail = tail.to(device=frames.device, dtype=frames.dtype).div(127.5).sub(1.0)
    else:
        tail = tail.to(device=frames.device, dtype=frames.dtype)
    frames[:, :overlap].copy_(tail[:, -overlap:])
    return frames


def _make_continue_cache(frames: torch.Tensor, scale: float, variant: str, overlap_frames: int = FLASHVSR_CONTINUE_CACHE_FRAMES) -> dict[str, Any]:
    tail_len = min(overlap_frames, frames.shape[1])
    tail = frames[:, -tail_len:].detach().cpu()
    if tail.dtype != torch.uint8:
        tail = tail.float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return {"tail_frames": tail.contiguous(), "scale": scale, "variant": variant}


def _wavelet_color_fix(frames: torch.Tensor, lq_video: torch.Tensor) -> torch.Tensor:
    if frames.shape != lq_video[:, :, :frames.shape[2]].shape:
        return frames
    for start in range(0, frames.shape[2], 4):
        end = min(start + 4, frames.shape[2])
        frame_chunk = frames[:, :, start:end]
        lq_chunk = lq_video[:, :, start:end].to(device=frames.device, dtype=frames.dtype)
        mean_frames = frame_chunk.mean(dim=(3, 4), keepdim=True)
        std_frames = frame_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        mean_lq = lq_chunk.mean(dim=(3, 4), keepdim=True)
        std_lq = lq_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        frame_chunk.sub_(mean_frames).div_(std_frames).mul_(std_lq).add_(mean_lq).clamp_(-1.0, 1.0)
    return frames


def _wavelet_color_fix_from_sample(frames: torch.Tensor, sample: torch.Tensor, scale: float, output_height: int, output_width: int, padded_output_height: int, padded_output_width: int) -> torch.Tensor:
    step = 1 if frames.dtype == torch.uint8 else 4
    for start in range(0, min(int(frames.shape[2]), int(sample.shape[1])), step):
        end = min(start + step, int(frames.shape[2]), int(sample.shape[1]))
        frame_chunk = frames[:, :, start:end]
        if frames.dtype == torch.uint8:
            frame_float = frame_chunk.float()
            lq_chunk = sample[:, start:end].unsqueeze(0).to(device=frames.device, dtype=torch.float32)
            if sample.dtype != torch.uint8:
                lq_chunk.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)
            mean_frames = frame_float.mean(dim=(3, 4), keepdim=True)
            std_frames = frame_float.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
            mean_lq = lq_chunk.mean(dim=(3, 4), keepdim=True)
            std_lq = lq_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
            frame_float.sub_(mean_frames).div_(std_frames).mul_(std_lq).add_(mean_lq).round_().clamp_(0, 255)
            frame_chunk.copy_(frame_float.to(torch.uint8))
            del frame_float, lq_chunk, mean_frames, std_frames, mean_lq, std_lq
            continue
        lq_chunk = sample[:, start:end].unsqueeze(0).to(device=frames.device, dtype=frames.dtype)
        if sample.dtype == torch.uint8:
            lq_chunk.div_(127.5).sub_(1.0)
        else:
            lq_chunk.clamp_(-1.0, 1.0)
        mean_frames = frame_chunk.mean(dim=(3, 4), keepdim=True)
        std_frames = frame_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        mean_lq = lq_chunk.mean(dim=(3, 4), keepdim=True)
        std_lq = lq_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        frame_chunk.sub_(mean_frames).div_(std_frames).mul_(std_lq).add_(mean_lq).clamp_(-1.0, 1.0)
        del lq_chunk, mean_frames, std_frames, mean_lq, std_lq
    return frames


def _denoise_stream_chunk(
    dit: WanModel,
    x: torch.Tensor,
    context: torch.Tensor | None,
    lq_layer_chunks: list[list[torch.Tensor | None]],
    block_cache_k: list[torch.Tensor | None],
    block_cache_v: list[torch.Tensor | None],
    chunk_index: int,
    timestep_embed: torch.Tensor,
    timestep_mod: torch.Tensor,
    *,
    topk_ratio: float = 2.0,
    kv_ratio: float = FLASHVSR_KV_CACHE_WINDOWS,
    local_range: int = 9,
    cache_next: bool = True,
    abort_callback=None,
) -> tuple[torch.Tensor | None, list[torch.Tensor | None], list[torch.Tensor | None]]:
    x, (frames, height, width) = dit.patchify(x)
    win = (2, 8, 8)
    seqlen = frames // win[0]
    window_size = win[0] * height * width // 128
    topk = int(window_size * window_size * topk_ratio) - 1
    kv_len = max(1, int(kv_ratio))
    if chunk_index == 0:
        freqs_t = dit.freqs[0][:frames]
    else:
        start = 4 + chunk_index * 2
        freqs_t = dit.freqs[0][start:start + frames]
    freqs = tuple((freq.real.to(device=x.device, dtype=x.dtype), freq.imag.to(device=x.device, dtype=x.dtype)) for freq in (freqs_t, dit.freqs[1][:height], dit.freqs[2][:width]))
    for block_id, block in enumerate(dit.blocks):
        if _abort_requested(abort_callback):
            return None, block_cache_k, block_cache_v
        if block_id < len(lq_layer_chunks[0]):
            offset = 0
            for chunk in lq_layer_chunks:
                lq = chunk[block_id].to(x.device, dtype=x.dtype)
                next_offset = offset + lq.shape[1]
                x[:, offset:next_offset].add_(lq)
                offset = next_offset
                chunk[block_id] = None
                del lq
        cache_refs = None
        if block_cache_k[block_id] is not None:
            cache_refs = [block_cache_k[block_id].to(x.device, dtype=x.dtype), block_cache_v[block_id].to(x.device, dtype=x.dtype)]
            block_cache_k[block_id] = None
            block_cache_v[block_id] = None
        x_ref = [x]
        x = None
        x, next_cache_k, next_cache_v = block(
            x_ref, context, timestep_mod, freqs, frames, height, width, seqlen, topk,
            block_id=block_id, kv_len=kv_len, is_stream=True,
            pre_cache_refs=cache_refs, local_range=local_range, cache_next=cache_next,
        )
        x_ref.clear()
        block_cache_k[block_id] = next_cache_k
        del next_cache_k
        block_cache_v[block_id] = next_cache_v
        del next_cache_v, cache_refs
        if _abort_requested(abort_callback):
            return None, block_cache_k, block_cache_v
    x = dit.head([x], timestep_embed)
    return dit.unpatchify([x], (frames, height, width)), block_cache_k, block_cache_v


class FlashVSRRuntime:
    def __init__(self) -> None:
        self.variant: str | None = None
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dit: WanModel | None = None
        self.lq_proj: Causal_LQ4x_Proj | None = None
        self.tcdecoder: torch.nn.Module | None = None
        self.vae: WanVAE | None = None
        self.offloadobj = None
        self.prompt_context: torch.Tensor | None = None
        self.timestep: torch.Tensor | None = None
        self.timestep_embed: torch.Tensor | None = None
        self.timestep_mod: torch.Tensor | None = None
        self.profile = None

    def load(self, paths: FlashVSRPaths, variant: str, profile, init_pipe) -> None:
        require_sparge_attention()
        variant = variant or FLASHVSR_VARIANT_TINY_LONG
        if self.dit is not None and self.variant == variant and self.profile == profile:
            return
        self.release()
        self.variant = variant
        self.profile = profile
        with init_empty_weights(include_buffers=True), _default_dtype(self.dtype):
            self.dit = WanModel(**WAN_1_3B_CONFIG).eval()
            self.lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).eval()
        self.dit._offload_hooks = ["reinit_cross_kv"]
        self.lq_proj._offload_hooks = ["stream_forward"]
        offload.load_model_data(self.dit, paths.transformer, writable_tensors=False, preprocess_sd=_preprocess_transformer_state_dict, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
        self.dit.freqs = precompute_freqs_cis_3d(WAN_1_3B_CONFIG["dim"] // WAN_1_3B_CONFIG["num_heads"])
        offload.load_model_data(self.lq_proj, paths.lq_proj, writable_tensors=False, default_dtype=self.dtype, verboseLevel=-1)
        self.dit.requires_grad_(False)
        self.lq_proj.requires_grad_(False)
        self.prompt_context = load_file(paths.posi_prompt, device="cpu")["context"].to(self.dtype)
        pipe = {"transformer": self.dit, "lq_proj": self.lq_proj}
        if variant in (FLASHVSR_VARIANT_TINY, FLASHVSR_VARIANT_TINY_LONG):
            self.tcdecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=self.dtype, new_latent_channels=16 + 768).eval()
            self.tcdecoder._offload_hooks = ["decode_video"]
            offload.load_model_data(self.tcdecoder, paths.tcdecoder, writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
            self.tcdecoder.requires_grad_(False)
            pipe["tcdecoder"] = self.tcdecoder
        else:
            self.vae = WanVAE(vae_pth=paths.vae, dtype=self.dtype, upsampler_factor=1, device="cpu")
            self.vae.device = self.device
            self.vae.model.requires_grad_(False)
            pipe["vae"] = self.vae.model
        kwargs = {"coTenantsMap": FLASHVSR_COTENANTS_MAP}
        profile_no = init_pipe(pipe, kwargs, profile)
        self.offloadobj = offload.profile(pipe, profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=self.dtype, verboseLevel=-1, **kwargs)

    def _prepare_run_state(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("FlashVSR requires CUDA.")
        context = self.prompt_context.to(self.device, dtype=self.dtype)
        self.dit.reinit_cross_kv(context)
        self.timestep = torch.tensor([1000.0], device=self.device, dtype=self.dtype)
        self.timestep_embed = self.dit.time_embedding(_sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.timestep_mod = self.dit.time_projection(self.timestep_embed).unflatten(1, (6, self.dit.dim))

    def _clear_runtime_caches(self) -> None:
        if self.dit is not None:
            self.dit.clear_cross_kv()
        if self.lq_proj is not None:
            self.lq_proj.clear_cache()
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        self.timestep = None
        self.timestep_embed = None
        self.timestep_mod = None

    def _unload_mmgp(self) -> None:
        self._clear_runtime_caches()
        if self.offloadobj is not None:
            self.offloadobj.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_tcdecoder(self, latents: torch.Tensor, sample: torch.Tensor, lq_start: int, lq_end: int, output_height: int, output_width: int, padded_output_height: int, padded_output_width: int, tile_size: int, tile_mems: dict[tuple[int, int], Any] | None, abort_callback=None, progress_callback=None, progress_step: int | None = None, progress_total: int | None = None) -> tuple[torch.Tensor | None, dict[tuple[int, int], Any] | None]:
        if self.tcdecoder is None:
            raise RuntimeError("FlashVSR tiny variants require TCDecoder.")
        _report_progress(progress_callback, "TCDecoder Decoding", progress_step, progress_total)
        tile_size = int(tile_size or 0)
        cur_lq = _prepare_conditioning_range(sample, lq_start, lq_end, output_height, output_width, padded_output_height, padded_output_width, dtype=self.dtype).unsqueeze(0)
        if tile_size <= 0 or (padded_output_height <= tile_size and padded_output_width <= tile_size):
            cur_lq = cur_lq.to(self.device, dtype=self.dtype)
            frames = self.tcdecoder.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cur_lq).transpose(1, 2).mul_(2).sub_(1)
            del cur_lq
            _report_progress(progress_callback, "TCDecoder Decoding", progress_step + 1 if progress_step is not None else None, progress_total)
            return frames, tile_mems

        halo = _tcdecoder_mem_halo_latents(self.tcdecoder)
        latent_tile = max(1, tile_size // 8)
        latent_height = padded_output_height // 8
        latent_width = padded_output_width // 8
        tile_mems = {} if tile_mems is None else tile_mems
        frames_out = None
        for latent_y0 in range(0, latent_height, latent_tile):
            latent_y1 = min(latent_y0 + latent_tile, latent_height)
            write_y0, write_y1 = latent_y0 * 8, min(latent_y1 * 8, output_height)
            if write_y1 <= write_y0:
                continue
            expanded_y0, expanded_y1 = max(0, latent_y0 - halo), min(latent_height, latent_y1 + halo)
            crop_y0 = (latent_y0 - expanded_y0) * 8
            for latent_x0 in range(0, latent_width, latent_tile):
                if _abort_requested(abort_callback):
                    del cur_lq
                    return None, tile_mems
                latent_x1 = min(latent_x0 + latent_tile, latent_width)
                write_x0, write_x1 = latent_x0 * 8, min(latent_x1 * 8, output_width)
                if write_x1 <= write_x0:
                    continue
                expanded_x0, expanded_x1 = max(0, latent_x0 - halo), min(latent_width, latent_x1 + halo)
                crop_x0 = (latent_x0 - expanded_x0) * 8
                tile_key = (latent_y0, latent_x0)
                saved_mem = tile_mems.get(tile_key)
                if saved_mem is None:
                    self.tcdecoder.clean_mem()
                else:
                    self.tcdecoder.mem = _nested_tensors_to(saved_mem, self.device, self.dtype)
                cur_lq_tile = cur_lq[:, :, :, expanded_y0 * 8:expanded_y1 * 8, expanded_x0 * 8:expanded_x1 * 8].contiguous().to(self.device, dtype=self.dtype)
                cur_latents = latents[:, :, :, expanded_y0:expanded_y1, expanded_x0:expanded_x1].to(self.device, dtype=self.dtype)
                tile_frames = self.tcdecoder.decode_video(cur_latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cur_lq_tile).transpose(1, 2).mul_(2).sub_(1)
                tile_mems[tile_key] = _nested_tensors_to(self.tcdecoder.mem, "cpu")
                self.tcdecoder.clean_mem()
                tile_frames = tile_frames[:, :, :, crop_y0:crop_y0 + latent_y1 * 8 - latent_y0 * 8, crop_x0:crop_x0 + latent_x1 * 8 - latent_x0 * 8]
                if frames_out is None:
                    frames_out = torch.empty((tile_frames.shape[0], tile_frames.shape[1], tile_frames.shape[2], output_height, output_width), dtype=torch.float32, device="cpu")
                tile_cpu = tile_frames[:, :, :, :write_y1 - write_y0, :write_x1 - write_x0].detach().cpu().float()
                frames_out[:, :, :, write_y0:write_y1, write_x0:write_x1].copy_(tile_cpu)
                del cur_lq_tile, cur_latents, tile_frames, tile_cpu
        del cur_lq
        _report_progress(progress_callback, "TCDecoder Decoding", progress_step + 1 if progress_step is not None else None, progress_total)
        return frames_out, tile_mems

    def release(self) -> None:
        self._clear_runtime_caches()
        if self.offloadobj is not None:
            self.offloadobj.release()
            self.offloadobj = None
        self.dit = None
        self.lq_proj = None
        self.tcdecoder = None
        self.vae = None
        self.prompt_context = None
        self.timestep = None
        self.timestep_embed = None
        self.timestep_mod = None
        self.variant = None
        self.profile = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def upscale(
        self,
        sample: torch.Tensor,
        scale: float,
        *,
        seed: int = 0,
        continue_cache: Any = None,
        return_continue_cache: bool = False,
        persistent_models: bool = False,
        vae_tile_size: int | None = None,
        topk_ratio: float = FLASHVSR_TOPK_RATIO,
        abort_callback=None,
        progress_callback=None,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        if self.dit is None or self.lq_proj is None:
            raise RuntimeError("FlashVSR models are not loaded.")
        def abort_result():
            self._unload_mmgp()
            if not persistent_models:
                self.release()
            return None, None

        input_frames = sample.shape[1]
        num_frames = _next_conditioning_frame_count(input_frames)
        output_height, output_width, padded_output_height, padded_output_width = _conditioning_sizes(sample, scale)
        configured_topk_ratio = max(0.0, min(4.0, float(topk_ratio or 0.0)))
        if configured_topk_ratio > 0:
            topk_ratio = configured_topk_ratio
            print(f"[FlashVSR] Sparse top-k ratio fixed to {topk_ratio:.3f}")
        else:
            raw_topk_ratio = min(2.0, 2.0 * 768 * 1280 / max(int(padded_output_height) * int(padded_output_width), 1))
            topk_ratio = max(raw_topk_ratio, FLASHVSR_FULL_MIN_AUTO_TOPK_RATIO)
            if topk_ratio != raw_topk_ratio:
                print(f"[FlashVSR] Sparse top-k ratio adjusted to {topk_ratio:.3f} for {padded_output_width}x{padded_output_height} (minimum; raw auto {raw_topk_ratio:.3f})")
            elif topk_ratio < 2.0:
                print(f"[FlashVSR] Sparse top-k ratio adjusted to {topk_ratio:.3f} for {padded_output_width}x{padded_output_height}")
        self._prepare_run_state()
        self.lq_proj.clear_cache()
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        print(f"[FlashVSR] Stream KV cache windows: {max(1, int(FLASHVSR_KV_CACHE_WINDOWS))}")
        tcdecoder_tile_size = int(vae_tile_size or 0) if self.tcdecoder is not None else 0
        tcdecoder_tile_mems = None
        if self.tcdecoder is not None:
            if tcdecoder_tile_size > 0 and (padded_output_height > tcdecoder_tile_size or padded_output_width > tcdecoder_tile_size):
                print(f"[FlashVSR] TCDecoder spatial tiling policy: tile_size={tcdecoder_tile_size}px, halo={_tcdecoder_mem_halo_latents(self.tcdecoder) * 8}px")
                tcdecoder_tile_mems = {}
            else:
                print("[FlashVSR] TCDecoder spatial tiling policy: tile_size=0px")
        generator = torch.Generator(device="cpu").manual_seed(0 if seed is None or seed < 0 else int(seed))
        latents = torch.empty((1, 16, (num_frames - 1) // 4, padded_output_height // 8, padded_output_width // 8), device="cpu", dtype=self.dtype)
        latents.normal_(generator=generator)
        process_total = (num_frames - 1) // 8 - 2
        pre_cache_k = [None] * len(self.dit.blocks)
        pre_cache_v = [None] * len(self.dit.blocks)
        frames_out = None
        frames_cursor = 0
        lq_pre_idx = 0
        lq_cur_idx = 0
        _report_progress(progress_callback, "Denoising", 0, process_total)
        for process_idx in tqdm(range(process_total), desc="FlashVSR"):
            if _abort_requested(abort_callback):
                return abort_result()
            lq_layer_chunks = []
            torch.cuda.empty_cache()
            if process_idx == 0:
                for inner_idx in range(7):
                    if _abort_requested(abort_callback):
                        return abort_result()
                    lq_chunk = _prepare_conditioning_range(sample, max(0, inner_idx * 4 - 3), (inner_idx + 1) * 4 - 3, output_height, output_width, padded_output_height, padded_output_width, dtype=self.dtype).unsqueeze(0).to(self.device, dtype=self.dtype)
                    lq_list = [lq_chunk]
                    del lq_chunk
                    cur = self.lq_proj.stream_forward(lq_list)
                    if cur is not None:
                        lq_layer_chunks.append(cur)
                    del cur
                lq_cur_idx = 21
                latent_start, latent_end = 0, 6
                cur_latents = latents[:, :, :6].to(self.device, dtype=self.dtype)
            else:
                for inner_idx in range(2):
                    if _abort_requested(abort_callback):
                        return abort_result()
                    lq_start = process_idx * 8 + 17 + inner_idx * 4
                    lq_chunk = _prepare_conditioning_range(sample, lq_start, lq_start + 4, output_height, output_width, padded_output_height, padded_output_width, dtype=self.dtype).unsqueeze(0).to(self.device, dtype=self.dtype)
                    lq_list = [lq_chunk]
                    del lq_chunk
                    cur = self.lq_proj.stream_forward(lq_list)
                    if cur is not None:
                        lq_layer_chunks.append(cur)
                    del cur
                lq_cur_idx = process_idx * 8 + 21
                latent_start, latent_end = 4 + process_idx * 2, 6 + process_idx * 2
                cur_latents = latents[:, :, latent_start:latent_end].to(self.device, dtype=self.dtype)
            torch.cuda.empty_cache()

            noise_pred, pre_cache_k, pre_cache_v = _denoise_stream_chunk(
                self.dit, cur_latents, None, lq_layer_chunks, pre_cache_k, pre_cache_v, process_idx,
                self.timestep_embed, self.timestep_mod, topk_ratio=topk_ratio, cache_next=process_idx + 1 < process_total, abort_callback=abort_callback,
            )
            if noise_pred is None:
                return abort_result()
            cur_latents = cur_latents - noise_pred
            _report_progress(progress_callback, "Denoising", process_idx + 1, process_total)
            if self.variant == FLASHVSR_VARIANT_TINY_LONG:
                cur_frames, tcdecoder_tile_mems = self._decode_tcdecoder(cur_latents, sample, lq_pre_idx, lq_cur_idx, output_height, output_width, padded_output_height, padded_output_width, tcdecoder_tile_size, tcdecoder_tile_mems, abort_callback=abort_callback, progress_callback=progress_callback, progress_step=process_idx, progress_total=process_total)
                if cur_frames is None:
                    return abort_result()
                cur_frames = _crop_output_frames(cur_frames.detach().cpu(), output_height, output_width)
                copy_frames = min(int(cur_frames.shape[2]), input_frames - frames_cursor)
                if copy_frames > 0:
                    if frames_out is None:
                        frames_out = torch.empty((cur_frames.shape[0], cur_frames.shape[1], input_frames, output_height, output_width), dtype=torch.float32, device="cpu")
                    frames_out[:, :, frames_cursor:frames_cursor + copy_frames].copy_(cur_frames[:, :, :copy_frames].float())
                    frames_cursor += copy_frames
                lq_pre_idx = lq_cur_idx
                del cur_frames
            else:
                latents[:, :, latent_start:latent_end].copy_(cur_latents.detach().cpu())
            lq_layer_chunks = None
        self.lq_proj.clear_cache()
        pre_cache_k = pre_cache_v = None
        self.dit.clear_cross_kv()
        gc.collect()
        if self.variant == FLASHVSR_VARIANT_TINY_LONG:
            frames = frames_out
        else:
            if self.variant == FLASHVSR_VARIANT_TINY:
                if _abort_requested(abort_callback):
                    return abort_result()
                self.tcdecoder.clean_mem()
                frames_out = None
                frames_cursor = 0
                lq_pre_idx = 0
                for decode_idx in range(process_total):
                    if _abort_requested(abort_callback):
                        return abort_result()
                    if decode_idx == 0:
                        lq_cur_idx = 21
                        latent_start, latent_end = 0, 6
                    else:
                        lq_cur_idx = decode_idx * 8 + 21
                        latent_start, latent_end = 4 + decode_idx * 2, 6 + decode_idx * 2
                    cur_latents = latents[:, :, latent_start:latent_end].to(self.device, dtype=self.dtype)
                    cur_frames, tcdecoder_tile_mems = self._decode_tcdecoder(cur_latents, sample, lq_pre_idx, lq_cur_idx, output_height, output_width, padded_output_height, padded_output_width, tcdecoder_tile_size, tcdecoder_tile_mems, abort_callback=abort_callback, progress_callback=progress_callback, progress_step=decode_idx, progress_total=process_total)
                    if cur_frames is None:
                        return abort_result()
                    cur_frames = _crop_output_frames(cur_frames.detach().cpu(), output_height, output_width)
                    copy_frames = min(int(cur_frames.shape[2]), input_frames - frames_cursor)
                    if copy_frames > 0:
                        if frames_out is None:
                            frames_out = torch.empty((cur_frames.shape[0], cur_frames.shape[1], input_frames, output_height, output_width), dtype=torch.float32, device="cpu")
                        frames_out[:, :, frames_cursor:frames_cursor + copy_frames].copy_(cur_frames[:, :, :copy_frames].float())
                        frames_cursor += copy_frames
                    lq_pre_idx = lq_cur_idx
                    del cur_latents, cur_frames
                frames = frames_out
            else:
                if _abort_requested(abort_callback):
                    return abort_result()
                _report_progress(progress_callback, "VAE Decoding")
                if self.vae is None:
                    raise RuntimeError("FlashVSR full variant requires the Wan VAE.")
                vae_tile_size = int(vae_tile_size or 0)
                print(f"[FlashVSR] Wan VAE tiling policy: tile_size={vae_tile_size}px")
                frames = self.vae.decode_to_cpu_uint8([latents[0]], vae_tile_size, target_frames=input_frames, target_height=output_height, target_width=output_width)[0]
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        latents = frames_out = pre_cache_k = pre_cache_v = tcdecoder_tile_mems = None
        noise_pred = cur_latents = lq_layer_chunks = None
        lq_chunk = cur = cur_lq = cur_frames = None
        if torch.is_tensor(frames) and frames.dtype == torch.uint8 and frames.ndim == 4:
            if frames.shape[1:] != (input_frames, output_height, output_width):
                frames = frames[:, :input_frames, :output_height, :output_width].contiguous()
        else:
            decoded_frames = frames
            frames = _decoded_frames_to_cpu(decoded_frames, input_frames, output_height, output_width)
            del decoded_frames
        gc.collect()
        _report_progress(progress_callback, "Color Correction")
        _wavelet_color_fix_from_sample(frames.unsqueeze(0), sample, scale, output_height, output_width, output_height, output_width)
        if frames.dtype != torch.uint8:
            frames.clamp_(-1.0, 1.0)
        frames = _apply_continue_cache(frames, continue_cache)
        cache = _make_continue_cache(frames, scale, self.variant) if return_continue_cache else None
        sample = None
        self._unload_mmgp()
        if not persistent_models:
            self.release()
        return frames, cache


_RUNTIME = FlashVSRRuntime()


def upscale_video(
    sample: torch.Tensor,
    scale: float,
    paths: FlashVSRPaths,
    *,
    variant: str = FLASHVSR_VARIANT_TINY_LONG,
    seed: int = 0,
    continue_cache: Any = None,
    return_continue_cache: bool = False,
    persistent_models: bool = False,
    vae_tile_size: int | None = None,
    topk_ratio: float = FLASHVSR_TOPK_RATIO,
    init_pipe,
    profile,
    abort_callback=None,
    progress_callback=None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    _report_progress(progress_callback, "Caching")
    _RUNTIME.load(paths, variant, profile=profile, init_pipe=init_pipe)
    try:
        result = _RUNTIME.upscale(sample, scale, seed=seed, continue_cache=continue_cache, return_continue_cache=return_continue_cache, persistent_models=persistent_models, vae_tile_size=vae_tile_size, topk_ratio=topk_ratio, abort_callback=abort_callback, progress_callback=progress_callback)
        if result[0] is None:
            if persistent_models:
                _RUNTIME._unload_mmgp()
            else:
                _RUNTIME.release()
        return result
    except Exception:
        if persistent_models:
            _RUNTIME._unload_mmgp()
        else:
            _RUNTIME.release()
        raise


def release_models() -> None:
    _RUNTIME.release()
