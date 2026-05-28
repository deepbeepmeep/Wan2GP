from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from mmgp import offload as mmgp_offload
from shared.attention import pay_attention

EDITANYTHING_REF_START_BLOCK = 12
EDITANYTHING_REF_END_BLOCK = 35
EDITANYTHING_REF_CONTEXT_SCALE = 0.01
EDITANYTHING_REF_TOKEN_SCALE = 0.25
EDITANYTHING_ADALN_SCALE = 2.0


def _module_state(module_paths) -> dict[str, torch.Tensor]:
    paths = module_paths if isinstance(module_paths, (list, tuple)) else [module_paths]
    state = {}
    for path in paths:
        if not path or "edit_anything" not in os.path.basename(str(path)).lower():
            continue
        sd, _, _ = mmgp_offload.load_sd(path, writable_tensors=False)
        state.update(sd)
    return state


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


class _LoRALinear(torch.nn.Module):
    def __init__(self, base_linear: torch.nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        super().__init__()
        object.__setattr__(self, "base_linear", base_linear)
        self.lora_A = torch.nn.Parameter(lora_a, requires_grad=False)
        self.lora_B = torch.nn.Parameter(lora_b, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_linear(x)
        lora_dtype = self.lora_A.dtype
        lora_out = F.linear(F.linear(x.to(dtype=lora_dtype), self.lora_A), self.lora_B)
        return out.add(lora_out.to(device=out.device, dtype=out.dtype))


class EditAnythingRefAttention(torch.nn.Module):
    def __init__(self, base_attn: torch.nn.Module, state: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        object.__setattr__(self, "base_attn", base_attn)
        self.heads = int(base_attn.heads)
        self.dim_head = int(base_attn.dim_head)
        self.to_q = _LoRALinear(base_attn.to_q, state[f"{prefix}to_q.lora_A.weight"], state[f"{prefix}to_q.lora_B.weight"])
        self.to_k = _LoRALinear(base_attn.to_k, state[f"{prefix}to_k.lora_A.weight"], state[f"{prefix}to_k.lora_B.weight"])
        self.to_v = _LoRALinear(base_attn.to_v, state[f"{prefix}to_v.lora_A.weight"], state[f"{prefix}to_v.lora_B.weight"])
        self.to_out = _LoRALinear(base_attn.to_out[0], state[f"{prefix}to_out.0.lora_A.weight"], state[f"{prefix}to_out.0.lora_B.weight"])

    def forward(self, x_list: list[torch.Tensor], context_list: list[torch.Tensor] | None = None) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        context = context_list[0] if context_list is not None else x
        if context_list is not None:
            context_list.clear()
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        self.base_attn.q_norm(q)
        self.base_attn.k_norm(k)
        q = q.view(q.shape[0], -1, self.heads, self.dim_head)
        k = k.view(k.shape[0], -1, self.heads, self.dim_head)
        v = v.view(v.shape[0], -1, self.heads, self.dim_head)
        force_attention, attention_version = self.base_attn._resolve_attention_override()
        out = pay_attention([q, k, v], force_attention=force_attention, version=attention_version, recycle_q=True)
        out = out.flatten(2, 3)
        return self.to_out(out)


class EditAnythingRefVisualProj(torch.nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        fc1_weight = state["fc1.weight"]
        proj_weight = state["proj.weight"]
        self.fc1 = torch.nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0], bias="fc1.bias" in state)
        self.proj = torch.nn.Linear(proj_weight.shape[1], proj_weight.shape[0], bias="proj.bias" in state)
        self.norm = torch.nn.LayerNorm(proj_weight.shape[0])
        self.pos_embed = torch.nn.Parameter(state["pos_embed"], requires_grad=False)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)

    def forward(self, ref_latent: torch.Tensor, token_scale: float = EDITANYTHING_REF_TOKEN_SCALE) -> torch.Tensor:
        ref_frame = ref_latent.mean(dim=2)
        local = F.adaptive_avg_pool2d(ref_frame, (4, 8)).permute(0, 2, 3, 1).reshape(ref_frame.shape[0], 32, -1)
        global_mean = ref_frame.mean(dim=(-2, -1))
        global_std = ref_frame.std(dim=(-2, -1), unbiased=False)
        stats = torch.cat([global_mean, global_std], dim=-1).unsqueeze(1).expand(-1, local.shape[1], -1)
        tokens = torch.cat([local, stats], dim=-1)
        tokens = self.proj(F.silu(self.fc1(tokens)))
        tokens = self.norm(tokens)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]].to(device=tokens.device, dtype=tokens.dtype)
        return tokens * float(token_scale)


class EditAnythingRefAdaLNProj(torch.nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        fc1_weight = state["fc1.weight"]
        proj_weight = state["proj.weight"]
        self.fc1 = torch.nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0], bias="fc1.bias" in state)
        self.proj = torch.nn.Linear(proj_weight.shape[1], proj_weight.shape[0], bias="proj.bias" in state)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)

    def forward(self, ref_latent: torch.Tensor, adaln_scale: float = EDITANYTHING_ADALN_SCALE) -> torch.Tensor:
        ref_frame = ref_latent.mean(dim=2)
        avg_1x1 = F.adaptive_avg_pool2d(ref_frame, (1, 1)).flatten(1)
        avg_2x2 = F.adaptive_avg_pool2d(ref_frame, (2, 2)).flatten(1)
        max_1x1 = F.adaptive_max_pool2d(ref_frame, (1, 1)).flatten(1)
        pooled = torch.cat([avg_1x1, avg_2x2, max_1x1], dim=-1)
        return self.proj(F.silu(self.fc1(pooled))) * float(adaln_scale)


def install_editanything_modules(velocity_model: torch.nn.Module, module_paths, model_def: dict | None = None) -> None:
    state = _module_state(module_paths)
    if not state:
        return
    model_def = model_def or {}
    velocity_model.editanything_ref_start_block = int(model_def.get("ltx2_edit_anything_ref_start_block", EDITANYTHING_REF_START_BLOCK))
    velocity_model.editanything_ref_end_block = int(model_def.get("ltx2_edit_anything_ref_end_block", EDITANYTHING_REF_END_BLOCK))
    velocity_model.editanything_ref_context_scale = float(model_def.get("ltx2_edit_anything_ref_context_scale", EDITANYTHING_REF_CONTEXT_SCALE))
    velocity_model.editanything_ref_token_scale = float(model_def.get("ltx2_edit_anything_ref_token_scale", EDITANYTHING_REF_TOKEN_SCALE))
    velocity_model.editanything_adaln_scale = float(model_def.get("ltx2_edit_anything_adaln_scale", EDITANYTHING_ADALN_SCALE))
    visual_state = _strip_prefix(state, "ref_visual_proj.")
    if visual_state:
        velocity_model.editanything_ref_visual_proj = EditAnythingRefVisualProj(visual_state)
    adaln_state = _strip_prefix(state, "ref_adaln_proj.")
    if adaln_state:
        velocity_model.editanything_ref_adaln_proj = EditAnythingRefAdaLNProj(adaln_state)
    role_weight = state.get("role_embedding.embedding.weight")
    if role_weight is not None:
        role_embedding = torch.nn.Embedding(role_weight.shape[0], role_weight.shape[1])
        role_embedding.weight = torch.nn.Parameter(role_weight, requires_grad=False)
        velocity_model.editanything_role_embedding = role_embedding
    for block in getattr(velocity_model, "transformer_blocks", []):
        prefix = f"diffusion_model.transformer_blocks.{block.idx}.ref_attn."
        if f"{prefix}to_q.lora_A.weight" not in state:
            continue
        block.ref_attn = EditAnythingRefAttention(block.attn2, state, prefix)
        block.editanything_ref_start_block = velocity_model.editanything_ref_start_block
        block.editanything_ref_end_block = velocity_model.editanything_ref_end_block
        block.editanything_ref_context_scale = velocity_model.editanything_ref_context_scale
    velocity_model.editanything_module_loaded = True
    print("[WAN2GP][LTX2] EditAnything reference module installed.")


def build_editanything_reference_conditioning(
    transformer: torch.nn.Module,
    ref_images,
    height: int,
    width: int,
    video_encoder: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config=None,
):
    from .ltx_core.conditioning import VideoConditionByReferenceLatent
    from .ltx_core.model.video_vae import encode_video as vae_encode_video
    from .ltx_pipelines.utils.media_io import load_image_conditioning

    velocity_model = getattr(transformer, "velocity_model", transformer)
    if not getattr(velocity_model, "editanything_module_loaded", False) or not ref_images:
        return [], None, None
    ref_image = ref_images[0] if isinstance(ref_images, (list, tuple)) else ref_images
    image = load_image_conditioning(ref_image, height=height, width=width, dtype=dtype, device=device, resample="lanczos")
    ref_latent = vae_encode_video(image, video_encoder, tiling_config).to(dtype=dtype)
    conditionings = [VideoConditionByReferenceLatent(ref_latent, strength=1.0)]
    ref_context = ref_adaln = None
    visual_proj = getattr(velocity_model, "editanything_ref_visual_proj", None)
    if visual_proj is not None:
        visual_param = next(visual_proj.parameters())
        ref_context = visual_proj(ref_latent.to(device=device, dtype=visual_param.dtype), velocity_model.editanything_ref_token_scale).detach()
    adaln_proj = getattr(velocity_model, "editanything_ref_adaln_proj", None)
    if adaln_proj is not None:
        adaln_param = next(adaln_proj.parameters())
        ref_adaln = adaln_proj(ref_latent.to(device=device, dtype=adaln_param.dtype), velocity_model.editanything_adaln_scale).detach()
    return conditionings, ref_context, ref_adaln
