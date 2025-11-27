from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    EmbedNDFlux2,
    LastLayer,
    MLPEmbedder,
    Modulation,
    SingleStreamBlock,
    timestep_embedding,
    DistilledGuidance,
    ChromaModulationOut,
    SigLIPMultiFeatProjModel,
)
from .modules.lora import LinearLora, replace_linear_with_lora
from .radiance import apply_radiance_head, inject_radiance_modules


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    chroma: bool = False
    eso: bool = False
    radiance: bool = False
    radiance_patch_size: int = 16
    radiance_hidden_size: int = 64
    radiance_mlp_ratio: int = 4
    radiance_depth: int = 4
    radiance_max_freqs: int = 8
    radiance_tile_size: int = 0
    radiance_final_head_type: str = "conv"
    single_linear1_mlp_ratio: float | None = None
    single_mlp_hidden_ratio: float | None = None
    double_mlp_ratio: float | None = None
    double_linear1_mlp_ratio: float | None = None
    flux2: bool = False

class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")
    
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.chroma = params.chroma
        self.is_flux2 = getattr(params, "flux2", False)
        self.radiance = getattr(params, "radiance", False)
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        if self.is_flux2:
            self.pe_embedder = EmbedNDFlux2(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        else:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = (
            nn.Identity()
            if self.radiance
            else nn.Linear(self.in_channels, self.hidden_size, bias=not self.is_flux2)
        )

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=not self.is_flux2)
            if params.guidance_embed
            else None
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=not self.is_flux2)
        if self.is_flux2:
            self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, bias=False)
            self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, bias=False)
            self.single_stream_modulation = Modulation(self.hidden_size, double=False, bias=False)
        if self.chroma:
            self.distilled_guidance_layer = DistilledGuidance(
                        in_dim=64,
                        hidden_dim=5120,
                        out_dim=3072, 
                        n_layers=5,
                )
        else:
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=not self.is_flux2)
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, bias=not self.is_flux2) if not self.is_flux2 else nn.Identity()

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    double_mlp_ratio=getattr(params, "double_mlp_ratio", None),
                    double_linear1_mlp_ratio=getattr(params, "double_linear1_mlp_ratio", None),
                    mod_bias=not self.is_flux2,
                    mlp_bias=not self.is_flux2,
                    proj_bias=not self.is_flux2,
                    qkv_bias=params.qkv_bias,
                    shared_modulation = self.chroma or self.is_flux2,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    shared_modulation=self.chroma or self.is_flux2,
                    single_linear1_mlp_ratio=getattr(params, "single_linear1_mlp_ratio", None),
                    single_mlp_hidden_ratio=getattr(params, "single_mlp_hidden_ratio", None),
                    qk_scale=None,
                    linear_bias=not self.is_flux2,
                    modulation_bias=not self.is_flux2,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if self.radiance:
            inject_radiance_modules(self, params)
            self.final_layer = None
        else:
            self.final_layer = LastLayer(
                self.hidden_size,
                1,
                self.out_channels,
                chroma_modulation=self.chroma,
                use_linear=True,
                linear_bias=not self.is_flux2,
                modulation_bias=not self.is_flux2,
            )

    def _apply_final_layer(self, tokens: Tensor, vec):
        final_layer = self.final_layer
        normed = final_layer.norm_final(tokens)
        if self.chroma:
            shift, scale = vec
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)
        else:
            shift, scale = final_layer.adaLN_modulation(vec).chunk(2, dim=1)
        modulated = torch.addcmul(shift[:, None, :], 1 + scale[:, None, :], normed)
        if final_layer.linear is None:
            raise RuntimeError("Final layer projection is not available.")
        base_tokens = final_layer.linear(modulated)
        return modulated, base_tokens

    def preprocess_loras(self, model_type, sd):
        new_sd = {}
        if len(sd) == 0: return sd

        def swap_scale_shift(weight):
            shift, scale = weight.chunk(2, dim=0)
            new_weight = torch.cat([scale, shift], dim=0)
            return new_weight

        first_key= next(iter(sd))
        if first_key.startswith("lora_unet_"):
            new_sd = {}
            print("Converting Lora Safetensors format to Lora Diffusers format")
            repl_list = ["linear1", "linear2", "modulation", "img_attn", "txt_attn", "img_mlp", "txt_mlp", "img_mod", "txt_mod"]
            src_list = ["_" + k + "." for k in repl_list]
            src_list2 = ["_" + k + "_" for k in repl_list]
            tgt_list = ["." + k + "." for k in repl_list]

            for k,v in sd.items():
                k = k.replace("lora_unet_blocks_","diffusion_model.blocks.")
                k = k.replace("lora_unet__blocks_","diffusion_model.blocks.")
                k = k.replace("lora_unet_single_blocks_","diffusion_model.single_blocks.")
                k = k.replace("lora_unet_double_blocks_","diffusion_model.double_blocks.")

                for s,s2, t in zip(src_list, src_list2, tgt_list):
                    k = k.replace(s,t)
                    k = k.replace(s2,t)

                k = k.replace("lora_up","lora_B")
                k = k.replace("lora_down","lora_A")

                new_sd[k] = v

        elif first_key.startswith("transformer."):
            root_src = ["time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2", "time_text_embed.text_embedder.linear_1", "time_text_embed.text_embedder.linear_2",
                    "time_text_embed.guidance_embedder.linear_1", "time_text_embed.guidance_embedder.linear_2",
                    "x_embedder", "context_embedder", "proj_out" ]

            root_tgt = ["time_in.in_layer", "time_in.out_layer", "vector_in.in_layer", "vector_in.out_layer",
                    "guidance_in.in_layer", "guidance_in.out_layer",
                    "img_in", "txt_in", "final_layer.linear" ]

            double_src = ["norm1.linear", "norm1_context.linear", "attn.norm_q",  "attn.norm_k", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2", "attn.to_out.0" ,"attn.to_add_out", "attn.to_out", ".attn.to_", ".attn.add_q_proj.", ".attn.add_k_proj.", ".attn.add_v_proj.",  ] 
            double_tgt = ["img_mod.lin", "txt_mod.lin", "img_attn.norm.query_norm", "img_attn.norm.key_norm", "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2", "img_attn.proj", "txt_attn.proj", "img_attn.proj", ".img_attn.", ".txt_attn.q.", ".txt_attn.k.", ".txt_attn.v."] 

            single_src = ["norm.linear", "attn.norm_q", "attn.norm_k", "proj_out",".attn.to_q.", ".attn.to_k.", ".attn.to_v.", ".proj_mlp."]
            single_tgt = ["modulation.lin","norm.query_norm", "norm.key_norm", "linear2", ".linear1_attn_q.", ".linear1_attn_k.", ".linear1_attn_v.", ".linear1_mlp."]


            for k,v in sd.items():
                if k.startswith("transformer.single_transformer_blocks"):
                    k = k.replace("transformer.single_transformer_blocks", "diffusion_model.single_blocks")
                    for src, tgt in zip(single_src, single_tgt):
                        k = k.replace(src, tgt)
                elif k.startswith("transformer.transformer_blocks"):
                    k = k.replace("transformer.transformer_blocks", "diffusion_model.double_blocks")
                    for src, tgt in zip(double_src, double_tgt):
                        k = k.replace(src, tgt)
                else:
                    k = k.replace("transformer.", "diffusion_model.")
                    for src, tgt in zip(root_src, root_tgt):
                        k = k.replace(src, tgt)

                    if "norm_out.linear" in k:
                        if "lora_B" in k:
                            v = swap_scale_shift(v)
                        k = k.replace("norm_out.linear", "final_layer.adaLN_modulation.1")            
                new_sd[k] = v
        # elif not first_key.startswith("diffusion_model.") and not first_key.startswith("transformer."):
        #     for k,v in sd.items():
        #         if "double" in k:
        #             k = k.replace(".processor.proj_lora1.", ".img_attn.proj.lora_")
        #             k = k.replace(".processor.proj_lora2.", ".txt_attn.proj.lora_")
        #             k = k.replace(".processor.qkv_lora1.", ".img_attn.qkv.lora_")
        #             k = k.replace(".processor.qkv_lora2.", ".txt_attn.qkv.lora_")
        #         else:
        #             k = k.replace(".processor.qkv_lora.", ".linear1_qkv.lora_")
        #             k = k.replace(".processor.proj_lora.", ".linear2.lora_")

        #         k = "diffusion_model." + k
        #         new_sd[k] = v
        #     from mmgp import safetensors2
        #     safetensors2.torch_write_file(new_sd, "fff.safetensors")
        else:
            new_sd = sd
        return new_sd    

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt_list,
        txt_ids_list,
        timesteps: Tensor,
        y_list,
        img_len = 0,
        guidance: Tensor | None = None,
        callback= None,
        pipeline =None,
        siglip_embedding = None,
        siglip_embedding_ids = None,
    ) -> Tensor:

        sz = len(txt_list)
        height = width = None
        base_image_list = None
        if self.radiance:
            patch_size = self.patch_size
            # Determine spatial dimensions from image ids.
            base_ids = img_ids[:, :img_len, :]
            height = int(base_ids[..., 1].max().item() + 1) * patch_size
            width = int(base_ids[..., 2].max().item() + 1) * patch_size
            # Convert tokens back to image space.
            tokens = img[:, :img_len, :].transpose(1, 2)
            image = F.fold(
                tokens,
                output_size=(height, width),
                kernel_size=patch_size,
                stride=patch_size,
            )
            # Project image patches into hidden space.
            hidden = self.img_in_patch(image).flatten(2).transpose(1, 2)
            img = hidden
            base_image_list = [image] if sz == 1 else [image, image.clone()]
        else:
            img = self.img_in(img)
        img_list = [img] if sz == 1 else [img, img.clone()]
        
        if self.chroma:
            mod_index_length = 344
            distill_timestep = timestep_embedding(timesteps, 16).to(img.device, img.dtype)
            guidance =  torch.tensor([0.]* distill_timestep.shape[0])
            distil_guidance = timestep_embedding(guidance, 16).to(img.device, img.dtype)
            modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
            timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)
            mod_vectors = self.distilled_guidance_layer(input_vec)
        else:
            vec = self.time_in(timestep_embedding(timesteps, 256))
            if self.params.guidance_embed and self.guidance_in is not None:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec +=  self.guidance_in(timestep_embedding(guidance, 256))
            base_vec_list = [vec + self.vector_in(y) for y in y_list]
            vec_list = base_vec_list

        img = None
        txt_list = [self.txt_in(txt) for txt in txt_list ]
        if siglip_embedding is not None:
            txt_list = [torch.cat((siglip_embedding, txt) , dim=1) for txt in txt_list]
            txt_ids_list = [torch.cat((siglip_embedding_ids, txt_id) , dim=1) for txt_id in txt_ids_list]

        pe_list = [self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1)) for txt_ids in txt_ids_list] 

        if self.is_flux2:
            double_vec_list = [ ( self.double_stream_modulation_img(base_vec_list[i]), self.double_stream_modulation_txt(base_vec_list[i]), ) for i in range(sz) ]

        for i, block in enumerate(self.double_blocks):
            if self.chroma:
                vec_list = [( self.get_modulations(mod_vectors, "double_img", idx=i), self.get_modulations(mod_vectors, "double_txt", idx=i))] * sz
            elif self.is_flux2:
                vec_list = double_vec_list
            if callback != None:
                callback(-1, None, False, True)
            if pipeline._interrupt:
                return [None] * sz
            for img, txt, pe, vec in zip(img_list, txt_list, pe_list, vec_list):
                img[...], txt[...] = block(img=img, txt=txt, vec=vec, pe=pe)
                img = txt = pe = vec= None

        img_list = [torch.cat((txt, img), 1) for txt, img in zip(txt_list, img_list)]

        if self.is_flux2:
            single_vec_list = [self.single_stream_modulation(base_vec_list[i])[0] for i in range(sz)]

        for i, block in enumerate(self.single_blocks):
            if self.chroma:
                vec_list= [self.get_modulations(mod_vectors, "single", idx=i)] * sz
            elif self.is_flux2:
                vec_list = single_vec_list
                
            if callback != None:
                callback(-1, None, False, True)
            if pipeline._interrupt:
                return [None] * sz
            for img, pe, vec in zip(img_list, pe_list, vec_list):
                img[...]= block(x=img, vec=vec, pe=pe)
                img = pe = vec = None
        img_list = [img[:, txt.shape[1] : txt.shape[1] + img_len, ...] for img, txt in zip(img_list, txt_list)]

        if self.radiance:
            final_vecs = None
        elif self.chroma:
            final_vecs = [self.get_modulations(mod_vectors, "final")] * sz
        else:
            final_vecs = base_vec_list
        out_list = []
        for i in range(sz):
            hidden_seq = img_list[i]
            if self.radiance:
                base_image = base_image_list[i]
                pred_tokens = apply_radiance_head(
                    module=self,
                    hidden_seq=hidden_seq,
                    base_image=base_image,
                    height=height,
                    width=width,
                )
                base_image_list[i] = base_image = None
                img_list[i] = hidden_seq = None
            else:
                vec = final_vecs[i]
                modulated, pred_tokens = self._apply_final_layer(hidden_seq, vec)
                img_list[i] = hidden_seq = vec = modulated = None
            out_list.append(pred_tokens)
        return out_list


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
