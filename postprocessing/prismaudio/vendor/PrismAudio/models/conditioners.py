from __future__ import annotations

#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc
from typing import Literal, Optional
import os 
import numpy as np
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from .mmmodules.model.low_level import ConvMLP, MLP
from torch.nn.utils.rnn import pad_sequence

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class Cond_MLP(Conditioner):
    def __init__(self, dim, output_dim, dropout = 0.0):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
        self.dropout = dropout
    def forward(self, x, device: tp.Any = "cuda"):
        x = pad_sequence(x, batch_first=True).to(device)
        # x = torch.stack(x, dim=0).to(device)

        if self.dropout > 0.0:
            if self.training:
                null_embed = torch.zeros_like(x, device=device)
                dropout_mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), self.dropout, device=device)).to(torch.bool)
                x = torch.where(dropout_mask, null_embed, x)
            elif x.shape[0] < 16: # default test batch size=1
                null_embed = torch.zeros_like(x, device=device)
                x = torch.cat([x, null_embed], dim=0)

        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Global_MLP(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)
        x = x.mean(dim=1)
        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Cond_MLP_1(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.SiLU(),
                MLP(output_dim, output_dim * 4),
            )
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Cond_MLP_Global(Conditioner):
    def __init__(self, dim, output_dim, dropout = 0.0):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
        self.global_embedder = nn.Sequential(
                nn.Linear(output_dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
        self.dropout = dropout
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)
        if self.dropout > 0 and self.training:
            null_embed = torch.zeros_like(x, device=device)
            dropout_mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), self.dropout, device=device)).to(torch.bool)
            x = torch.where(dropout_mask, null_embed, x)
        x = self.embedder(x)        # B x 117 x C
        global_x = self.global_embedder(x[:,0,:])
        return [x, torch.ones(x.shape[0], 1).to(device), global_x, torch.ones(global_x.shape[0], 1).to(device)]

class Cond_MLP_Global_1(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.SiLU(),
                MLP(output_dim, output_dim * 4),
            )
        self.global_embedder = nn.Sequential(
                nn.Linear(dim, output_dim),
                MLP(output_dim, output_dim * 4),
            )
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        global_x = self.global_embedder(x.mean(dim=1))
        return [x, torch.ones(x.shape[0], 1).to(device), global_x, torch.ones(global_x.shape[0], 1).to(device)]

class Cond_MLP_Global_2(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
        self.global_embedder = nn.Sequential(
                nn.Linear(output_dim, output_dim, bias=False),
            )
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        global_x = self.global_embedder(x.mean(dim=1))
        return [x, torch.ones(x.shape[0], 1).to(device), global_x, torch.ones(global_x.shape[0], 1).to(device)]

class Sync_MLP(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim, bias=False),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=False)
            )
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, dim)))
        nn.init.constant_(self.sync_pos_emb, 0)
    def forward(self, x, device: tp.Any = "cuda"):
        sync_f = torch.stack(x, dim=0).to(device)
        bs, length, dim = sync_f.shape
        #print(sync_f.shape,flush=True)
        # B * num_segments (24) * 8 * 768
        num_sync_segments = length // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)
        x = self.embedder(sync_f)        # B x 117 x C
        x = x.transpose(1,2)
        x = F.interpolate(x, ((int)(194*sync_f.shape[1]/216), ), mode='linear', align_corners=False)
        x = x.transpose(1,2)
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Cond_ConvMLP(Conditioner):
    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.SiLU(),
                ConvMLP(output_dim, output_dim * 4, kernel_size=1, padding=0),
            )
    def forward(self, x, device: tp.Any = "cuda"):
        x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Video_Global(Conditioner):
    """ Transform the video feat encoder"""

    def __init__(self, dim, output_dim, global_dim=1536):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(nn.Linear(dim, output_dim))
        self.global_proj = nn.Sequential(nn.Linear(output_dim, global_dim))

    def forward(self, x, device: tp.Any = "cuda"):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(x[0], torch.Tensor):
            video_feats = []
            for path in x:
                if '.npy' in path:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
                elif '.pth' in path:
                    data = torch.load(path)
                    video_feats.append(data['metaclip_features'].to(device))
                else:
                    video_feats.append(torch.from_numpy(np.load(path)['feat']).to(device))
            x = torch.stack(video_feats, dim=0).to(device)
        else:
            # Revise the shape here:
            x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        global_x = self.global_proj(x.mean(dim=1))
        return [x, torch.ones(x.shape[0], 1).to(device), global_x, torch.ones(global_x.shape[0], 1).to(device)]

class Video_Sync(Conditioner):
    """ Transform the video feat encoder"""

    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(nn.Linear(dim, output_dim))

    def forward(self, x, device: tp.Any = "cuda"):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(x[0], torch.Tensor):
            video_feats = []
            for path in x:
                if '.npy' in path:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
                elif '.pth' in path:
                    video_feats.append(torch.load(path)['sync_features'].to(device))
                else:
                    video_feats.append(torch.from_numpy(np.load(path)['feat']).to(device))
            x = torch.stack(video_feats, dim=0).to(device)
        else:
            # Revise the shape here:
            x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class Text_Linear(Conditioner):
    """ Transform the video feat encoder"""

    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)
        self.embedder = nn.Sequential(nn.Linear(dim, output_dim))

    def forward(self, x, device: tp.Any = "cuda"):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(x[0], torch.Tensor):
            video_feats = []
            for path in x:
                if '.npy' in path:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
                elif '.pth' in path:
                    video_feats.append(torch.load(path)['metaclip_text_features'].to(device))
                else:
                    video_feats.append(torch.from_numpy(np.load(path)['feat']).to(device))
            x = torch.stack(video_feats, dim=0).to(device)
        else:
            # Revise the shape here:
            x = torch.stack(x, dim=0).to(device)

        x = self.embedder(x)        # B x 117 x C
        return [x, torch.ones(x.shape[0], 1).to(device)]

class mm_unchang(Conditioner):
    """ Transform the video feat encoder"""

    def __init__(self, dim, output_dim):
        super().__init__(dim, output_dim)

    def forward(self, x, device: tp.Any = "cuda"):
        # import ipdb
        # ipdb.set_trace()
        if not isinstance(x[0], torch.Tensor):
            video_feats = []
            for path in x:
                if '.npy' in path:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
                elif '.pth' in path:
                    video_feats.append(torch.load(path)['metaclip_features'].to(device))
                else:
                    video_feats.append(torch.from_numpy(np.load(path)['feat']).to(device))
            x = torch.stack(video_feats, dim=0).to(device)
        else:
            # Revise the shape here:
            x = torch.stack(x, dim=0).to(device)
        return [x]

class CLIPConditioner(Conditioner):

    CLIP_MODELS = ["metaclip-base", "metaclip-b16", "metaclip-large", "metaclip-huge"]
    
    CLIP_MODEL_DIMS = {
        "metaclip-base": 512,
        "metaclip-b16": 512,
        "metaclip-large": 768,
        "metaclip-huge": 1024,
    }

    def __init__(
            self,
            dim: int,
            output_dim: int,
            clip_model_name: str = "metaclip-huge",
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown CLIP model name: {clip_model_name}"
        super().__init__(self.CLIP_MODEL_DIMS[clip_model_name], output_dim, project_out=project_out)
        
        self.enable_grad = enable_grad
        model = AutoModel.from_pretrained(f"useful_ckpts/{clip_model_name}").train(enable_grad).requires_grad_(enable_grad).to(torch.float16)

        
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, images: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)
        # import ipdb
        # ipdb.set_trace()

        self.model.eval()
        if not isinstance(images[0], torch.Tensor):
            video_feats = []
            for path in images:
                if '.npy' in path:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
                else:
                    video_feats.append(torch.from_numpy(np.load(path)).to(device))
            images = torch.stack(video_feats, dim=0).to(device)
        else:
            images = torch.stack(images, dim=0).to(device)    
        bsz, t, c, h, w = images.shape
        # 使用 rearrange 进行维度合并
        images = rearrange(images, 'b t c h w -> (b t) c h w')
        with torch.set_grad_enabled(self.enable_grad):
            image_features = self.model.get_image_features(images)
        image_features = rearrange(image_features, '(b t) d -> b t d', b=bsz, t=t)
        image_features = self.proj_out(image_features)


        return [image_features, torch.ones(image_features.shape[0], 1).to(device)]

class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        from .adp import NumberEmbedder
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:

            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                self.model = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                self.model.model.audio_branch.requires_grad_(False)
                self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "t5-v1_1-xl", "google/t5-v1_1-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 77,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(os.path.join('useful_ckpts', t5_model_name))
                model = T5EncoderModel.from_pretrained(os.path.join('useful_ckpts', t5_model_name)).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    
            
        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model

class CLIPTextConditioner(Conditioner):
    def __init__(
            self,
            output_dim: int,
            max_length: str = 77,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        super().__init__(1024, output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer
        import open_clip
        from open_clip import create_model_from_pretrained

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384',cache_dir='useful_ckpts/DFN5B-CLIP-ViT-H-14-384',
                                                           return_transform=False).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
                model = patch_clip(model)
                self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts
        ).to(device)

        # input_ids = encoded["input_ids"].to(device)
        # attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model.encode_text(
                encoded
            )
            
        embeddings = self.proj_out(embeddings.float())

        # embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, torch.ones(embeddings.shape[0], 1).to(device)

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_get_text_features(self, input_ids=None, attention_mask=None, position_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = text_outputs[0]
        # pooled_output = text_outputs[1]
        # text_features = self.text_projection(pooled_output)

        return last_hidden_state

    clip_model.get_text_features = new_get_text_features.__get__(clip_model)
    return clip_model

class MetaCLIPTextConditioner(Conditioner):
    def __init__(
            self,
            output_dim: int,
            max_length: str = 77,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        super().__init__(1024, output_dim, project_out=project_out)
        
        from transformers import AutoModel
        from transformers import AutoProcessor

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model = AutoModel.from_pretrained("useful_ckpts/metaclip-huge")
                self.model = patch_clip(self.model)
                self.clip_processor = AutoProcessor.from_pretrained("useful_ckpts/metaclip-huge")
            finally:
                logging.disable(previous_level)


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)
        encoded = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

        # input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model.get_text_features(
                **encoded
            )
            
        embeddings = self.proj_out(embeddings.float())

        # embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, torch.ones(embeddings.shape[0],1).to(device)
    
class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        from ..inference.utils import set_audio_channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                    
                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)
            
            cond_output = conditioner(conditioner_inputs, device)
            if len(cond_output) == 1:
                output[key] = cond_output[0]
            elif len(cond_output) == 2:
                output[key] = cond_output
            elif len(cond_output) == 4:
                output[key] = cond_output[:2]
                output[f'{key}_g'] = cond_output[2:]

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])
        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clip_text":
            conditioners[id] = CLIPTextConditioner(**conditioner_config)
        elif conditioner_type == "metaclip_text":
            conditioners[id] = MetaCLIPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "cond_mlp":
            conditioners[id] = Cond_MLP(**conditioner_config)
        elif conditioner_type == "global_mlp":
            conditioners[id] = Global_MLP(**conditioner_config)
        elif conditioner_type == "sync_mlp":
            conditioners[id] = Sync_MLP(**conditioner_config)
        elif conditioner_type == "cond_mlp_1":
            conditioners[id] = Cond_MLP_1(**conditioner_config)
        elif conditioner_type == "cond_convmlp":
            conditioners[id] = Cond_ConvMLP(**conditioner_config)
        elif conditioner_type == "cond_mlp_global":
            conditioners[id] = Cond_MLP_Global(**conditioner_config)
        elif conditioner_type == "cond_mlp_global_1":
            conditioners[id] = Cond_MLP_Global_1(**conditioner_config)
        elif conditioner_type == "cond_mlp_global_2":
            conditioners[id] = Cond_MLP_Global_2(**conditioner_config)
        elif conditioner_type == "video_linear":
            conditioners[id] = Video_Linear(**conditioner_config)
        elif conditioner_type == "video_global":
            conditioners[id] = Video_Global(**conditioner_config)
        elif conditioner_type == "video_sync":
            conditioners[id] = Video_Sync(**conditioner_config)
        elif conditioner_type == "text_linear":
            conditioners[id] = Text_Linear(**conditioner_config)
        elif conditioner_type == "video_clip":
            conditioners[id] = CLIPConditioner(**conditioner_config)
        elif conditioner_type == "video_hiera":
            conditioners[id] = VideoHieraConditioner(**conditioner_config)
        elif conditioner_type == "meta_query":
            from .meta_queries.model import MLLMInContext
            conditioners[id] = MLLMInContext(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            from .factory import create_pretransform_from_config
            from .utils import load_ckpt_state_dict
            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
        elif conditioner_type == "mm_unchang":
            conditioners[id] = mm_unchang(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)
