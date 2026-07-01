#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import typing as tp
from enum import Enum
from math import pi
from typing import List, Union
from torch import Tensor, nn
from einops import rearrange
from .blocks import ExpoFourierFeatures
from .utils import enable_torch_compile
import os
from mmgp import offload

class PaddingMode(str, Enum):
    """Enum for handling padding in text conditioner embeddings."""
    NONE = "none"       # No padding handling (raw embeddings with pad token)
    ZERO = "zero"       # Zero out padding positions (default)
    LEARNED = "learned" # Use learned padding embedding

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            padding_mode: str = "zero"
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        # Learned padding embedding (only created if needed)
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply padding handling based on padding_mode.

        Args:
            embeddings: [batch, seq_len, dim] - the embeddings to process
            attention_mask: [batch, seq_len] bool/int, True/1 = valid token

        Returns:
            embeddings with padding handled according to mode
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings)
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, std: float = 16.0):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim) * std)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
        fourier_features_type: tp.Literal["learned", "expo"] = "learned"
    ):
        super().__init__()
        self.features = features
        if fourier_features_type == "expo":
            self.embedding = nn.Sequential(ExpoFourierFeatures(dim=dim), nn.Linear(in_features=dim, out_features=features))
        else:
            self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore


class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1,
                fourier_features_type : tp.Literal["learned", "expo"] = "learned"
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

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

class T5GemmaConditioner(Conditioner):

    T5GEMMA_MODELS = ["google/t5gemma-b-b-ul2"]

    T5GEMMA_MODEL_DIMS = {
        "google/t5gemma-b-b-ul2": 768,
    }

    def __init__(
            self,
            output_dim: int,
            model_name: str = "google/t5gemma-b-b-ul2",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero",
            model_path: str = None,
            repo_id: str = None,
            subfolder: str = None,
            weights_path: str = None,
            tokenizer_path: str = None,
            dtype: torch.dtype = torch.bfloat16,
    ):
        assert model_name in self.T5GEMMA_MODELS, f"Unknown T5 model name: {model_name}"
        super().__init__(self.T5GEMMA_MODEL_DIMS[model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        load_from = tokenizer_path or model_path or repo_id or model_name

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.dtype = dtype

        # Set environment variables to disable progress bars BEFORE importing transformers
        # This is the most reliable way to suppress HuggingFace progress bars
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import AutoTokenizer
                from transformers.models.t5gemma import T5GemmaEncoderModel
                logging.info(f"Loading T5Gemma tokenizer and model from: {load_from}")
                if tokenizer_path is not None:
                    subfolder = None
                hf_kwargs = {"subfolder": subfolder} if subfolder else {}
                self.tokenizer = AutoTokenizer.from_pretrained(load_from, local_files_only=weights_path is not None, **hf_kwargs)
                if weights_path is None:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(load_from, **hf_kwargs)
                    config.is_encoder_decoder = False
                    model = T5GemmaEncoderModel.from_pretrained(load_from, config=config, **hf_kwargs).train(enable_grad).requires_grad_(enable_grad)
                else:
                    model_prefix = "encoder" if "quanto" in os.path.basename(weights_path).lower() else "model.encoder"
                    model = offload.fast_load_transformers_model(
                        weights_path,
                        writable_tensors=False,
                        modelClass=T5GemmaEncoderModel,
                        defaultConfigPath=os.path.join(load_from, "config.json"),
                        modelPrefix=model_prefix,
                        configKwargs={"is_encoder_decoder": False},
                        default_dtype=dtype,
                    ).train(enable_grad).requires_grad_(enable_grad)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self._device_initialized = False

    def forward(self, inputs: tp.Union[tp.List[str], tp.List[tp.Dict[str, torch.Tensor]]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        if not self._device_initialized:
            self.model.eval()
            self._device_initialized = True

        # Handle pre-tokenized inputs (dicts with input_ids/attention_mask from DataLoader workers)
        # or raw strings (from demo generation / inference)
        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

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

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output
    

