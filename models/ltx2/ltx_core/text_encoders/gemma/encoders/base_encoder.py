import functools
from typing import NamedTuple
from pathlib import Path

import torch
from einops import rearrange
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor, Gemma3ForCausalLM

from mmgp import offload
from ....loader.module_ops import ModuleOps
from ..embeddings_connector import Embeddings1DConnector
from ..feature_extractor import GemmaFeaturesExtractorProjLinear
from ..tokenizer import LTXVGemmaTokenizer
from shared.utils import files_locator as fl
from .....ltx2_handler import  _GEMMA_FOLDER, family_handler

import os

class GemmaTextEncoderModelBase(torch.nn.Module):
    """
    Gemma Text Encoder Model.
    This base class combines the tokenizer, Gemma model and feature extractor to provide a preprocessing
    for implementation classes for multimodal pipelines. It processes input text through tokenization,
    obtains hidden states from the base language model, applies a linear feature extractor.
    Args:
        tokenizer (LTXVGemmaTokenizer): The tokenizer used for text preprocessing.
        model (Gemma3ForConditionalGeneration): The base Gemma LLM.
        feature_extractor_linear (GemmaFeaturesExtractorProjLinear): Linear projection for hidden state aggregation.
        dtype (torch.dtype, optional): The data type for model parameters (default: torch.bfloat16).
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear | None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        img_processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._gemma_root = None
        self.tokenizer = tokenizer
        self.model = model
        self.processor = img_processor
        if feature_extractor_linear is not None:
            self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)
        else:
            self.feature_extractor_linear = None

    def encode_raw(self, text: str, padding_side: str = "left") -> "RawTextEmbeddings":
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return RawTextEmbeddings(outputs.hidden_states, attention_mask, padding_side)

    def _run_feature_extractor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor:
        if self.feature_extractor_linear is None:
            raise ValueError("feature_extractor_linear is not available for this text encoder")
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )

        return self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features_dtype))

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max

    def _preprocess_text(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.
        Args:
            text (str): Input string to encode.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        raw = self.encode_raw(text, padding_side=padding_side)
        attention_mask = raw.attention_mask
        projected = self._run_feature_extractor(
            hidden_states=raw.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    def _init_image_processor(self) -> None:
        img_processor = AutoImageProcessor.from_pretrained(self._gemma_root, local_files_only=True)
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded, cannot load image processor")
        self.processor = Gemma3Processor(image_processor=img_processor, tokenizer=self.tokenizer.tokenizer)

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> str:
        if self.processor is None:
            self._init_image_processor()
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for T2V generation."""

        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    def forward(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is not implemented for the base class")


class RawTextEmbeddings(NamedTuple):
    hidden_states: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    padding_side: str


def _apply_feature_extractor(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    padding_side: str,
    feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
) -> torch.Tensor:
    encoded_text_features = torch.stack(hidden_states, dim=-1)
    encoded_text_features_dtype = encoded_text_features.dtype
    sequence_lengths = attention_mask.sum(dim=-1)
    normed = _norm_and_concat_padded_batch(
        encoded_text_features,
        sequence_lengths,
        padding_side=padding_side,
    )
    return feature_extractor_linear(normed.to(encoded_text_features_dtype))


def _apply_connectors(
    encoded_input: torch.Tensor,
    attention_mask: torch.Tensor,
    embeddings_connector: Embeddings1DConnector,
    audio_embeddings_connector: Embeddings1DConnector,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    connector_attention_mask = (attention_mask - 1).to(encoded_input.dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(encoded_input.dtype).max
    encoded, encoded_connector_attention_mask = embeddings_connector(
        encoded_input,
        connector_attention_mask,
    )
    attention_mask_out = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
    attention_mask_out = attention_mask_out.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * attention_mask_out
    encoded_for_audio, _ = audio_embeddings_connector(encoded_input, connector_attention_mask)
    return encoded, encoded_for_audio, attention_mask_out.squeeze(-1)

def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(t, device=device)[None, :]  # [1, T]

    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + eps)

    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(b, t, -1)  # [B, T, D * L]

    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _find_matching_dir(root_path: str, pattern: str) -> str:
    """
    Recursively search for files matching a glob pattern and return the parent directory of the first match.
    """

    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


def _find_merged_gemma_file(gemma_root: str) -> str | None:
    root_path = Path(gemma_root)
    if root_path.is_file() and root_path.suffix == ".safetensors":
        return str(root_path)

    base_name = root_path.name
    candidates = [
        root_path / "model.safetensors",
        root_path / f"{base_name}.safetensors",
        root_path / f"{base_name}_bf16.safetensors",
        Path("c:/temp") / f"{base_name}.safetensors",
        Path("c:/temp") / f"{base_name}_bf16.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _preprocess_gemma_state_dict(sd, qm, twm):
    if len(sd) == 0:
        return sd
    sd.pop("spiece_model", None)
    from mmgp.offload import map_state_dict
    rules = {"model.language_model": "model",  "model.vision_tower": None, "model.multi_modal_projector": None, "multi_modal_projector": None }
    sd, qm, twm = map_state_dict([sd, qm, twm], rules=rules)

    if twm is None or len(twm)==0:
        twm = {"model.embed_tokens.weight": ["lm_head.weight"]}

    return sd, qm, twm


def build_gemma_text_encoder(
    gemma_root: str, default_dtype: torch.dtype = torch.bfloat16
) -> GemmaTextEncoderModelBase:

    gemma_path = gemma_root
    if not gemma_path or not os.path.isfile(gemma_path):
        raise FileNotFoundError(f"Gemma checkpoint not found: {gemma_root}")
    gemma_dir = os.path.dirname(gemma_path)
    tokenizer_path = fl.locate_folder(os.path.join(_GEMMA_FOLDER))
    config_path = fl.locate_file(os.path.join(_GEMMA_FOLDER, "config_light.json"))
    from accelerate import init_empty_weights
    with init_empty_weights():
        text_encoder = GemmaTextEncoderModelBase(feature_extractor_linear=None, tokenizer=None, model=None, dtype=default_dtype)
    text_encoder.model = offload.fast_load_transformers_model(
        gemma_path,
        modelClass=Gemma3ForCausalLM, #Gemma3ForConditionalGeneration,
        defaultConfigPath=config_path,
        writable_tensors=False,
        preprocess_sd=_preprocess_gemma_state_dict,
        forcedConfigPath= config_path,
        default_dtype=default_dtype,
    )
    text_encoder.tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
    text_encoder._gemma_root = gemma_dir

    return text_encoder


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    gemma_path = gemma_root
    gemma_root = os.path.dirname(gemma_root)
    tokenizer_path =  fl.locate_folder(os.path.join(_GEMMA_FOLDER)) #, "tokenizer.model"

    def load_gemma(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        config_path = fl.locate_file(os.path.join(_GEMMA_FOLDER, "config.json"))
        module.model = offload.fast_load_transformers_model(
            gemma_path,
            modelClass=Gemma3ForConditionalGeneration,
            defaultConfigPath=config_path,
            writable_tensors=False,
            preprocess_sd=_preprocess_gemma_state_dict,
        )
        module._gemma_root = module._gemma_root or gemma_root
        return module

    def load_tokenizer(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
        module._gemma_root = module._gemma_root or gemma_root
        return module

    gemma_load_ops = ModuleOps(
        "GemmaLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.model is None,
        mutator=load_gemma,
    )
    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    return (gemma_load_ops, tokenizer_load_ops)


def encode_text(text_encoder: GemmaTextEncoderModelBase, prompts: list[str]) -> list[RawTextEmbeddings]:
    """
    Encode prompts with the Gemma text encoder, returning raw embeddings for later post-processing.
    """
    result = []
    for prompt in prompts:
        result.append(text_encoder.encode_raw(prompt))
    return result


def postprocess_text_embeddings(
    embeddings: list[RawTextEmbeddings],
    feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
    embeddings_connector: Embeddings1DConnector,
    audio_embeddings_connector: Embeddings1DConnector,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply projection and connector post-processing to raw embeddings.
    """
    projected = []
    for item in embeddings:
        attention_mask = item.attention_mask.to("cuda")
        encoded_input = _apply_feature_extractor(
            item.hidden_states,
            attention_mask,
            item.padding_side,
            feature_extractor_linear,
        )
        projected.append((encoded_input, attention_mask))
    results = []
    for encoded_input, attention_mask in projected:
        video_ctx, audio_ctx, _ = _apply_connectors(
            encoded_input,
            attention_mask,
            embeddings_connector,
            audio_embeddings_connector,
        )
        results.append((video_ctx, audio_ctx))
    return results


def resolve_text_connectors(
    text_encoder: GemmaTextEncoderModelBase,
    text_connectors: dict | None,
) -> tuple[GemmaFeaturesExtractorProjLinear, Embeddings1DConnector, Embeddings1DConnector]:
    if text_connectors is not None:
        feature_extractor = text_connectors.get("feature_extractor_linear")
        video_connector = text_connectors.get("embeddings_connector")
        audio_connector = text_connectors.get("audio_embeddings_connector")
    else:
        feature_extractor = getattr(text_encoder, "feature_extractor_linear", None)
        video_connector = getattr(text_encoder, "embeddings_connector", None)
        audio_connector = getattr(text_encoder, "audio_embeddings_connector", None)

    missing = []
    if feature_extractor is None:
        missing.append("feature_extractor_linear")
    if video_connector is None:
        missing.append("embeddings_connector")
    if audio_connector is None:
        missing.append("audio_embeddings_connector")
    if missing:
        raise ValueError(f"Missing text connector modules: {', '.join(missing)}")
    return feature_extractor, video_connector, audio_connector


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility.
    Flash Attention within SDPA requires sequence lengths aligned to 8 bytes.
    This pads input_ids, attention_mask, and token_type_ids (if present) to prevent
    'p.attn_bias_ptr is not correctly aligned' errors.
    """
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)

        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)

        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs
