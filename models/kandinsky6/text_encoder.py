# Kandinsky 6 text conditioning (kandinskylab/kandinsky-6 kandinsky/core/components/text_embedder.py, MIT License): last hidden state of
# Qwen2.5-VL-7B-Instruct over a fixed video template, the template tokens cropped, and the pooled CLIP ViT-L/14 text embedding.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os

import torch
from mmgp import offload
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration

from shared.utils.phase_progress import text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache

# Exact upstream template (including its spelling): CROP_START counts its tokens before the user prompt.
PROMPT_TEMPLATE = "\n".join([
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, "
    "some kind of animation, video game or scren content.",
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
])
CROP_START = 129
MAX_LENGTH = 1024
CLIP_MAX_LENGTH = 77


class Kandinsky6TextEncoder:
    def __init__(self, qwen_filename, clip_folder):
        qwen_folder = os.path.dirname(qwen_filename)
        qwen = offload.fast_load_transformers_model(qwen_filename, writable_tensors=False, modelClass=Qwen2_5_VLForConditionalGeneration,
                                                    defaultConfigPath=os.path.join(qwen_folder, "config.json"), default_dtype=torch.bfloat16)
        # Only the language model conditions Kandinsky 6: the vision tower and the LM head are never loaded to the GPU.
        self.model = qwen.model.language_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_folder)
        self.clip = offload.fast_load_transformers_model(os.path.join(clip_folder, "model.safetensors"), writable_tensors=False, modelClass=CLIPTextModel,
                                                         ignore_unused_weights=True, forcedConfigPath=os.path.join(clip_folder, "text_config.json"),
                                                         default_dtype=None).eval()
        self.clip._convertWeightsFloatTo = None  # FP32 checkpoint, run in FP32 as upstream
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_folder)
        self.cache = TextEncoderCache()

    def _encode(self, prompts, device):
        layers = list(self.model.layers) + list(self.clip.text_model.encoder.layers)
        with text_encoding_progress(layers, prompt_count=len(prompts)):
            # Upstream's padded batch, mask and default KV cache: the attention kernels these select give its exact hidden states, which
            # differ by about 2% from an unpadded or cache-less call. Only the valid tokens condition the DiT.
            inputs = self.tokenizer([PROMPT_TEMPLATE.format(prompt) for prompt in prompts], max_length=MAX_LENGTH + CROP_START, truncation=True,
                                    padding="max_length", return_tensors="pt").to(device)
            valid_tokens = int(inputs.attention_mask[0, CROP_START:].sum())
            text_embeds = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state[:, CROP_START:CROP_START + valid_tokens]
            clip_inputs = self.clip_tokenizer(prompts, max_length=CLIP_MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt").to(device)
            pooled_embed = self.clip(**clip_inputs).pooler_output
        # the DiT receives BF16 text tokens and pooled embedding, as upstream
        return [{"text_embeds": text_embeds.to(torch.bfloat16), "pooled_embed": pooled_embed.to(torch.bfloat16)}]

    def encode(self, prompt, device):
        """One prompt, cached on the CPU by prompt."""
        return self.cache.encode(lambda prompts: self._encode(prompts, device), [prompt], device=device, cache_keys=[prompt])[0]
