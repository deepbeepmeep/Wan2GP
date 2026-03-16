import os
from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer, Qwen3ForCausalLM

from mmgp import offload
from shared.utils import files_locator as fl

OUTPUT_LAYERS = [9, 18, 27]
MAX_LENGTH = 512


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str | None = None,
        tokenizer_path: str | None = None,
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()
        file_path = model_spec
        default_config = os.path.join(os.path.dirname(file_path), "config.json")
        self.model = offload.fast_load_transformers_model(
            file_path,
            writable_tensors=False,
            modelClass=Qwen3ForCausalLM,
            defaultConfigPath=default_config,
        )

        tokenizer_root = tokenizer_path or os.path.dirname(file_path)
        tokenizer_subdir = os.path.join(tokenizer_root, "tokenizer")
        if os.path.isdir(tokenizer_subdir):
            tokenizer_root = tokenizer_subdir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, trust_remote_code=True)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: List[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")
