"""Image-only Qwen3-VL preprocessing, following the release processor config."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature


class Qwen21Processor:
    def __init__(self, folder, preprocessor_path=None, tokenizer_path=None, tokenizer_config_path=None):
        # Newer tokenizer exports store this as a list; older Transformers
        # expects a named-token dictionary. The tokens already exist in
        # tokenizer.json, so no vocabulary additions are needed here.
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(folder, extra_special_tokens={}, local_files_only=True)
        else:
            from transformers import Qwen2TokenizerFast
            config = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
            # Token definitions are embedded in tokenizer.json. Explicit paths
            # avoid borrowing another consumer's same-folder tokenizer config.
            for key in ("tokenizer_class", "processor_class", "added_tokens_decoder"):
                config.pop(key, None)
            config["extra_special_tokens"] = {}
            self.tokenizer = Qwen2TokenizerFast(tokenizer_file=str(tokenizer_path), **config)
        self.config = json.loads(Path(preprocessor_path or Path(folder) / "preprocessor_config.json").read_text(encoding="utf-8"))

    def __call__(self, text, images=None, padding=True, padding_side="left", return_tensors="pt"):
        prompts = list(text)
        patches, grids = [], []
        patch, merge, temporal = (self.config[key] for key in ("patch_size", "merge_size", "temporal_patch_size"))
        index = 0
        for row, prompt in enumerate(prompts):
            while "<|image_pad|>" in prompt:
                image = images[index].convert("RGB")
                width, height = image.size
                if width % (patch * merge) or height % (patch * merge):
                    raise ValueError("Reference image dimensions must be multiples of 32.")
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
                pixels *= self.config["rescale_factor"]
                pixels = (pixels - pixels.new_tensor(self.config["image_mean"])[:, None, None]) / pixels.new_tensor(self.config["image_std"])[:, None, None]
                gh, gw = height // patch, width // patch
                pixels = pixels.unsqueeze(0).expand(temporal, -1, -1, -1)
                pixels = pixels.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
                pixels = pixels.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, -1)
                patches.append(pixels)
                grids.append([1, gh, gw])
                prompt = prompt.replace("<|image_pad|>", "<|placeholder|>" * (gh * gw // (merge * merge)), 1)
                index += 1
            prompts[row] = prompt.replace("<|placeholder|>", "<|image_pad|>")
        encoded = dict(self.tokenizer(prompts, padding=padding, padding_side=padding_side, return_tensors=return_tensors))
        if patches:
            encoded.update(pixel_values=torch.cat(patches), image_grid_thw=torch.tensor(grids, dtype=torch.long, device="cpu"))
        return BatchFeature(encoded)


def load_processor(folder, preprocessor_path=None, **kwargs):
    return Qwen21Processor(folder, preprocessor_path, **kwargs)
