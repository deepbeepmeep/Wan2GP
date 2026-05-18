# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

# Set env vars for local T5 loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from .sa_audio_module import create_model_from_config

from inference.utils import print_rank_0


def _remap_legacy_weight_norm_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            key = key[:-9] + ".parametrizations.weight.original0"
        elif key.endswith(".weight_v"):
            key = key[:-9] + ".parametrizations.weight.original1"
        remapped[key] = value
    return remapped


class SAAudioFeatureExtractor:
    """Stable Audio Feature Extractor that loads model once and reuses it."""

    def __init__(self, device, model_path):
        """Initialize the extractor with model loading."""
        self.device = device
        self.vae_model, self.sample_rate = self._get_vae_only(model_path)
        # self.vae_model.to(self.device).to(torch.bfloat16)
        self.resampler = None  # Will be initialized when needed

    def _get_vae_only(self, model_path):
        """Load VAE only, skip T5 and diffusion model."""
        if isinstance(model_path, str) and Path(model_path).is_dir():
            try:
                # Read full config
                model_config_path = os.path.join(model_path, "model_config.json")
                with open(model_config_path) as f:
                    full_config = json.load(f)

                vae_config = full_config["model"]["pretransform"]["config"]
                sample_rate = full_config["sample_rate"]

                # Rebuild config structure expected by create_autoencoder_from_config
                autoencoder_config = {
                    "model_type": "autoencoder",
                    "sample_rate": sample_rate,  # sample_rate is required
                    "model": vae_config,  # create_autoencoder_from_config expects key "model"
                }

                vae_model = create_model_from_config(autoencoder_config)
                # Load weights
                weights_path = Path(model_path) / "model.safetensors"

                if not weights_path.exists():
                    raise FileNotFoundError(f"Weight file does not exist: {weights_path}")

                # Load full state dict
                full_state_dict = load_file(weights_path, device=str(self.device))

                # Filter VAE-related weights (prefix: pretransform.model)
                vae_state_dict = {}
                for key, value in full_state_dict.items():
                    if key.startswith("pretransform.model."):
                        vae_key = key[len("pretransform.model.") :]
                        vae_state_dict[vae_key] = value
                vae_state_dict = _remap_legacy_weight_norm_state_dict(vae_state_dict)

                # Check expected model keys
                model_keys = set(vae_model.state_dict().keys())
                vae_keys = set(vae_state_dict.keys())

                missing_keys = model_keys - vae_keys
                extra_keys = vae_keys - model_keys

                if missing_keys:
                    print_rank_0(f"Missing keys ({len(missing_keys)}):")
                    for key in list(missing_keys)[:5]:
                        print_rank_0(f"  - {key}")

                if extra_keys:
                    print_rank_0(f"Unexpected keys ({len(extra_keys)}):")
                    for key in list(extra_keys)[:5]:
                        print_rank_0(f"  + {key}")

                # Load VAE weights
                vae_model.load_state_dict(vae_state_dict)
                vae_model.to(self.device)

                return vae_model, sample_rate

            except Exception as e:
                print_rank_0(f"audio model loading failed: {e}")
                raise RuntimeError(
                    "Failed to load VAE-only Stable Audio model from local path"
                ) from e
        else:
            print_rank_0("Non-local path is not supported in audio model loading")

    def decode(self, latents):
        with torch.no_grad():
            waveform_out = self.vae_model.decode(latents)
        return waveform_out

    def encode(self, waveform):
        with torch.no_grad():
            latents = self.vae_model.encode(waveform)
        return latents
