Qwen Image 2.1 model code is adapted from Qwen/Hugging Face Diffusers
(`transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`, and prompt encoding
from `pipeline_qwenimage21.py`). Copyright 2026 Qwen and Hugging Face teams.

The local Qwen3-VL layers, position calculations, and vision utilities are adapted
from Hugging Face Transformers **v5.17.0**. Copyright 2025 Qwen and Hugging Face.
Image preprocessing follows the released Qwen Image 2.1 processor configuration
and Qwen3-VL spatial/temporal patch ordering. Only image prefill is used here;
autoregressive generation, training, distributed execution, and hub kernel
integrations are not included.

These sources are licensed under Apache License 2.0; see LICENSE-APACHE-2.0. The model weights retain the Qwen Research License in LICENSE.

Weights: https://huggingface.co/Qwen/Qwen-Image-2.1
Pinned weight revision: `b3179ad355be050328e483a9dfdd9e60cd62adfa`.
Verified original and Quanto INT8 checkpoints and required assets were uploaded to https://huggingface.co/DeepBeepMeep/Qwen_image_2 at commit `a629f8f95c11cc18b8dbd11137452f14a7c9aa9f`. All 22 remote file sizes and every checkpoint LFS SHA-256 match the local upload manifest.

Enhancer profiles adapt the official Qwen Image 2.1 prompt_rewrite guidance to WanGP plain-text output and user-controlled dimensions.

Diffusers transformer source revision: `6256aa7666cedd47443adc8f82da9a10e110b09c`. Official prompt guidance revision: `7307809d2c9d582be700a1b9b04d393fc9cdf865`.
RGB preview coefficients are the published QwenImage21 values from ComfyUI `comfy/latent_formats.py`.
