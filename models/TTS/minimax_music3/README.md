# MiniMax Music 3

This directory vendors and adapts the MiniMax Music 3 component implementations added to Diffusers by
`huggingface/diffusers@2da7040be1a2e5f2fcbc8b985083342a308f5a86` (PR #14456). The component source retains its
Apache-2.0 notices. WanGP replaces the experimental Diffusers modular pipeline with a compact local pipeline, routes
flow and RVQ attention through WanGP's shared attention dispatcher, and delegates all model placement/offloading to
MMGP.

The model weights and associated assets are distributed under the MiniMax-Music3 Community License; users are
responsible for its attribution, commercial-use, safeguard, and acceptable-use terms. License and model-card files are
repository metadata and are not included in WanGP's runtime asset manifest.

WanGP treats the autoregressive Qwen3 stage as the model's `text_encoder`. Its BF16/ConvRot checkpoints and tokenizer
files share the root-level `MiniMaxMusic3-Qwen3` checkpoint folder. The flow transformer remains the sole root
transformer checkpoint. The MiniMax-specific RVQ decoder, condition encoder, and vocoder weights/configs, along with
the scheduler config, are flattened directly into `MiniMax-Music3`. Transformer and Qwen architecture configs are
versioned beside this source rather than downloaded as checkpoint assets.

The integration deliberately uses WanGP's pinned Diffusers and Transformers versions. The published Qwen config's
newer `rope_parameters.rope_theta` schema is translated to the pinned Qwen3 `rope_theta` field during construction so
the released value of 1,000,000 is preserved without upgrading Transformers.

Autoregressive decoding follows WanGP's shared `lm_decoder_engine` selection. `cg` uses fixed-shape Qwen and RVQ KV
caches with CUDA graphs and SDPA. `vllm` adds FlashAttention2 plus measured shape-specific Triton kernels, and is only
selected when both runtimes pass WanGP's availability probe. Accelerated modes use the RVQ ConvRot checkpoint while
MMGP remains responsible for loading, residency, and offloading.
