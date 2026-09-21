# Qwen Image 2.1 in WanGP

Unified 7B text-to-image and image editing model, registered as `qwen_image_21_7B` in the Qwen family. The original 20B architectures remain separate.

Supports up to ten ordered image references, RGBA output, image control, masked denoising, LanPaint, outpainting, sequential-within-layer CFG, negative attention guidance and LoRAs. Image dimensions are multiples of 32. Use 40 steps and CFG 4 as a starting point. NAG implementation is retained, but its UI controls are temporarily hidden. Standard CFG and negative prompts remain available.

## Prompt enhancement

Two system prompts live in `enhancer.py` and use the existing enhancer selector. Enhancement stays off by default.

* **New image (`T`)**: present-tense description of the finished image, with composition, materials and lighting.
* **Edit with references (`TI`)**: actionable operation, targeted attributes and a concise preservation clause. Multiple references use `<image1>`, `<image2>`, etc. in selected order.
* **Edit text only (`T1`)**: the same editing instruction profile when image inspection is not requested.
* **New composition from references (`TI1`)**: editing profile with explicit identity sources and a new scene.

Derived from the official [generation](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt) and [editing](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt) guidance. WanGP consumes plain prompt text; upstream JSON aspect-ratio fields are intentionally omitted. User language and exact visible text are preserved rather than automatically translated. Do not alter the encoder chat template to customize the enhancer.

## Checkpoints and source

Transformer and VAE defaults use `DeepBeepMeep/Qwen_image_2`; shared encoder weights use `DeepBeepMeep/Ideogram4/Qwen3-VL-8B-Instruct`. Transformer files are at the checkpoint root, the encoder and tokenizer are in `Qwen3-VL-8B-Instruct/`, and the VAE/configs are in `qwen_image_21/`. The shared encoder omits the unused language-generation head. Both shared encoder-only and original conditional-generation key layouts and RoPE configs are accepted. All encoder support assets are hosted in that same Ideogram4 folder. The distinct older export uses config_legacy.json, tokenizer_legacy.json and tokenizer_config_legacy.json; generic files retain the Ideogram4 export. Explicit paths preserve the intended tokenizer settings even across multiple local checkpoint roots.

Transformer and encoder support BF16 and INT8 ConvRot weights. The released VAE checkpoint remains native FP32. Following the explicit request to support 16-bit VAE execution, the shared 16-bit setting now selects validated BF16 through MMGP; the 32-bit setting preserves FP32. FP16 is not enabled because real-image decoding produced nonfinite values. Attention uses SDPA through the shared helper; normalization reductions and final byte quantization retain FP32 arithmetic. Experimental standard Quanto VAE conversion replaces the custom five-dimensional convolution wrappers and fails decoding; it is not advertised or uploaded.

The model/encoder/processor sources are local, adapted from Diffusers and Transformers v5.17.0, and tested with installed Transformers 4.54.0 and Diffusers 0.36.0. No package downgrade or upgrade is needed for the vendored architecture. VAE decoding uses 1024px tiles for the 16GB+ preset, 512px for 8GB+, and 256px for 6GB+, all with 25% overlap. Auto selects these at GPU capacity thresholds of 16000 and 8000 MiB. Images fitting within a selected tile decode whole; batch items decode sequentially. Encoding retains the established smaller working tiles because the larger-size benchmark covered decoding only. Image-only encoding/decoding does not retain unused temporal feature caches. Completed tiles go directly to one CPU UINT8 output buffer; only float overlap strips survive for blending. A smooth overlap window suppresses tile-edge padding artifacts. MMGP owns all weight residency and quantized compute. All checkpoint loads use read-only mmap.

## LoRA interchange

Use the `qwen21` LoRA folder. Only adapters trained for this architecture are compatible. Shared `shared/utils/lora_mapping.py` converts Diffusers/PEFT, WanGP and Kohya names in either direction without changing tensor values. It also serves Qwen 20B, including split aliases for fused QKV modules. It converts names, not incompatible architecture shapes. Step-varying multipliers invalidate cached conditioning. A standalone converter is available as `python -m tools.convert_lora_keys INPUT OUTPUT --base-weights ORIGINAL_CHECKPOINT --target kohya` (also `wangp` or `diffusers`).

## Validation

See `VALIDATION.md`. Reproducible utilities: `tools/prepare_qwen_image_21.py`, `tools/validate_qwen21_runtime.py`, `tools/validate_qwen21_features.py`, and `tools/probe_qwen21_vae_int8.py`, and `tools/validate_qwen21_vae.py`. Run tools as modules from the repository root, e.g. `python -m tools.validate_qwen21_runtime CHECKPOINTS OUTPUT --size 1024`.


## Deepy discovery and templates

The handler declares only `text rendering` (aliases: text writing, lettering), with moderate quality best suited to short text. Dense infographics are not recommended. Full and compact help disclose spelling/layout limitations. Resolution choices use the same `<=4096p` category as SenseNova. Outpainting is available in both image generation/editing and inpainting modes when a control image or main reference is selected.

`Qwen Image 2.1 7B` templates are available for `gen_image` and `edit_image` (the established editing tool name). Both use 40 steps, CFG 4, 1024-square output and enhancement off. Editing enables ordered reference inputs with `KI`, matching the existing Qwen Edit workflow. These are new templates with current settings version 2.79; no migration of existing user selections is required. Neither template loads an incompatible older Lightning adapter.


## Outpainting and transparency

Outpainting uses 32-pixel alignment for source placement and masks. New margins receive the full noise schedule even with reduced denoising strength for an inner edit.

Outpainting always uses **Red Canvas** in the UI. The developer-only `OUTPAINTING_METHOD` variable in `pipeline.py` retains the alternative implementation for experiments; saved custom settings cannot select it.

- **Red Canvas** (active method): fill new margins with opaque red and give the full canvas to both the vision/text encoder and VAE. Append `Remove the red paddings on the sides and show what's behind them.` to the positive prompt. For ordinary outpainting, the model edits the full canvas without hard latent reinjection between the source and margins. This avoids a processing discontinuity but allows changes to the original content. When combined with inpainting, only the painted interior mask receives the requested denoising strength: those pixels stay on their source-noise trajectory until the selected start step. Outpainting margins remain red conditioning and run all denoising steps, without mask reinjection. Interior pixels are not painted red. This hybrid uses the strength schedule rather than LanPaint; pixels outside the interior edit remain instruction-guided, not pixel-locked. No overlap blending or pixel restoration is applied. This replaces the failed Edge Padding experiment; obsolete `outpainting_borders` settings are removed when loading settings.
- **Overlap Blend (previous method)** (`OUTPAINTING_METHOD = "Overlap Blend"`, Python only): both encoders receive only the cropped source, excluding the added canvas. An adaptive 128–256-pixel overlap inside expanded source edges is regenerated (reduced for small sources to retain a core). After decoding, original pixels are restored with a smooth transition through that overlap; unmasked core pixels are preserved exactly. Explicit edit masks remain editable. RGBA transitions use premultiplied alpha, with CPU blending in bounded row chunks.

The overlap remains strongly anchored during early denoising and is gradually released with `(1 - sigma)^4`; immediately freeing a wide band can relocate subjects and cause double images during blending. The outermost 32 pixels remain fully editable throughout to remove cropped VAE boundary artifacts. Explicit interior edit masks retain their normal schedule.

RGBA defaults to Disabled: outputs are RGB and use the selected image format. Enable the RGBA custom setting for transparent cutouts, then explicitly prompt: `This is an RGBA image with transparency. A red ceramic teapot, isolated. The image has an alpha channel and the background is transparent.` With RGBA Enabled, WanGP automatically saves four-channel output as RGBA PNG, even if JPEG is selected, retaining the model-generated alpha channel. Disabled writes directly into a three-channel CPU UINT8 buffer and allows the selected JPEG, PNG or WebP format. JPEG cannot preserve it. This is distinct from reference-image background removal.

Text conditioning uses WanGP's shared `TextEncoderCache` (100 MB CPU LRU per loaded pipeline). Keys include the final prompt, prompt templates, encoder identity/precision, tokenizer identity, processor configuration and ordered reference-image pixel hashes, including alpha. Cached single-image embeddings and masks are expanded for the requested batch, so seed and batch-size changes do not require re-encoding. Entries too large for the shared budget are not retained. Cache misses retain normal layer progress and cancellation; completed entries can be reused until evicted or the pipeline is unloaded. VAE encoding remains separate.

**KV Cache** is a separate custom setting (`qwen21_kv_cache`), Disabled by default, following SenseNova's dropdown convention. Disabled recomputes the conditioning prefix at every denoising step without retaining per-layer K/V tensors on GPU. Enabled retains the previous prefix-cache behavior for faster denoising with greater VRAM use. NAG remains supported in disabled mode using temporary negative-conditioning K/V that is released after each layer. This setting does not disable the CPU text-embedding cache.
