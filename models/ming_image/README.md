# Ming Image 0.1 Design provenance

Inference source: inclusionAI/Ming-Image, commit 578e21e081a7cd385b6c79e36df7e419155681b1 (MIT license). The files in `upstream/` are adapted from that revision: local imports, PyTorch RMSNorm in place of Transformer Engine, shared attention in the diffusion transformer, and cooperative abort checks. The VAE uses WanGP's existing Qwen Image implementation instead of a second vendored copy. See `LICENSE.upstream`.

Checkpoint source: inclusionAI/Ming-Image-0.1-Design, revision 208087ada1486931692c1896f38d4cd16ff3df82. The prepared distribution is published at [DeepBeepMeep/MingImage](https://huggingface.co/DeepBeepMeep/MingImage).

## WanGP scope

The `ming_image_0_1_design` model type supports text-to-image and single-reference image editing. An optional RGBA setting retains the model's alpha channel in PNG output. The separate Design-Layer checkpoint is not part of this model type.

The default prompt is the upstream four-season structured JSON example at 2048x2048. WanGP accepts that JSON directly and also accepts plain-language prompts. The Prompt Helper edits the JSON canvas settings and back-to-front visible layers on a visual canvas, using normalized boxes and hex colors. Whole lines whose first nonspace character is `#` are ignored for JSON parsing and preserved on save, including WanGP's `#!PROMPT!:` provenance line. It also accepts a Markdown-escaped `\#!PROMPT!:` line and saves it with the normal `#` prefix. Hex colors and inline `#` text remain intact. It can complete missing closing JSON brackets, remove Markdown-escaped underscores in keys, normalize doubled quote escapes from pasted JSON, and move numeric boxes inside the canvas before saving; other malformed JSON appears in a raw editor for correction. Loading and selecting layers do not count as edits, so closing an untouched prompt needs no discard confirmation. The optional prompt enhancer offers classic text and reference-edit prompts plus two JSON variants for those workflows; enhancement stays disabled by default. Every enhancer variant preserves exact user-supplied copy and writes suitable titles, labels, captions, and explanations when the design calls for them. Every intended visible string is quoted in full, with no unspecified text zones left for the image model. The JSON variants include a complete schema example, center-coordinate bounds, row-layout examples, exact-text ownership, and a closing-brace check. The JSON reference-edit variant adapts the upstream text-to-image rewriter guidance to WanGP's edit path.

MMGP manages the diffusion transformer, BailingMM2 text encoder, vision tower, and Qwen Image VAE as separate components. Design and Design-Layer share one Bailing language-model checkpoint and one vision checkpoint. Each variant has its own connector and conditioning projections. The transformer and each encoder component have BF16 and INT8 ConvRot files. The connector weights from upstream were FP32 and are converted to BF16 for this prepared BF16 variant, matching upstream's BF16 inference path.

## Prepared checkpoint layout

```text
ckpts/
  Ming-Image-0.1-Design_bf16.safetensors
  Ming-Image-0.1-Design_int8_convrot.safetensors
  Ming-Image-0.1-Design-Layer_bf16.safetensors
  Ming-Image-0.1-Design-Layer_int8_convrot.safetensors
  BailingMM2-Ming-Image/
    BailingMM2-Ming-Image-Core_bf16.safetensors
    BailingMM2-Ming-Image-Core_int8_convrot.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    preprocessor_config.json
    special_tokens_map.json
  ming_image/
    conditioning_bf16.safetensors
    conditioning_int8_convrot.safetensors
    transformer_config.json
    connector_config.json
    mlp_config.json
    scheduler_config.json
    vae_config.json
    vae.safetensors
    LICENSE.upstream
  ming_image_shared/
    vision_encoder_bf16.safetensors
    vision_encoder_int8_convrot.safetensors
  ming_image_layer/
    conditioning_bf16.safetensors
    conditioning_int8_convrot.safetensors
    transformer_config.json
    connector_config.json
    mlp_config.json
    scheduler_config.json
```

`prepare_checkpoints.py` reads the pinned upstream snapshot from `ckpts/.ming_image_source`, verifies every merged tensor, and removes source shards. `prepare_layer_checkpoints.py` and `convert_text_encoder_convrot.py` prepare the full source encoder variants. `split_encoder_checkpoints.py` streams them into the shared core, shared vision, and variant conditioning files, checking every shared tensor byte against both source variants. The runtime loader combines the core and selected conditioning file through one MMGP `load_model_data` call; it loads the vision file as a separate MMGP model. The transformer ConvRot file is generated through WanGP's persistent `--save-quantized --convrot --test` path. The normal loader does not perform encoder conversion.

From the WanGP root, prepare a new checkpoint root with:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='inclusionAI/Ming-Image-0.1-Design', revision='208087ada1486931692c1896f38d4cd16ff3df82', local_dir='D:/ml/wangp/ckpts/.ming_image_source')"
python -m models.ming_image.prepare_checkpoints D:/ml/wangp/ckpts
python -m models.ming_image.convert_text_encoder_convrot D:/ml/wangp/ckpts
python -m models.ming_image.prepare_layer_checkpoints D:/ml/wangp/ckpts
python -m models.ming_image.convert_text_encoder_convrot D:/ml/wangp/ckpts --layer
python -m models.ming_image.split_encoder_checkpoints D:/ml/wangp/ckpts
```

Use a WanGP headless `--save-quantized --convrot --test` run for the transformer. This conversion uses the handler's persistent `save_quantized_model` call; the model type and a non-empty prompt must be present in the queue JSON.

The model defaults point to `DeepBeepMeep/MingImage`. Local testing uses the files in the configured checkpoint root.

## Local validation (2026-09-23)

The prepared BF16 and ConvRot files passed headless WanGP loading. Full 1024px generation completed with both formats. A twelve-step text-to-image run produced a clean red-circle sample; the BF16 and ConvRot samples differed by 1.49 mean pixel levels on a 0–255 scale. A twelve-step reference edit changed the title to `OPEN STUDIO` and preserved the red background, ribbon, and frame after background removal was disabled. A shorter edit smoke run exercised the final reference reuse and vision-layer progress path.

Two poster tasks ran in one queue, at CFG 1 and 1.5. The 1.5 case also completed with sequential CFG branches within each diffusion layer; its image differed from the batch CFG baseline by 3.22 mean pixel levels. The poster title and fern layout were coherent, though fine footer text was imperfect. RGB output saved as RGB. The optional RGBA path saved a four-channel PNG, but the tested transparent-apple prompt produced alpha values of 252–255, so transparency should not be assumed.

The resolution menu retains square and rectangular WanGP presets, including custom presets, whose pixel count is no more than 2048x2048. A long side may exceed 2048 when the other side is shorter. The default is 2048x2048 to match the upstream structured JSON example and recommended square resolution; choose 1024x1024 for lower memory use. At 2048x2048 the text-to-image generation bucket is native 2048; other aspect ratios snap to the nearest upstream working bucket. Reference editing uses a 1024 working bucket, so larger edit outputs are resized. The previously exposed 4096x4096 mode merely resized a 2048 image and has been removed from the menu. A twelve-step botanical poster completed directly at 2048x2048 with VAE tiling (`--vae-config 2`) and saved an RGB PNG. Its main title and fern illustration were recognizable, but it invented fine text in the subtitle and footer. Use VAE tiling for 2048 output to limit decoding memory. Cancellation was not validated in this run.

A separate twelve-step run with the new four-season default and VAE tiling saved an RGB 2048x2048 PNG. All four season labels were legible and the fixed cabin composition carried across panels. A browser check of the Prompt Helper loaded the default JSON, saved an edit, added a box with Alt-drag, undid it, updated the aspect ratio for a rectangular resolution, and rejected malformed JSON.

The text-only JSON enhancer produced valid structured JSON and a rendered poster with both requested strings. The reference-aware JSON enhancer was checked with a downloaded wagon photo; after limiting edit prompts to a full-canvas base layer and requested overlays, two repeated runs produced valid JSON and in-canvas boxes. A browser regression check resized a layer to a canvas boundary and confirmed its outline stayed visible after three-decimal coordinate rounding. The Prompt Helper button uses the same small blue outline style as the prompt microphone.

## Inference memory and preview

Ming's VAE inherits the shared Qwen Image tiled encoder and decoder. WanGP
presets 1/2/3 use 1024/512/256 sample-pixel tiles, with 75% strides, following
Qwen Image 2.1's memory tiers. Auto selects a tier by device memory. The same
setting is applied before reference-image encoding and output decoding.
The released Ming VAE is RGBA with scalar latent scaling; it is not the Qwen
Image RGB VAE checkpoint with per-channel latent normalization. The shared
Qwen/Wan RGB preview factors therefore do not apply to Ming.

The transformer rotates disposable q/k in place, hands q/k/v to shared
attention, chunks large visual FFNs, and runs CFG branches sequentially inside
each loaded layer. Live RGB previews use a 16-channel linear fit from 1000
images in `E:/ML/images`; the reproducible fit is in `regress_preview.py`.
Measurements and validation are recorded in `specs/MING_IMAGE_OPTIMIZATION.md`.

Ming caches text and vision conditioning in the shared bounded CPU text
encoder cache. Text-only keys include the prompt and active encoder identity,
so changing output resolution or seed reuses the embedding when the prompt is
unchanged. Reference-image keys also include image pixels and target geometry,
because the reference is cropped and resized for the requested output shape.
If Prompt Helper saves a new aspect ratio into the JSON prompt, the prompt
itself changes and needs fresh text conditioning. Generation reports zero
denoising steps after reference VAE encoding and before the first transformer
step. Step previews use the fitted Ming RGB factors.
