# Validation record

Tested on NVIDIA RTX 5090 32 GiB, PyTorch 2.10 CUDA 13.0, Transformers 4.54 and Diffusers 0.36. No dependency upgrade/downgrade, distributed execution or internal model offloading.

## Checkpoints

Original transformer: 297 BF16 tensors. Original text encoder: 750 BF16 tensors. Native VAE: 238 FP32 tensors. Merged source tensors were checked for exact key/dtype/value equality before removing shards. Main and text encoder INT8 conversion ran through a headless `--test` queue. Persistent transformer conversion remains supported; the temporary text encoder conversion hook was removed.

The optional standard Quanto VAE experiment failed because its module replacement removes the custom single-frame convolution wrapper. This experimental file stays outside the checkpoint root and is not uploaded or advertised.

## Actual generation

* BF16 load-only, text generation and reference editing.
* INT8 text generation, single-reference orange-cat edit and transparent RGBA generation.
* Masked denoising, LanPaint, RGB and RGBA outpainting through real headless queues.
* Native 2048-square generation, negative attention guidance at CFG 1, two-reference composition and batch size two.
* Synthetic nonzero LoRA on real INT8 weights. Diffusers and Kohya names produce exactly identical images and differ from the no-adapter run. This tests adapter application, not the quality of a trained public LoRA.
* Both model-specific enhancer profiles exercised through the local Qwen3.8/Bonsai enhancer. Generation produces a scene description; editing produces an action and preservation clause. Enhancement remains off by default.
* Unit tests cover cached/uncached transformer agreement, sequential joint CFG, neutral and active NAG, key round trips, collision failures and the old Qwen fused-QKV aliases.

Downloaded reference fixture: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png . Outputs were numerically checked and representative images visually inspected.

## Optimization

The working pre-optimization source backup is at `D:/ml/wangp/qwen_image_21_preparation/backup_before_optimization`. Optimization followed the local `skill-optimize-model/SKILL.md` after that backup.

Checklist: read-only/native-precision loading done; shared attention done; inference-only execution done; RMSNorm/FFN chunking/in-place residual and RoPE done; prefix projection/layout reuse done; layerwise sequential CFG done; cache invalidation/cleanup done; tiled CPU UINT8 output done; actual inference and cancellation regression exercised.

On the fixed-seed 1024-square transformer regression, mean image difference versus the saved baseline was 0.96/255. Overall peak allocated memory remained 14.41 GiB, dominated by earlier stages; no overall speed improvement is claimed.

For isolated **2048-square VAE decoding**, using native FP32 weights and 256-pixel tiles:

| Output path | Peak allocated VRAM | Output storage | Decode time |
|---|---:|---:|---:|
| Retained float tiles | 2.211 GiB | 64 MiB GPU FP32 | 2.38 s |
| Streamed UINT8 | 2.109 GiB | 16 MiB CPU UINT8 | 2.93 s |

Streamed output is pixel-identical to the reference float result after UINT8 conversion. The reference and streaming paths use the same improved tile blending. Tile blending reduces the 512-square reconstruction MAE versus untiled decoding from 0.00588 to 0.00348 in the [-1, 1] range and removes the observed colored boundary stripes. Tiled and untiled decodes are not mathematically identical because each tile has less spatial context.

Detailed queues, samples, logs, checkpoint inventory and metrics are retained at `D:/ml/wangp/qwen_image_21_preparation/`.


## Final regression and publishing

Seven targeted unit tests passed. The final headless UINT8 queue completed all six tasks and saved eight distinct RGBA images, including generation and reference-edit batches of two. Both CPU and CUDA default-device runtime tests completed denoising cancellation, VAE cancellation, cleanup and successful repeated generation.

Two 2048-square, 40-step English/French infographics were visually inspected. Both have coherent layouts and substantial readable copy, but the English sample contains a spelling error and the French sample has several misspellings and incorrect panel numbers. These are raw model outputs, with no text repair. Do not interpret text-rendering support as guaranteed exact transcription.

Validated checkpoint upload: https://huggingface.co/DeepBeepMeep/Qwen_image_2/commit/a629f8f95c11cc18b8dbd11137452f14a7c9aa9f . All 22 file sizes and every safetensors SHA-256 were verified against the local manifest. The experimental VAE INT8 file was excluded.

Deepy validation used the real WanGPSession catalog: speciality search for `infographics` plus the `text writing` alias returned an exact Qwen Image 2.1 match. Both assistant-facing template catalogs include the new preset, settings resolve to 40 steps/CFG 4, edit references route correctly, and both full and compact model help expose the guidance. Evidence: `deepy_validation.json` in the preparation directory.


## Second memory audit (4K)

The first 4K headless attempt used an isolated test config with VAE tiling explicitly **Off**. It reached 32,030 MiB total GPU usage in `nvidia-smi` and was stopped during the untiled decode. This was not a failure of the tiled algorithm. The isolated test config now uses Auto; Auto additionally accounts for output size, batch size and reference size, while explicit Off remains respected.

The transformer audit fixed retained caller Q/K/V references, handed disposable normalized activations through cleared lists, reused FFN input storage for its output, released FFN intermediates each chunk, bounded native RMSNorm/RoPE scratch, and replaced full per-token modulation arrays with broadcast slices. Actual first-block FFN projection inputs ranged from 21,844 to 22,065 tokens for the 65,536-token image plus prompt; expansion ratio 3 therefore bounds the expanded activation to approximately the full input size. Small text sequences are not chunked.

Using real INT8 weights, SageAttention and image offload profile 1, both CFG branches on both tested steps were **bit-identical** to the saved transformer implementation. Measured live PyTorch allocation peaks:

| 4096-square phase | Before | After |
|---|---:|---:|
| First transformer step | 12.778 GiB | 10.520 GiB |
| Cached transformer step | 12.194 GiB | 10.464 GiB |

These are allocation measurements, not `nvidia-smi` totals including allocator reservations. Baseline and revised runs use the same instrumentation and offload profile. Timing was approximately 18 seconds per cached step in both runs; no speed improvement is claimed.

The single-frame VAE path no longer retains unused temporal feature caches or clones its residual shortcut unnecessarily. A real-checkpoint, 512-square tiled reconstruction remained pixel-identical in FP32, with an isolated peak reduction from 2.103 to 1.662 GiB. BF16 reduced that test peak to 0.831 GiB and remained finite; FP16 produced nonfinite decoder values and black regions and is excluded. The explicitly requested 16-bit execution mode therefore uses BF16 through MMGP. Uploaded checkpoint tensors remain original FP32; 32-bit runtime selection remains available.

The two-step 4K audit completed all 484 tiles in about 11–12 seconds. Hooks confirmed actual BF16 decoder inputs and weights and checked every decoder output tile for finite values. Whole-process allocated peaks during that phase included already resident models and were about 6.78 GiB; this is not the isolated VAE footprint.

The initial 4K memory audit used a temporary queue with mojibake from a Windows-default text read. Its before/after transformer comparisons used identical input bytes and remain valid as numerical/memory tests, but its image is excluded from typography quality comparisons. The corrected French queue explicitly uses UTF-8 and ASCII JSON escapes, with exact prompt equality and a SHA-256 recorded against the original 2K test.

Final real-checkpoint BF16 progress/cancellation regression passed under both CPU and CUDA default devices: 36-layer text progress, cancellation during text encoding, cancellation during denoising, 36-tile VAE progress and mid-decode cancellation, complete hook cleanup, and successful subsequent 1024-square generation. The subsequent output is bit-identical across the two default-device settings. See `runtime_bf16_final/phase_events.json` and `metrics.json`.


## Corrected 4K typography and precision checks

The UTF-8-verified French queue completed 40 steps at 4096 square through the actual headless runner. It did not improve exact typography/layout adherence over 2K: the result has eight panels instead of six, duplicated roasting panels, incorrect numbering and spelling mistakes. Both original 2K and 4K outputs and the exact input prompts are in the temporary image gallery. No text has been repaired.

The headless audit exposed a missing `VAE_dtype` forwarding argument in the family dispatcher, which has been fixed and covered by a routing regression test. The 40-step headless image above used FP32; its decode made 484 calls in 12.42 seconds and peaked at 6.824 GiB including other resident allocations.

The same saved final-generation latents were then decoded in BF16 with every tile checked for finite values. Actual decoder input and weight dtypes were BF16, all 484 tiles completed, and the **isolated 4K VAE peak was 0.831 GiB**. Relative to the FP32 image, mean pixel difference was **0.197/255**, maximum 14. Both raw decodes are available for comparison. This isolates decoder precision; it is not a second diffusion run.

The revised BF16 feature suite passed real NAG, synthetic LoRA application, exact Diffusers/Kohya output equivalence, two-reference composition and batch size two. The two-reference result was visually inspected and preserved both subjects.

The final headless regression completed all four tasks and saved six distinct images (masked editing, RGBA outpainting, generation batch two and reference-edit batch two). Audit hooks confirmed BF16 decoder inputs and weights on every task after fixing family precision forwarding. Ten targeted unit tests passed. The real Deepy discovery/template/help validation was rerun successfully after the final changes.


## ConvRot and shared encoder correction (2026-09-20)

- Reproduced the shared Ideogram tokenizer list-valued `extra_special_tokens` incompatibility; fixed loading without editing shared assets or changing installed Transformers. The preprocessing config is located independently, so split checkpoint roots work.
- Support both RoPE config schemas and encoder key layouts. Removed the unused LM head from the runtime and encoder BF16/ConvRot checkpoints. Verified all 749 retained BF16 tensors exactly before replacing the local full checkpoint.
- Headless `--save-quantized --convrot --test` converted 224 transformer and 360 encoder weights. The temporary encoder conversion hook was removed after use. `tools/prepare_qwen3_vl_encoder.py` reproduces the lossless encoder-only repack.
- Actual WanGP headless generation and reference editing passed at 512 square, 20 steps, CFG 4, with both ConvRot files, BF16 VAE, and E:/ML/Wan2GP/ckpts ahead of D:/ml/wangp/ckpts. Both images were visually inspected: clean red teapot and orange cat edit. Artifacts: `D:/ml/wangp/qwen_image_21_preparation/convrot_outputs`.
- All 12 Qwen21/LoRA regression tests passed, including the shared-tokenizer crash regression and encoder key remapping.
- Shared BF16 (16,289,676,320 bytes) and ConvRot (8,939,652,364 bytes) encoder files live in DeepBeepMeep/Ideogram4, commit c99227c9888cb5fe047b184ca03a5d968f40fe6d. That repository previously had only FP8/NF4, so a shared BF16 was added rather than pretending one already existed.
- Transformer ConvRot (7,256,804,890 bytes) replaced Quanto in DeepBeepMeep/Qwen_image_2, commit a7fdbd4d311bfdd50e1f589224e5e5c32f8d10c0. Duplicate BF16 and Quanto encoder weights were removed there. Remote checkpoint sizes and SHA-256 hashes were verified. `convrot_upload_result.json` records manifests and deletions.
- Earlier Quanto measurements above are historical; they do not establish ConvRot memory/performance numbers.
- ConvRot feature validation passed NAG, identical Diffusers/Kohya LoRA outputs with a nonzero adapter effect, two image references, and batch size 2. Runtime validation passed text/denoising/VAE cancellation, hook cleanup, and successful generation afterward under both CPU and CUDA default devices; the two outputs were identical. See `convrot_features/metrics.json` and `convrot_runtime/metrics.json`. All four declared checkpoint URLs returned HTTP 200.


## UI, capability and outpainting corrections

- Factory settings explicitly select image mode, and the family settings migration repairs saved video mode while retaining inpainting mode.
- Browser-tested on the isolated server at port 7865: image-only model selection, 4096p category and 4096x4096 choice, visible outpainting checkbox and four margin sliders after selecting Control Image, and visible outpainting in Image Inpainting. Test server stopped afterward.
- Infographic speciality removed. Full/compact help and model card now describe moderate text quality suited to short copy. NAG controls are hidden through the existing model flag; implementation and numerical tests are retained. Updated end-user model description explains generation, references, masks, outpainting, 4K and transparent PNGs.
- Real outpainting crop validation: the VAE encoded only a 416x352 source from a 512x512 canvas at offset (top=96,left=0). Replacing grey padding by red/transparent padding produced exactly identical output. Only the source crop reaches the vision/text encoder and VAE. Source latents are placed on the aligned canvas, with a 32-pixel transition band at expanded edges. Reduced-strength runs retain the full margin schedule and batch size 2 passes.
- Artifacts: `D:/ml/wangp/qwen_image_21_preparation/outpainting_crop_transition/metrics.json`, source and output PNGs. The transition-band output was visually inspected and the previous hard rectangular seam was substantially reduced.
- All 14 Qwen21/LoRA tests passed, including crop invariance, exact latent placement, 32-pixel model alignment, 4K category, capability metadata, and NAG UI visibility.
- The prior transparent dragon sample was checked: RGBA PNG, alpha range 0..255, 5033 fully transparent and 142641 partially transparent pixels.
- Actual shared save-path probe with a four-channel UINT8 tensor and JPEG preference saved PNG in RGBA mode, preserving all four channels byte-for-byte. Latest headless outpainting output was reopened and confirmed RGBA. VAE config has out_channels=4 and z_dim=64.


## RGBA output setting

- Added custom setting `rgba`, Disabled by default, with Disabled/Enabled dropdown choices. Disabled emits RGB into a three-channel CPU UINT8 buffer; Enabled emits RGBA. The decoder architecture remains four-channel. Deepy templates default to Disabled and help explains enabling RGBA before requesting transparent PNGs.
- Actual two-task headless queue with JPEG preference and batch size 2: default Disabled saved two RGB JPGs; Enabled saved two RGBA PNGs. Reopened all four files and checked modes/extensions. Evidence: `rgba_setting_headless.log`, `rgba_setting_outputs`.
- Real VAE at 512 square: tiled RGB output is pixel-identical to the first three channels of RGBA/float decoding, with output buffer 786432 vs 1048576 bytes. Untiled batch-size-2 RGB also matches RGBA's RGB bytes exactly. Evidence: `vae_rgb_validation/metrics.json`.
- Headless cropped outpainting passed normal image mode. The initial inpainting ratio task was skipped for lacking a mask; rerunning with an explicit empty mask passed (see `outpainting_crop_ratio.log`). Both outputs were stored RGBA before introducing the new Disabled default.

## Wider outpainting overlap with pixel restoration

- Added adaptive 128–256px overlap, reduced for small sources, aligned to 32px. Original unmasked core pixels are restored exactly after decoding. CPU blending uses 32-row chunks and premultiplied alpha; painted edits remain editable.
- Real ConvRot/BF16-VAE comparison at 512x512, seed 42, 20 steps: immediately freeing the wide overlap caused a double subject. Gradual release alone retained a dotted VAE crop outline. Final implementation combines gradual release `(1 - sigma)^4` with an always-editable outermost 32px strip. Visual inspection found the double subject and dotted outline removed, with a smoother wall transition than the previous 32px-only result. This is one image comparison, not a guarantee for all subjects.
- Final validation: exact original core, identical output for opaque-grey vs transparent-red padding, VAE sees only the 416x352 source, reduced-strength outpainting executes all steps, batch-size-2 succeeds. Evidence: `D:/ml/wangp/qwen_image_21_preparation/outpainting_overlap_final/metrics.json` and PNGs. Real inference used normal masked denoising; LanPaint was not rerun in this change.
- All 11 Qwen21 tests passed, including RGB/RGBA restoration, fully transparent hidden RGB, edit-mask protection, batch consistency, untouched new margins, small-source overlap and 256px cap.

## Switchable 32px edge-padding experiment

- Added `custom_settings.outpainting_borders`: `Edge Padding` (new default experiment) or `Overlap Blend` (preserves the preceding wide-overlap algorithm). Updated handler declarations/help, defaults and both Deepy templates.
- Edge Padding replicates all source channels into a 32px border on every side for VAE encoding only, then removes two latent cells on every side and releases padded latent storage. The vision/text encoder still sees the unpadded source. Actual output size, source placement, other references and non-outpainting paths retain their existing behavior. VAE Auto tiling accounts for the larger encoding footprint.
- Real ConvRot/BF16-VAE test: source 416x352 becomes 480x416 only at VAE input. Assertions confirm the exact replicated pixels, original latent placement, identical results for unrelated canvas padding colors, full-margin denoising at reduced strength, and batch size 2. Artifacts: `D:/ml/wangp/qwen_image_21_preparation/outpainting_edge_padding/`.
- Visual review: subject remains in place, but edge artifacts are still visible; this is an experiment, not a demonstrated quality improvement. Edge Padding deliberately omits overlap blending and original-pixel restoration to isolate the proposal.
- All 12 model tests passed, including exact replicated corners, latent crop dimensions/content, release of padded latent storage, unchanged unpadded encode path, and switch declaration.
- Selecting Overlap Blend passed the same real-checkpoint validation, and its 20-step comparison PNG is byte-identical to the pre-switch result. Evidence: `outpainting_switch_overlap/metrics.json` and `grey.png`.

## Red Canvas replacement

- Replaced Edge Padding with Red Canvas as the default border method. Saved Edge Padding values migrate, including direct queue compatibility. Overlap Blend remains selectable. Removed the replicated-edge VAE encoding helper.
- Both encoders see a full-size canvas with opaque red margins and unchanged original pixels. The pipeline appends the existing Qwen/FLUX red-padding replacement instruction once to the positive prompt. The family prompt hook defers to the 2.1 pipeline so Overlap Blend does not receive an inappropriate red-padding instruction.
- All 12 model tests passed. Real ConvRot/BF16-VAE validation passed full-size red VAE input assertions, padding-color invariance, reduced-strength full-margin schedule and batch-size-2 output. Artifacts: `D:/ml/wangp/qwen_image_21_preparation/outpainting_red_canvas/`.
- Visual inspection: red margins were replaced, but a prominent rectangular seam remains in the cat test. This implementation is not evidence of improved border quality. LanPaint was not revalidated for this change.

## Shared encoder asset consolidation and Red Canvas boundary correction

- Consolidated all ten encoder support assets into `DeepBeepMeep/Ideogram4/Qwen3-VL-8B-Instruct`. Differing files use generic suffixes: `config_legacy.json`, `tokenizer_legacy.json`, `tokenizer_config_legacy.json`. Identical chat template reused; other missing support files added. SHA256 verification preceded deleting the old Qwen_image_2 folder (commit d98dfef2fb1d59dbe37ad025f342acf321728e1a). Generic local files now match Ideogram4, while legacy variants retain the older export.
- Explicit legacy tokenizer paths preserve exact backend JSON equality and multilingual/special-token outputs versus the previous tokenizer. All 12 model tests passed; real ConvRot/BF16 VAE outpainting and batch validation passed after consolidation. The loader does not depend on generic tokenizer config files from another checkpoint root.
- Fixed normal image-mode guide padding by declaring `guide_inpaint_color=FF0000`; previously only the inpainting-mode color was declared. Shared canvas preprocessing probe now produces red padding. Probe artifact: `red_canvas_preview_probe.png`.
- Ordinary Red Canvas outpainting now uses a single instruction-guided edit without original-latent reinjection. Shared padding-only masks do not trigger masked processing. Explicit painted source edits still use masked denoising. This removes the hard source/margin processing discontinuity, at the cost of allowing source content changes.
- Same-seed 20-step comparison in `outpainting_red_instruction/` removed the conspicuous rectangular seam; padding-color invariance and batch-size-2 checks passed. Actual WanGP headless image-mode queue also completed 1/1 tasks with normal shared preprocessing and RGB output: `red_canvas_final_headless/`. Visual inspection found no comparable hard rectangular boundary. These examples do not guarantee perfect source preservation or artifact-free output for all images.

## Text conditioning cache

- Connected the pipeline to the shared 100 MB CPU LRU `TextEncoderCache`. Single-image embeddings, attention masks and image-token masks are cached before batch expansion. Keys include final prompt/templates, encoder identity and precision, tokenizer identity, processor config, and ordered RGBA image hashes.
- All 13 model tests passed, including repeated references by content rather than object identity, image-order and alpha changes, template changes, batch-size reuse, and avoiding mutation of cached tensors.
- Real ConvRot/BF16-VAE run: first positive/negative encoding called the encoder twice; identical second generation made no additional calls and produced identical pixels. A changed positive prompt with batch size 2 added only one encoder call (3 total). Cache occupied 7,078,752 bytes. The first output PNG also matched the pre-cache baseline byte-for-byte. Evidence: `D:/ml/wangp/qwen_image_21_preparation/text_cache_validation/metrics.json` and output PNGs.

## Hybrid inpainting plus Red Canvas outpainting

- Combined mode now splits the prepared mask at the original source rectangle. Outpainting margin pixels are removed from the denoising mask. Only margins are painted red in the reference; source interior pixels are retained for normal VAE conditioning.
- The painted interior follows the original/noise trajectory until the start step selected by denoising strength, then generates normally. Margins run the complete schedule as an instruction-guided red-canvas edit. No hard mask anchoring is applied across the outer source boundary. This hybrid uses the strength schedule rather than LanPaint. Outside the interior edit, source preservation remains model-guided.
- All 13 unit tests passed, including removal of margin-only masks and retaining only painted interior pixels. Real combined-mask validation passed expected VAE pixels (interior not painted red), output invariance to incoming padding colors, cached conditioning, and reduced-strength batch-size-2 generation with all four steps. Evidence: `red_canvas_hybrid_mask/metrics.json`.
- Actual Inpainting-tab headless settings (image_mode=2, VAG, interior mask plus top/bottom outpainting, strength 0.6) completed 1/1 tasks. Visual inspection showed the requested blue butterfly and extended scene without the earlier rectangular processing seam. Evidence: `hybrid_inpaint_outpaint_headless/` and corresponding log.

## Optional denoising KV cache

- Added SenseNova-style KV Cache dropdown (`qwen21_kv_cache`): Disabled by default in handler, defaults, Deepy templates and runtime fallback. Disabled passes no persistent K/V cache to the transformer; Enabled preserves the existing per-layer prefix cache. CPU text-embedding caching remains active independently.
- NAG's disabled-cache path recomputes negative conditioning and frees each layer's temporary K/V after use, rather than accumulating a full GPU cache.
- All 13 model tests passed, including cached/uncached joint CFG and exact NAG extraction-vs-temporary-cache equivalence. Real disabled-cache ConvRot/BF16-VAE validation passed repeated generation, retained text-cache hits, and batch-size-2 reduced-strength generation. Artifact folder: `kv_cache_disabled_validation/`.
- Same-seed 20-step cached/uncached image comparison: mean absolute byte difference 0.9975/255, maximum 81/255; visual inspection found consistent content. Results are numerically close, not byte-identical. No fresh whole-pipeline VRAM benchmark was performed for this switch.

## 4096-square VAE memory investigation (2026-09-21)

- Saved application config had `vae_config: 1` (the UI's 16GB+ preset), `vae_precision: "16"`, image profile 1. The initial investigation incorrectly described this as Off: the model implementation wrongly mapped that preset to disabled tiling, bypassing the Auto area threshold. See the preset correction below. The user's config was not changed.
- Ran real 4096x4096 text-to-image with one denoising step, CFG 4, ConvRot checkpoints, KV cache disabled, image profile 1, BF16 VAE and Auto tiling. Actual decode used 484 tiles: 16x16 latent inputs -> 256x256 output pixels, stride 192. All tile outputs were finite.
- Denoising: 23.12s, peak allocated 10.37 GiB. At VAE entry, previous-model residency was 6.776 GiB allocated / 11.172 GiB reserved. After MMGP switched to the VAE, sampled tile-end allocations stayed at 0.644 GiB and reserved memory at 1.072 GiB through tile 450. VAE phase peak (including handoff) was 6.776 GiB; decoding completed in 16.07s. No progressive per-tile growth was observed.
- Instrumentation: `tools/validate_qwen21_memory.py`; evidence: `D:/ml/wangp/qwen_image_21_preparation/vae_4k_one_step_audit/metrics.json` and adjacent `.log`.
- Fixed a separate one-step scheduler issue: terminal sigma stretching divides by zero when the only input sigma is 1. One-step generation now disables terminal stretching; multi-step schedules are unchanged. The completed one-step test validates finite decode after this fix.

## Correction: 16GB+ tiling preset mapping

- Verified the configuration UI: 0=Auto, 1=16GB+, 2=8GB+, 3=6GB+. There is no Off choice. The previous explanation of preset 1 as Off was incorrect.
- Fixed Qwen21's preset-1 mapping: it now retains resolution-aware selection instead of forcing `use_tiling=False`. At 4K, both Auto and preset 1 use 256px tiles / 192px stride. Presets 2 and 3 directly enable tiling. User config is unchanged.
- All 13 model tests passed, with coverage for preset 1 at 1K and 4K and direct preset values. Repeated the actual 4096-square one-denoising-step run using image profile 1, BF16 VAE, and specifically `--vae-config 1`: 484 tiles, finite outputs, 12.38s decoding. Tile-end allocations stayed at 0.644 GiB; reserved memory 1.072 GiB. Phase peak including model handoff was 6.776 GiB. Evidence: `D:/ml/wangp/qwen_image_21_preparation/vae_4k_preset1_audit/metrics.json` and adjacent log.


## Fully resident VAE tile budget benchmark

Dummy 4096x4096 latent decoding, batch 1, RGBA CPU UINT8 output, entire real VAE resident on CUDA with no MMGP offload profile. Fresh process for each precision/tile combination. All outputs had finite decoded tiles. Total parameter residency: BF16 0.629 GiB, FP32 1.258 GiB.

| Precision | Tile / stride | Peak allocated GiB | Peak reserved GiB | Seconds |
|---|---|---:|---:|---:|
| bf16 | 256 / 192 | 0.831 | 1.076 | 10.76 |
| fp32 | 256 / 192 | 1.663 | 1.906 | 14.03 |
| bf16 | 512 / 384 | 1.412 | 1.961 | 6.85 |
| fp32 | 512 / 384 | 2.826 | 3.729 | 11.26 |
| bf16 | 1024 / 768 | 3.735 | 5.609 | 7.26 |
| fp32 | 1024 / 768 | 7.472 | 12.027 | 11.27 |

All allocated peaks fit the proposed 4/6/12 GiB VAE working budgets for 256/512/1024 tiles. FP32 at 1024 reserves 12.027 GiB, just above the provisional 12 GiB target, while live allocation peaks at 7.472 GiB. These figures exclude other models, desktop applications and non-PyTorch CUDA allocations; they are decoder memory tests, not image-quality or encoder benchmarks. No preset-size mapping was changed by this benchmark.

Reproduce using `tools/benchmark_qwen21_vae_tiles.py`; results: `D:/ml/wangp/qwen_image_21_preparation/resident_vae_tile_benchmark/summary.json`.
