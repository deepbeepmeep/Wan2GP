# TinyVAE preview integration

## Scope and provenance

General config `tiny_vae_preview` accepts `disabled` (default) and `gpu`. A supported enabled architecture downloads its shared asset before model loading, then the WanGP pipe builder adds `tiny_vae` and `coTenantsMap["tiny_vae"] = "*"`. Unsupported architectures keep RGB previews. Disabled never invokes decoder preparation. CPU mode was removed at the user's request after native FP16 proved slow on the test CPU.

Adapted from GOvEy1nw's PR #2107, head `272fbafdee5930eb99ce8ec2d5a69a1749c7ce35`: decoder implementations, checkpoint specifications/compatibility, and LTX denoised-latent preview idea. The separate PR configuration tab, installer UI, animation/encoding/transport system, automatic CPU/OOM fallbacks and worker subsystem were not adopted. Upstream MIT/Apache notices and pinned checkpoint sources are in `shared/tinyvae/`. The original author is credited in the integration commit.

## Runtime

TinyVAE replaces RGB through the existing PIL-image preview command, renderer and API. Video output remains a strip of up to four frames. Decode requests are capped at approximately seven per denoising pass, including the final step. TAEHV runs sequentially with temporal memory intact, retaining only selected output frames; frame selection exactly matches full sequential decoding. H3/image decoders process selected frames separately. This bounds activation memory without independent decoding of temporal latent frames. It still computes temporal frames that are needed for context; large clips can increase preview latency.

Decoder-only weights load through `offload.load_model_data(... writable_tensors=False, default_dtype=None)`, preserve native tensor dtypes, and opt out of profile-wide FP32 conversion. CPU weights are initialized through meta modules, avoiding generation RNG consumption. The regular MMGP profile owns GPU transfers. Temporal decoding checks the generation abort flag between blocks and discards partial outputs. Explicit unload, model changes and cancellation use the normal MMGP release path.

MMGP's wildcard is symmetric and lazy: it does not eagerly load TinyVAE. Ordinary model pairs retain directional cotenant rules. Automatic incompatible stage changes retain universal cotenants while evicting other models. Explicit `unload_all()`/`release()` still unload everything. The source change is in the separate local `E:/ML/mmgp` repository; `supports_cotenant_wildcards` prevents running GPU previews against an older installed MMGP implementation. No package release or Git push was performed for this work.

## Checkpoints and publishing

`weights.json` records upstream revisions, original and published SHA-256 values, dtypes and tensor counts. All nine distinct decoders were uploaded to their matching DeepBeepMeep repositories under the shared local/remote `preview_decoders/` path. Qwen/Krea reuse the Wan checkpoint; Z-Image reuses Flux. `uploads.json` records the eight remote commit URLs. Remote LFS checksums were compared with each verified local file.

Reproduction: download the pinned `url` in each weights record and verify `sha256`. For `.pth`, use `torch.load(... map_location="cpu", weights_only=True)` and `safetensors.torch.save_file` with source/license metadata, then compare every tensor's keys, shapes, dtypes and values against the original. Other files are unchanged copies. Keep original tensor precision. Runtime URLs and expected published hashes are in `shared/tinyvae/decoders.json`.

## Validation (2026-09-26)

Environment: Python 3.11, PyTorch 2.10.0+cu130, RTX 5090, local MMGP 3.8.1 source with the cotenant patch. Tests used the isolated WanGP checkout and imported MMGP source through PYTHONPATH; the installed MMGP package was not replaced. Desktop/other GPU processes were active.

- Fresh download of the published TAEF2 checkpoint through the native WanGP downloader passed checksum validation. Final headless loading confirmed named Tiny VAE progress through the shared MMGP loader.
- Nine real decoder checkpoints: native dtype loading, GPU decoding, transformer/VAE/TinyVAE residency transitions, stable TinyVAE GPU storage across switches, explicit release. Small synthetic latent shapes; `decoder-checks.json` records cold component timings/peaks, not end-to-end performance claims.
- Flux 2 Klein 4B: headless `--test`, actual General UI construction with both modes, real 512x512 four-step generation, four previews received through the actual callback and API renderer. Final preview visually inspected. Initial 0/4 progress verified.
- LTX-2.3 Distilled 22B: headless `--test`, real 512x512 17-frame generation, seven four-frame previews over two stages; preview structure/colors inspected.
- Disabled/default: real generation with decoder preparation replaced by a failing sentinel; no decoder/download call and no TinyVAE MMGP component. Existing RGB previews remained available.
- Cancellation after a TinyVAE preview, then successful generation in the same session. Same-seed final image pixels with previews enabled/disabled were identical.
- Focused tests: directional/universal cotenant semantics, explicit unload, scheduler/abort/pass reset, unavailable architectures, temporal selection exact equality, abort during decoding and recovery.
- The required documentation MCP test passed with the original checkout's local `wangp-agent/skills` fixtures supplied read-only. Its first isolated-worktree run failed because those ignored skill files were absent; the new documentation itself was loaded from this worktree.

Other model families have decoder/load/contract checks, not full-generation validation in this task. Compilation, unusual derivatives and long/high-resolution clips were not validated. Test artifacts and playable output remain in the task's `tinyvae-validation` artifact directory.
