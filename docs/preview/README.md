# Tiny VAE live previews

WanGP's Tiny VAE preview is an opt-in diagnostic preview. RGB remains the
default and fallback; Off disables preview work entirely. Source-contract and
CPU strict-load evidence are distinct from full-model generation validation.

## Supported profiles

The registry enables `taeltx2_3.safetensors` for the four standard 22B LTX
profiles listed in [latent-contracts.md](latent-contracts.md), and
`taeh3.safetensors` for the four MiniMax H3 FL2VA/Ref2VA profiles
(including pruned variants). Other profiles remain ineligible until their
callback latent contracts are independently verified.

LTX-2.5 Dev and Distilled use the same `taeltx2_3` decoder through the shared
architecture/capability registry. A complete LTX-2.5 Distilled CUDA comparison
confirmed identical final pixels with previews off, RGB, and TAE; Dev and other
settings remain separate validation cases. See the [measured validation
record](validation-2026-09-22.md). Missing or failed decoders retain RGB fallback.

The six original `ltx2_19B` defaults use `taeltx_2.safetensors` through the
existing LTX adapter. Source and strict-loader evidence do not establish
original-LTX generation quality.

Static image previews are explicitly registered for Flux/Flux2 Klein,
Z-Image, Qwen Image, Krea2, and Ideogram4. The static adapters accept only a
finite `C,1,H,W` latent and publish one RGB frame; batch and image/video
output-kind mismatches use RGB. Wan and Hunyuan baseline video profiles use
explicit TAEHV decoder contracts. Their callbacks are post-scheduler states,
not claimed denoised x0 values.

## Settings and installation

**Configuration > Previews** controls the global preview defaults. Preview
mode is **Off**, **RGB**, or **TAE (if available)**. Selecting TAE uses Tiny
VAE for eligible models with a valid decoder and automatically falls back to
RGB otherwise.

The remaining settings apply only to TAE previews. **TAE Preview FPS** evenly
samples the full decoded clip at 16, 8, 4, or 2 FPS without changing generated
frames. TAE prefers fragmented H.264 MP4 through NVENC and falls back to
animated WebP when unavailable. For example, 241 decoded frames at 24 source
FPS produce 160 uniformly distributed samples at 16 Preview FPS. API clients
can override the global defaults with the reserved `_preview`
envelope documented in [API.md](../API.md).

Selecting **Install Tiny VAE Preview Decoder** downloads the pinned decoder
through WanGP's existing download-progress plumbing to:

```text
<configured model root>/preview_decoders/taehv/<eligible-decoder>.safetensors
```

LTX-2.3/2.5, Wan, Hunyuan, and Qwen use TAEHV weights in
`preview_decoders/taehv`; original LTX-2 uses `taeltx_2`. TAESD image weights
use `preview_decoders/taesd`; MiniMax H3 uses
`preview_decoders/taeh3/taeh3.safetensors`.

The file is checked by exact size and SHA-256 before it can be advertised or
loaded. Tests never download it.

## Troubleshooting

- **Missing decoder:** use the install button, or keep RGB selected.
- **Corrupt/incompatible decoder:** delete the file and install it again; the
  loader rejects size, hash, and strict state-dict mismatches.
- **Preview fallback warning:** the current generation continues with RGB;
  preview decode/encode failures do not fail generation.
- **CPU mode:** decoding is slower but avoids CUDA preview allocations.
- **VRAM pressure:** use Auto or CPU, lower Preview FPS/edge, or use
  RGB. CUDA OOM retries with sequential decode, then a CPU decoder before
  disabling Tiny VAE for that generation.
- **Soft output:** Tiny VAE is a low-cost diagnostic decoder, not final-quality
  VAE output. Standard `taeltx2_3` previews are intentionally softer.

## Architecture and diagnostics

The callback validates WanGP's `C,T,H,W` latent and converts it to each
decoder's expected layout. LTX uses temporal TAEHV decoding; MiniMax H3 uses
the pinned 2D TAE on each latent frame. CUDA decode is synchronous, while only
selected resized `uint8` frames are copied to CPU and encoded in a bounded worker.
There is at most one active encode and one replaceable pending job. Generation,
context, sequence, and cancellation tokens suppress stale publication.

Set `WANGP_PREVIEW_TRACE=1` on a target runtime to record model, architecture,
pass/window, shape, dtype, device, and latent statistics at eligible capture
callbacks. The pinned decoder was additionally verified in an isolated
`torch 2.11.0+cu128`/`safetensors 0.8.0` runtime on an RTX 4070 Ti SUPER:
the exact `taeltx2_3` weight hash and strict load passed, a fixed CUDA latent
decoded to 17 frames with 8 selected 64px RGB frames, and the real coordinator
published an 8-frame animated WebP. The original-LTX `taeltx_2` strict-load
check passes on CPU; original-LTX generation validation remains pending
verifier evidence. The full WanGP model-generation matrix, target generation-overhead
gate, and final-output equivalence remain deployment-time checks.

## Evidence and limits

The CPU strict-loader fixture checks pinned image, Wan, and Hunyuan decoder
weights and preserves the CPU torch RNG state. It does not run model
generation. The normal Z-Image solver callback can contain a noisy state;
the unified solver supplies x0. Wan and Hunyuan callbacks are documented as
post-scheduler state, not denoised x0. Generation quality, performance,
cancellation, and compilation remain unvalidated until target-model runs.
See [validation-2026-09-22.md](validation-2026-09-22.md) for the bounded
runtime evidence.

Future `taeltx2_3_wide` evaluation is intentionally not enabled by this
initial registry entry.

Completed/uploaded-video previews continue to use
`shared/gradio/video_preview.py`. No GPL KJNodes implementation is included.
