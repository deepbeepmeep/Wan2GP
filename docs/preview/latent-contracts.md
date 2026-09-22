# Live preview contracts

WanGP keeps the existing latent-RGB preview as the default. TAE requires an
explicit model capability and a compatible registered architecture. The
contracts below cover LTX, H3, the listed image families, Wan, and Hunyuan.
Strict real-weight loading and decoder rendering are distinct from full-model
generation evidence; see the [measured validation record](validation-2026-09-22.md).

## Support table

| WanGP model profile | Architecture | Tiny VAE decoder | TAE support |
| --- | --- | --- | --- |
| `ltx2_22B` | `ltx2_22B` | `taeltx2_3` | Existing registration; generation not rerun in this matrix |
| `ltx2_22B_distilled` | `ltx2_22B` | `taeltx2_3` | Existing registration; generation not rerun in this matrix |
| `ltx2_22B_1_1` | `ltx2_22B` | `taeltx2_3` | Existing registration; generation not rerun in this matrix |
| `ltx2_22B_distilled_1_1` | `ltx2_22B` | `taeltx2_3` | Existing registration; generation not rerun in this matrix |
| `ltx2_25_22B` | `ltx2_25_22B` | `taeltx2_3` | Source contract; Dev generation unrun |
| `ltx2_25_22B_distilled` | `ltx2_25_22B` | `taeltx2_3` | CUDA off/off/RGB/TAE final pixels exactly equal |
| `ltx2_19B` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |
| `ltx2_19B_nvfp4` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |
| `ltx2_distilled` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |
| `ltx2_distilled_gguf_q4_k_m` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |
| `ltx2_distilled_gguf_q6_k` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |
| `ltx2_distilled_gguf_q8_0` | `ltx2_19B` | `taeltx_2` | Source contract; strict CPU load; generation unrun |

Other LTX profiles, including Edit Anything and MSR, fall back to RGB when the
global mode is TAE until their callback latent contracts are separately
smoke-tested.

## Original LTX-2

Source review shows the original `ltx2_19B` callback supplies the unpatchified
`C,T,H,W` diffusion x0 tensor with 128 channels. It uses the same standard
TAEHV constructor topology as the existing LTX adapter: patch size 4 and three
temporal downscale/upscale stages. This supports the `taeltx_2` registry
contract only. Strict loading of the pinned weight passes on CPU; live
original-LTX generation has not yet been verified.

## Source-contract-gated image and baseline video mappings

| Architectures | Decoder | Contract and evidence |
| --- | --- | --- |
| `flux`, `z_image` | `taef1` | Static `C,1,H,W`; source contract and CPU strict load only. Z normal-solver callbacks can be noisy; its unified solver provides x0. |
| Flux2 Klein and Ideogram4 | `taef2` | Static `C,1,H,W` after the family callback unpacking; source contract and CPU strict load only. |
| Qwen Image and Krea2 | `taew2_1_image` | Generated-only static `C,1,H,W`; no production-VAE mean/std inversion; source contract and CPU strict load only. |
| Wan 2.1/2.2 baseline | `taew2_1` / `taew2_2` | `C,T,H,W`, including the 48-channel patch-2 TI2V contract; source contract and strict CPU decode only. |
| Hunyuan Video / 1.5 | `taehv` / `taehv1_5` | `C,T,H,W`, with 16/32 channels and patch 1/2 respectively; source contract and strict CPU decode only. |

The image adapters reject non-finite values, wrong channel counts, and every
batch other than one. They publish one static RGB frame and never reinterpret a
batch as video. `wgp.py` also requires an image adapter for image output and a
temporal adapter for video output, otherwise it falls back to RGB.

Wan `any2video.py` lines 1751-1773 removes reference and trimmed frames before
publishing the `C,T,H,W` callback payload. Hunyuan's `hunyuan.py` selects
`HunyuanVideoPipeline` around line 630; that pipeline publishes `C,T,H,W`
callbacks around line 1775. These are post-scheduler states, not claims of
denoised x0. All four decoder weight contracts strict-load on CPU and decode a
zero `C,2,8,8` latent to five frames, but full target-model generation,
quality, performance, cancellation, and compile behavior are not yet run.

## LTX-2.3

The denoising callback supplies an unpatchified `C,T,H,W` tensor with 128
channels. The adapter validates finite values and converts it to TAEHV's
`N,T,C,H,W` order without applying production-VAE mean/std scaling. TAEHV
decodes into RGB `[0, 1]` frames; the adapter uniformly selects frames,
resizes them to the configured edge, and transfers only those `uint8` frames
to CPU before MP4 encoding, with animated WebP fallback.

Each denoising step starts a fresh decode. `StreamingTAEHV` state is never
reused between steps because it would mix different diffusion states.

## Decoder provenance

- Decoder: `taeltx2_3.safetensors`
- Source: `madebyollin/taehv`
- Immutable source revision: `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`
- Size: `23,531,296` bytes
- SHA-256: `f0773b4e3e57318e6aa4dd4a35e1d16213a5f160fbc0376163f06888bbcbe246`
- License: MIT; see `LICENSES/taehv-MIT.txt`
- Decoder: `taeltx_2.safetensors`
- Source: `madebyollin/taehv`, immutable revision `011dfc2112197741c540e0bdd5b7b67bcc930771`
- Size: `23,531,296` bytes
- SHA-256: `6e4cc0469134213d0101a46877ea2bce1dc7cf06ff5f5aefb9e4076c03542f7b`
- Mapping: `ltx2_19B` architecture -> `taeltx_2`, patch size 4, latent channels 128
- Mapping: `ltx2_22B` architecture → `taeltx2_3`, patch size 4, latent channels 128

Weights are located through WanGP's configured checkpoint roots at
`preview_decoders/taehv/<decoder>.safetensors`. They are loaded lazily and
never downloaded by tests or at application import.

## Limits and fallback

TAE samples the decoded clip at 16/8/4/2 preview FPS (up to 1024 samples) and
prefers fragmented H.264/NVENC MP4; animated WebP retains all selected samples
when MP4 is unavailable. If animated encoding itself fails, the first frame is
published as a static WebP. Decoder failures,
missing/corrupt weights, and preview OOM fall back to RGB and never fail
the generation.

The WebUI's **Configuration > Previews** tab exposes the TAE-only update rate,
device, maximum edge, Preview FPS, and WebP fallback quality. Samples span the
decoded clip at the selected transport rate without changing generation
frames. Programmatic clients use the `_preview` envelope documented in
`docs/API.md`; `PreviewMedia.to_dict()` is available for JSON bridges, while
the in-process API keeps binary bytes unencoded.

Tiny VAE previews are diagnostic and intentionally softer than final VAE
output. They do not alter seeds, sampler state, denoising values, final VAE
decoding, or completed/uploaded-video preview handling.

Wan, Hunyuan, and the listed static image mappings are source-contract-gated
registrations. Their normal callback state quality is not a denoised-x0 claim;
unlisted variants retain RGB until separately traced.
