# Live preview contracts

WanGP keeps the existing latent-RGB preview as the default. The global TAE
mode currently binds only the validated `ltx2_22B`,
`ltx2_22B_distilled`, `ltx2_22B_1_1`, and `ltx2_22B_distilled_1_1` model IDs.

## Support table

| WanGP model profile | Architecture | Tiny VAE decoder | TAE support |
| --- | --- | --- | --- |
| `ltx2_22B` | `ltx2_22B` | `taeltx2_3` | Validated |
| `ltx2_22B_distilled` | `ltx2_22B` | `taeltx2_3` | Validated |
| `ltx2_22B_1_1` | `ltx2_22B` | `taeltx2_3` | Validated |
| `ltx2_22B_distilled_1_1` | `ltx2_22B` | `taeltx2_3` | Validated |

Other LTX profiles, including 19B, Edit Anything, MSR, and quantized derived
profiles, fall back to RGB when the global mode is TAE until their callback
latent contracts are separately smoke-tested.

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
- Mapping: `ltx2_22B` architecture → `taeltx2_3`, patch size 4, latent channels 128

Weights are located through WanGP's configured checkpoint roots at
`preview_decoders/taehv/taeltx2_3.safetensors`. They are loaded lazily and
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
