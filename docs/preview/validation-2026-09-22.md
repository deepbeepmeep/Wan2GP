# Tiny VAE validation — 22 September 2026

This records measured results separately from the source-verified compatibility
table in [latent-contracts.md](latent-contracts.md). Registration does not imply
that every architecture, sampler, offload profile, or conditioning mode has been
tested in a full generation.

## Environment

- Official WanGP base: `0d8ea0f61aed23f1d551825529839398c6392049`.
- Previous PR head: `708d72c3f5d83860f9bc61629fa9e3b1902c2c95`; merged base:
  `2b18bbc7`; integration implementation: `1a01c486`. Results below exercised
  that implementation (some runs preceded its commit with the same source).
- Windows, NVIDIA RTX 4070 Ti SUPER (16 GB), driver 616.56.
- Python 3.11.9, PyTorch 2.10.0+cu130, CUDA 13.0, mmgp 3.8.1.
- Public `shared.api` generation API; SDPA attention; offload profile 4;
  compilation disabled for the initial matrix.
- A process-local dependency/cache directory supplied mmgp 3.8.1 and the Numba
  cache. The existing AiVS Python installation and model store were not changed.
- Triton/Quanto reported a Windows permission error in its default cache and
  used its fallback. All four initial comparison cases used the same environment.

## LTX-2.5 distilled: complete generation comparison

Model: `ltx2_25_22B_distilled`, seed 2107, requested resolution 256×256,
8 inference steps and 9 frames. The effective two-stage pipeline produced
17 frames; comparisons use the actual `3×17×256×256` uint8 output tensor.
Prompt: “A small red paper boat drifting across a calm blue pond, cinematic
daylight.” Full effective API settings are retained in the local matrix artifact.

| Preview mode | Wall time | Final tensor versus first baseline |
| --- | ---: | --- |
| Off, first/cold | 67.420 s | Baseline |
| Off, repeat/warm | 23.307 s | Exactly equal |
| RGB | 21.523 s | Exactly equal |
| Tiny VAE | 24.623 s | Exactly equal |

All four final tensor SHA-256 values were
`b45641b813479c064e82ad8a5eb64a13994f585ee3dcea1cadb64662afc0b957`.
This comparison hashes decoded pixels, not timestamp-bearing container bytes.
The measured warm Tiny VAE run was 1.316 s longer than the warm off run; these
single runs establish observed overhead, not a statistically stable benchmark.

Tiny VAE emitted previews during inference in both stages. First-stage previews
contained 5 frames and used animated WebP (2,004–3,118 bytes); second-stage
inference previews used NVENC MP4 (41,336–47,444 bytes). The smaller first-stage
frames were rejected by NVENC and correctly fell back to WebP. A later valid
frame size successfully used NVENC in the same process.

Visual inspection of the second-stage Tiny VAE preview and final frame confirmed
matching boat/water composition and color, with the expected softer preview.
Cancelling on the first actual Tiny VAE preview returned the normal cancelled
result; a subsequent previews-off job succeeded with the exact baseline tensor.

Instrumented TAE follow-ups also reproduced the baseline. They observed CUDA
bfloat16 callback tensors in `C,T,H,W` layout: `128×3×4×4` in the first pass and
`128×3×8×8` in the second. The direct-timing run took 54.801 s, delivered its
first TAE preview at 32.119 s, and peaked at 4.266 GB allocated / 4.425 GB reserved
CUDA memory. First-preview decode/encode took 250.587 / 97.915 ms; the last
preview took 5.416 / 341.662 ms. These cold-run measurements are separate from
the warm comparison.

Two subsequent complete jobs selected isolated missing/corrupt decoder paths.
Both continued with RGB fallback, emitted no TAE media, and reproduced the exact
baseline tensor (22.491 / 22.256 s; 1.818 GB allocated / 1.896 GB reserved peak).
Existing model/decoder files were not damaged or renamed for these fault probes.

## Flux2 Klein 4B: complete image comparison

Model: `flux2_klein_4b`, seed 2107, 512×512, 4 inference steps. The off repeat,
RGB, and Tiny VAE outputs all matched the first off output exactly after decoding
the saved images to RGBA pixels (SHA-256
`b53626862f98ff97a4ae26f3c8f89ed206265ce7166ca2438ef6d7442f93e002`).

| Preview mode | Wall time | Peak allocated / reserved CUDA memory |
| --- | ---: | ---: |
| Off, first/cold | 8.085 s | 1,269 / 1,808 MB |
| Off, repeat/warm | 2.812 s | 1,139 / 1,808 MB |
| RGB | 2.884 s | 1,139 / 1,808 MB |
| Tiny VAE | 3.043 s | 1,142 / 1,810 MB |

Memory uses decimal MB. These are observed single-run peaks/times, not a
statistical benchmark. The first single-frame WebP preview arrived at 1.042 s,
before final decode. Visual inspection confirmed the expected noisy early state,
a recognizable boat by step 3, and matching final composition at step 4. This
callback supplies post-scheduler latents, not denoised x0.

The first attempt encountered a permission error reading the existing Qwen3
tokenizer template. Copying the nine small tokenizer/config files into the
isolated checkpoint root resolved it; model weights, the user's config, and
model-store permissions were unchanged.

## Z-Image and Qwen Image 2512: complete image comparisons

Both used seed 2107 and 512×512 output; `z_image` used 2 steps and
`qwen_image_2512_20B` used 4. Within each family, the off repeat, RGB, and TAE
outputs matched the first off output exactly as decoded RGBA pixels.

| Model / preview mode | Wall time | Peak allocated / reserved CUDA memory |
| --- | ---: | ---: |
| Z-Image off, first | 9.833 s | 1,328 / 1,583 MB |
| Z-Image off, repeat | 2.378 s | 1,015 / 1,583 MB |
| Z-Image RGB | 2.441 s | 1,015 / 1,583 MB |
| Z-Image TAE | 2.493 s | 1,018 / 1,585 MB |
| Qwen 2512 off, first | 39.006 s | 2,823 / 2,892 MB |
| Qwen 2512 off, repeat | 9.454 s | 892 / 919 MB |
| Qwen 2512 RGB | 9.452 s | 892 / 919 MB |
| Qwen 2512 TAE | 9.814 s | 1,370 / 1,518 MB |

The first TAE previews arrived during inference at 1.195 s (Z-Image, 4,986-byte
WebP) and 2.792 s (Qwen, 60,652-byte WebP). Visual inspection of each last TAE
preview against its final frame confirmed matching composition/color with the
expected decoder softness. The short 2-step Z-Image output is deliberately a
contract smoke test, not a model-quality benchmark.

Decoded pixel SHA-256 values:

- Z-Image: `7e8a83c8707559da25919876ed43a16d4a0dff39945ed496fb4c24ec31b31e26`.
- Qwen 2512: `3da0c926910cee51ffe46d7b933fd61a65d9cdbb59b58464a9d32eb58759eefa`.

## MiniMax H3 hybrid: complete video comparison

An isolated temporary finetune used the core `minimax_h3_fl2va_pruned`
architecture with the installed `hybrid_fl2va_ref2va_b25-49` checkpoint and
existing FP8 mixed-precision video VAE. The runtime detected the hybrid Ref2VA
AdaLN blocks 25–49. Settings: seed 2107, 256×256, the legal minimum 107 frames,
8 steps, guidance scale 1 and one guidance phase.

| Preview mode | Wall time | Peak allocated / reserved CUDA memory |
| --- | ---: | ---: |
| Off, first/cold | 63.804 s | 2,823 / 2,955 MB |
| Off, repeat/warm | 26.362 s | 2,528 / 2,592 MB |
| RGB | 26.311 s | 2,528 / 2,592 MB |
| Tiny VAE | 28.143 s | 2,533 / 3,077 MB |

All four final `3×107×256×256` video tensors were exactly equal (SHA-256
`02feb0184ca508a04ea61c4637e12a6e5158607880010f84af18aebeaa25f402`).
The first TAE MP4 arrived during inference at 3.760 s: 10 preview frames,
100,784 bytes, 219.717 ms decode and 135.144 ms encode. The last preview used
47.499 ms decode and 318.397 ms encode. No asset transfer occurred during this
successful matrix.

Visual inspection confirmed consistent scene orientation/color with the
expected coarser boat/foliage detail in the flat H3 preview. Actual MP4 decoding
verified 10 frames at the configured 8 FPS (1.25 s); the inspected LTX MP4 had
5 frames at 8 FPS (0.625 s). These are sampled diagnostic clips: transport
duration follows selected frame count and configured preview FPS, and need not
equal the full generated video's duration.

Decoded AAC audio also matched exactly across all four outputs: 32 kHz stereo,
139 decoded audio frames / 1,138,688 bytes, PCM SHA-256
`e02314adea534dd79715c4e622ecd0b3794a83033ac3743e8ad7afbe2fff716f`.

A separate first-frame-conditioned comparison used the generated final frame
as an input image. The public API inferred `image_prompt_type: S`; all other
short-run settings stayed fixed. Off and TAE completed with exactly equal final
video tensors (SHA-256
`e06d25ad48352662226e5dec8361fd90c7cedb3d5283f0f7845afb595e468e7a`),
and TAE emitted 8 real MP4 previews. This exercises the preview-only conditioning
path; arbitrary masks, reference-mode inputs and tiled/multistage H3 remain
separate cases.

The existing model store lacked the mandatory 690,592,992-byte H3 latent
upscaler. It was downloaded only into the isolated validation checkpoint root
after reporting its size/disk requirement, and verified against SHA-256
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`.
The upscaler repository revision was
`7b61c8edb895aaf25b248f064e9726ffdcc7ec46`.
The temporary profile is retained with local evidence, outside the PR source.

## Real decoder and transport checks

Pinned assets were checked by size/hash and loaded strictly. Real CUDA decoding
covered original LTX-2 (`taeltx_2`), LTX-2.3/2.5 (`taeltx2_3`), and the existing
flat Kijai H3 decoder. Synthetic latent inputs established tensor/layout and
encoding behavior; they do not establish full-model image quality or generation
overhead.

- `taeltx_2`: `128×2×8×8` → 9 RGB frames at 256×256.
- `taeltx2_3`: `128×2×8×8` → 9 RGB frames at 256×256; actual NVENC MP4 and WebP.
- Flat H3: `24×2×16×16` → 2 RGB frames at 256×256; actual NVENC MP4 and WebP.
- Wan 2.1/2.2 and Hunyuan 1/1.5 standard author weights loaded strictly on CPU
  and decoded their explicit channel/patch configurations to 5 frames. Decoder
  construction preserved the CPU RNG state.
- A real 64×64 NVENC failure followed by successful 256×256 encoding verified
  that an unsupported size does not permanently disable NVENC.

## Native installation

A real install of the pinned 2,704,756-byte `taef2_decoder.pth` used the current
official `download_file` through its `gen` context. Size and SHA-256 matched,
the previous context callback was restored, and no staging directory remained.
The native callback received 8 progress events; the preview callback received 7
events with both native `completed`/`speed` and UI `current`/`speed_bps` fields.
This used an isolated checkpoint directory and required no separate progress PR.

## Focused regression checks

The embedded runtime ran `unittest.defaultTestLoader.discover("tests", pattern=...)`
with `TextTestRunner` after prepending the repository to `sys.path`:

- `test_preview_subsystem.py`: 31 tests run, passing with 2 opt-in skips.
- With `WANGP_PREVIEW_REAL_DECODER_TEST=1`: 31 run, passing with 1 fixture skip;
  this strictly loads all added image/Wan/Hunyuan assets and checks CPU RNG.
- The separate `WANGP_PREVIEW_FIXTURE_TEST=1` loader fixture passed earlier.
- `test_preview_download_install.py`: 16 tests passed, including cancellation
  during validation preserving an existing destination.
- The final Qwen `NTCHW` metadata/guard alignment passed both existing image
  adapter/registry tests (`unittest discover ... -k image`). Tensor computation
  is unchanged from the measured Qwen matrix.
- Changed-source AST and default JSON parsing passed; `git diff --check` passed.
- Independent review found no remaining implementation defects.

No broad model test suite or performance benchmark was run.

## Compiled execution: blocked by the embedded runtime

A bounded Flux2 compile-mode baseline was attempted before a compiled TAE
comparison. Redirecting `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `TEMP`,
and `TMP` into the task workspace eliminated the initial shared-cache permission
error. The corrected baseline then failed after 21.185 s when Triton's C
compiler reported `cuda_utils.c:18: error: include file 'Python.h' not found`.
Its include path pointed at the embedded Python runtime's missing development
headers. The script process returned normally with a failed generation result;
this is not a passing compile test.

Compiled TAE was not attempted after the baseline failure. No Python headers,
compiler, package, or system configuration were installed or modified to mask
this prerequisite. Compiled comparisons require a working Python 3.11
development toolchain first; eager results above remain valid for their stated
fallback environment and profile 4.

## Local evidence

Reproducible scripts, full effective settings, JSON results, and preview/final
media are retained under the ignored `.validation/pr2107/` directory:
`ltx25_matrix.py/json`, `ltx25_cancel_followup.py/json`,
`ltx_fallback_metrics.py/json`, `flux2_matrix.py/json`,
`ltx_actual_timing.json`,
`zimage_matrix.json`, `qwen2512_matrix.json`, `h3_probe_matrix.py/json`,
`h3_conditioned_compare.json`, `h3_audio_validation.json`,
`flux2_compile_retry_matrix.json` and its stdout/stderr logs,
`decoder_cuda_validation.py/json`, `native_installer_smoke.py/json`, and
`nvenc_cache_check.py/json`.
The representative inspected images are `ltx25_tae_preview_10.png` and
`ltx25_tae_final.png`, plus `flux2_tae_preview_01/03/04.png` and
`flux2_tae_final.png`. Model/checkpoint assets and generated media are not part
of the source change. Z-Image and Qwen inspected snapshots use the
`zimage_tae_*` and `qwen2512_tae_*` filenames in the same directory.

## Remaining validation boundaries

The initial LTX matrix does not establish LTX audio equality, other offload
profiles, LTX source/mask conditioning, every sampler, or cancellation during
each decoder/encoder phase. Other registered families require their own complete
generation cases; source contracts and synthetic decoding are identified
separately above. Original LTX-2, Wan, and Hunyuan diffusion checkpoints were not
installed for this run. H3's Auto VAE initially attempted an unneeded INT8 VAE
download; it was stopped and the 1.56 GB partial file removed. The validation
profile then selected the already-installed FP8 VAE explicitly.
