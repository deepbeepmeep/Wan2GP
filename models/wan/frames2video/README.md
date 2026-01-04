# 📘 Frames‑to‑Video (Wan2.2 I2V Morph)

### Wan2GP Model Backend — Module Overview

This module implements the **Frames‑to‑Video morphing backend** for **Wan2GP**, built on top of the **Wan 2.2 Image‑to‑Video (I2V)** model. It provides a clean, unified interface for interpolating between two images — optionally with middle frames and timestamps — matching the behavior of the original **morphicfilms/frames-to-video** demo.

Wan2GP handles:

- model loading
- LoRA application
- quantization
- VRAM/offload
- latent decoding
- video writing

This module focuses **only** on the algorithmic heart of the morph.

## 🚀 What This Module Does

- Loads the **Wan2.2 I2V 14B** model using Wan2GP’s unified loader
- Applies the **high‑noise interpolation LoRA** used in the original demo
- Accepts:
  - start frame
  - end frame
  - optional middle frames
  - optional timestamps
- Passes these into the Wan2.2 I2V pipeline via `self.generate()`
- Returns a latent video tensor **[C, F, H, W] in [-1, 1]**
- Leaves decoding + video writing to Wan2GP

There is **no custom latent math or scheduler logic** here — the original Frames‑to‑Video repo does not implement any. All interpolation behavior is handled inside the Wan2.2 I2V model.

## 🧠 Architecture

Code

```
models/
└── wan/
    └── frames2video/
        ├── core.py      ← main backend logic
        ├── README.md    ← this file
```

### `core.py` Responsibilities

- tensor preparation
- device placement
- argument forwarding
- calling `self.generate()` with the correct parameters

This mirrors the behavior of `generate.py` from the original Frames‑to‑Video repo.

## 📦 Model Definition (`defaults/frames2video.json`)

Defines:

- model URLs (Wan2.2 I2V 14B)
- quantized variants
- LoRA URL
- LoRA multiplier
- default parameters (frame count, steps, solver, shift, etc.)

Wan2GP automatically:

- downloads the model
- selects the correct quantized variant
- loads the LoRA
- merges the LoRA into the UNet
- initializes the I2V pipeline

No manual loading is required inside `core.py`.

## 🎛 Runtime Parameters

These parameters are passed directly to `self.generate()`:

- `frame_num`
- `sampling_steps`
- `sample_solver`
- `shift`
- `guide_scale`
- `seed`
- `offload_model`
- `max_area`
- `middle_images`
- `middle_images_timestamps`

Defaults match the original Frames‑to‑Video demo.

## 🔍 Why No Vendored Scheduler or Latent Logic?

The original `morphicfilms/frames-to-video` repo does **not** implement:

- custom schedulers
- custom latent scaling
- custom VAE decode logic
- custom interpolation math

It simply calls the Wan2.2 I2V model’s `.generate()` method.

Therefore:

- Wan2GP’s native Wan2.2 I2V implementation is the source of truth
- No vendored code is needed
- Reproducibility comes from matching parameters, not patching internals

## 🧪 Testing

To validate the integration:

1. Provide a start and end frame
2. Use default settings (81 frames, 40 steps, shift 5.0)
3. Compare output to the original Frames‑to‑Video demo

Minor differences may occur due to:

- diffusers version
- transformers version
- numpy version

These are expected.

## 📄 Summary

This module provides a clean, maintainable, Wan2GP‑native implementation of Frames‑to‑Video:

- No duplicated code
- No vendored logic
- No fragile patches
- Full compatibility with Wan2GP’s model loader
- Full support for interpolation, middle frames, and timestamps
- Reproducible defaults matching the original demo

It is the simplest and most robust way to integrate Frames‑to‑Video into Wan2GP.

## 🧩 Architecture Diagram

Code

```
┌──────────────────────────────────────────────────────────────┐
│                        Wan2GP Engine                          │
│  (model loader, LoRAs, quantization, VRAM mgmt, video writer) │
└───────────────▲──────────────────────────────────────────────┘
                │ calls run_frames2video()
┌───────────────┴──────────────────────────────────────────────┐
│         models/wan/frames2video/core.py                       │
│  • Receives start/end frames + optional middle frames         │
│  • Normalizes to [-1, 1] tensors                              │
│  • Moves tensors to device                                    │
│  • Forwards parameters to self.generate()                     │
│  • Returns latent video tensor [C, F, H, W]                   │
└───────────────▲──────────────────────────────────────────────┘
                │ calls
┌───────────────┴──────────────────────────────────────────────┐
│                Wan2.2 I2V Pipeline (Wan2GP)                   │
│  • Loads Wan2.2 I2V 14B model                                 │
│  • Applies high-noise interpolation LoRA                      │
│  • Handles CLIP, VAE, UNet, schedulers                        │
│  • Performs latent interpolation internally                   │
│  • Generates latent video frames                              │
└───────────────▲──────────────────────────────────────────────┘
                │ returns latent video
┌───────────────┴──────────────────────────────────────────────┐
│                    Wan2GP Video Writer                        │
│  • Decodes latents → RGB frames                               │
│  • Writes MP4/WebM output                                     │
│  • Handles audio merging (if provided)                        │
└──────────────────────────────────────────────────────────────┘
```

## 🛠 Troubleshooting

### 🔹 Output differs from the original demo

Expected due to newer versions of:

- diffusers
- transformers
- numpy

These affect:

- latent scaling
- scheduler defaults
- VAE decode behavior
- CLIP normalization

Interpolation remains correct.

### 🔹 Video too short or too long

Ensure:

- `frame_num`
- `video_length`

match. Wan2GP uses `frame_num` internally; `video_length` is UI‑only.

### 🔹 Middle frames not used

Check:

- `middle_images` is a list
- `middle_images_timestamps` is a list of floats (0–1)
- lengths match

Example:

Code

```
"middle_images_timestamps": [0.25, 0.75]
```

### 🔹 VRAM spikes / OOM

Set:

Code

```
"offload_model": true
```

### 🔹 Interpolation too linear or too sharp

Adjust:

- `shift`
- `guide_scale`
- `sampling_steps`

Defaults:

Code

```
shift = 5.0
guide_scale = 5.0
sampling_steps = 40
```

### 🔹 Output resolution incorrect

Controlled by `max_area`:

Code

```
max_area = width * height
```

Examples:

- 1280×720 → OK under 1024×1024
- 1920×1080 → too large unless max_area increased

### 🔹 Seed has little effect

Normal for pure morphing. Increase:

- `guide_scale`
- `shift`

to introduce more stochasticity.

### 🔹 LoRA not applied

Increase multiplier:

Code

```
loras_multipliers: [1.0 → 1.5]
```

### 🔹 Black or blank video

Usually caused by:

- corrupted input images
- missing `image_start` or `image_end`
- tensors not normalized