# Kijai's SVI + End Frame Implementation

This document describes how [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) implements SVI (Stable Video Infinity) with end frame support, and how our Wan2GP implementation aligns with it.

## Overview

Kijai's implementation separates two distinct concepts:

| Node | Purpose | End Frame Support |
|------|---------|-------------------|
| `WanVideoSVIProEmbeds` | SVI continuation (chaining segments via latent motion) | ❌ No |
| `WanVideoImageToVideoEncode` | I2V encoding with start/end frames | ✅ Yes |

**For SVI + end frames, kijai uses `WanVideoImageToVideoEncode`** (the I2V path) while loading SVI LoRAs.

## Kijai's WanVideoImageToVideoEncode (nodes.py lines 969-1159)

### Step 1: Build Pixel Tensor

```python
# Build [start | zeros | end] in PIXEL space
if start_image is not None and end_image is not None:
    if fun_or_fl2v_model:
        zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, ...)
    else:
        zero_frames = torch.zeros(3, num_frames-1, H, W, ...)
    concatenated = torch.cat([resized_start_image, zero_frames, resized_end_image], dim=1)
```

**Key:** Empty frames are **zero pixels**, not zero latents.

**Nuance (multi-frame refs):** In the `not fun_or_fl2v_model` branch, kijai uses `zero_frames = torch.zeros(..., num_frames-1, ...)` (it does **not** subtract `start_image.shape[0]` / `end_image.shape[0]`). In practice this path assumes `start_image` and `end_image` are typically **single-frame** (so `1 + (num_frames-1) + 1 = num_frames+1`, matching `base_frames`). If you pass multi-frame start/end batches, the concatenated pixel timeline becomes `start_T + (num_frames-1) + end_T`, and will no longer match `base_frames` unless `start_T=end_T=1`.

### Step 2: Optional Empty Frame Padding (empty_frame_pad_image)

```python
if empty_frame_pad_image is not None:
    # Resize pad image to match dimensions
    pad_img = (pad_img.movedim(-1, 0) * 2 - 1).to(device, dtype=vae.dtype)
    
    # Expand to cover all frames if needed
    if num_pad_frames < num_target_frames:
        pad_img = torch.cat([pad_img, pad_img[:, -1:].expand(-1, num_target_frames - num_pad_frames, -1, -1)], dim=1)
    
    # Replace ONLY the empty frames (those with mask < 0.5)
    frame_is_empty = (pixel_mask[0].mean(dim=(-2, -1)) < 0.5)[:concatenated.shape[1]].clone()
    if start_image is not None:
        frame_is_empty[:start_image.shape[0]] = False  # Don't replace start
    if end_image is not None:
        frame_is_empty[-end_image.shape[0]:] = False   # Don't replace end
    
    concatenated[:, frame_is_empty] = pad_img[:, frame_is_empty]
```

**Key:** The mask still says "generate here" for padded frames—the padding just helps VAE temporal coherence.

### Step 3: Build Mask in Pixel-Frame Space

```python
# Mask: 1 = known (preserve), 0 = unknown (generate)
mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
if start_image is not None:
    mask[:, 0:start_image.shape[0]] = 1   # First N frames known
if end_image is not None:
    mask[:, -end_image.shape[0]:] = 1     # Last M frames known
```

### Step 4: Expand Mask for Latent Subframes

```python
# First frame gets expanded to 4 subframes (VAE temporal stride)
start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)

# End frame also gets expanded to 4 subframes (when not fun_or_fl2v_model)
if end_image is not None and not fun_or_fl2v_model:
    end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
    mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
else:
    mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

# Reshape: [1, T*4, H, W] -> [4, T, H, W]
mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
mask = mask.movedim(1, 2)[0]  # [4, T, H, W]
```

### Step 5: Single VAE Encode

```python
# CRITICAL: Single encode of entire concatenated tensor with end_= flag
y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model), tiled=tiled_vae)[0]
```

**Key:** The `end_=True` flag switches kijai’s VAE into a **different encode/decode mode** (internally `encode_2` / `decode_2`; and in tiled mode it also increases the latent time length by +1). This is only enabled when `end_image is not None and not fun_or_fl2v_model`.

## Kijai's WanVideoSVIProEmbeds (nodes.py lines 913-966)

This node is for **continuation only** (chaining segments), NOT for end frames:

```python
def add(self, anchor_samples, num_frames, prev_samples=None, motion_latent_count=1):
    anchor_latent = anchor_samples["samples"][0].clone()
    
    if prev_samples is None or motion_latent_count == 0:
        # No continuation: [anchor | zeros]
        padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
        y = torch.concat([anchor_latent, padding], dim=1)
    else:
        # Continuation: [anchor | motion_latents_from_prev | zeros]
        motion_latent = prev_latent[:, -motion_latent_count:]
        padding = torch.zeros(C, padding_size, H, W, dtype=dtype, device=device)
        y = torch.concat([anchor_latent, motion_latent, padding], dim=1)
    
    # Mask: only first frame known
    msk = torch.ones(1, num_frames, H, W, device=device, dtype=dtype)
    msk[:, 1:] = 0  # Everything except first frame = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
```

**Key differences from I2V encode:**
- Works in **latent space** (anchor_latent already encoded)
- Uses **zero latents** for padding
- **No end frame support**
- Mask only marks first frame as known

## Our Wan2GP Implementation (any2video.py)

We aligned our SVI + end frame path with kijai's `WanVideoImageToVideoEncode`:

### When `svi_pro=True` AND `any_end_frame=True`:

```python
# Step 1: Determine start frames (single anchor OR prefix context for continuation)
if prefix_video is not None and prefix_video.shape[1] >= (5 + overlap_size):
    start_pixels = prefix_video[:, -prefix_context_count:]  # Continuation
    svi_start_frame_count = prefix_context_count
else:
    start_pixels = image_ref  # First segment
    svi_start_frame_count = 1

# Step 2: Build empty frames (zeros by default, like kijai)
empty_pixels = image_ref.new_zeros((C, empty_frame_count, H, W))

# Optional: pad with anchor (equivalent to kijai's empty_frame_pad_image)
if model_def.get("svi_pad_empty_frames_with_anchor", False):
    empty_pixels = image_ref.expand(-1, empty_frame_count, -1, -1).clone()

# Step 3: Concatenate in pixel space: [start | empty | end]
concatenated = torch.cat([start_pixels, empty_pixels, img_end_frame], dim=1)

# Step 4: Single VAE encode with end frame mode
lat_y = self.vae.encode([concatenated], VAE_tile_size, any_end_frame=vae_end_frame_mode)[0]

# Step 5: Build mask (pixel-frame space, then expand)
msk = torch.zeros(1, frame_num, lat_h, lat_w, ...)
msk[:, :svi_start_frame_count] = 1  # Start frames known
msk[:, -1:] = 1                      # End frame known

# Expand first and last to 4 subframes
start_mask_repeated = torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1)
end_mask_repeated = torch.repeat_interleave(msk[:, -1:], repeats=4, dim=1)
msk = torch.cat([start_mask_repeated, msk[:, 1:-1], end_mask_repeated], dim=1)
```

## Key Differences: Our Implementation vs Kijai

| Aspect | Kijai | Our Wan2GP |
|--------|-------|------------|
| Empty frames | Zeros by default | Zeros by default |
| Optional padding | `empty_frame_pad_image` param | `svi_pad_empty_frames_with_anchor` model config |
| Start frames | Single start_image | Single anchor OR prefix context frames |
| VAE encode | Single encode with `end_=` | Single encode with `any_end_frame=` |
| Mask expansion | First 4x + last 4x | First 4x + last 4x (matching kijai) |

## Configuration

### Model Config Options

```json
{
  "svi2pro": true,                           // Enable SVI Pro encoding path
  "svi_pad_empty_frames_with_anchor": false  // Optional: pad empties with anchor (kijai's empty_frame_pad_image equivalent)
}
```

### Example Models

- `wan_2_2_i2v_lightning_svi_3_3.json` - Has `"svi2pro": true`
- `wan_2_2_i2v_lightning_baseline_2_2_2.json` - Does NOT have svi2pro

## Debug Logging

Enable SVI debug logs with `--debug` flag:

```bash
uv run --python 3.10 python worker.py --debug --supabase-url ... --supabase-access-token ...
```

Look for `[SVI_DEBUG]` prefix in logs.

## Dependency Drift Note

`Wan2GP/` is now a git submodule pinned to the maintained fork. If kijai-side dependency requirements change, do not edit the worker runtime ad hoc inside the pod. Mirror the requirement change back into the root worker project metadata, regenerate `uv.lock`, and then let the normal `uv sync --locked --python 3.10` launch path pick it up. Any temporary fork drift is acceptable only until that lock refresh lands.

If a migration rollback is needed while debugging dependency changes, keep the same repo-level rule: do not add a runtime pip fallback on the uv branch. Restore the timestamped `*.pre-uv-*` environment backup for a first-migration failure, or revert the rollout commits to return to the older `requirements.txt` bootstrap.

## Common Issues

### svi2pro Not Taking Effect

**Symptom:** SVI path not taken even when `svi2pro=True` is passed in task params.

**Cause:** The `svi2pro` flag must be in the **model_def**, not just generation params. It's read at `model_def.get("svi2pro", False)`.

**Solution:** Any of:
1. Use a model that has `"svi2pro": true` in its JSON config (e.g., `wan_2_2_i2v_lightning_svi_3_3`)
2. Pass `svi2pro=True` in task params - **now automatically patched into model_def at runtime**

### How Runtime svi2pro Patching Works

When a task includes `svi2pro=True`, `headless_model_management.py` will:

1. **Before generation:** Patch `wgp.models_def[model]['svi2pro'] = True`
2. **After generation:** Restore the original value (prevents cross-task contamination)

```
[SVI2PRO] Patched wgp.models_def['wan_2_2_i2v_lightning_baseline_2_2_2']['svi2pro'] = True (was: None)
... generation happens with SVI enabled ...
[SVI2PRO] Restored wgp.models_def['wan_2_2_i2v_lightning_baseline_2_2_2']['svi2pro'] to None
```

### No [SVI_DEBUG] Logs

**Cause:** `offload.default_verboseLevel` not set to 2+

**Solution:** Ensure `--debug` flag is passed to worker.py
