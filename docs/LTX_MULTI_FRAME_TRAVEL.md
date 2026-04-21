# LTX Multi-Frame Travel: Current Implementation and LTX-Desktop Comparison

This document explains what "multi-frame" means in the LTX travel path in this repository, and compares it to the implementation in `Lightricks/LTX-Desktop` at commit `4a8e74b9ae47795fd43983e437d02ef0930a0b2f` (inspected on 2026-03-18).

## Update (2026-04-21): `ltx_anchor` — native anchor path added

A new `travel_guidance.kind = "ltx_anchor"` has been added as a sibling to `ltx_control` and `ltx_hybrid`. This is the path to use when a travel segment's only guidance is anchor images (first frame, optional middle frames, last frame) without any pose/depth/canny control video.

- Opt-in only. `ltx_control` remains the default for backwards compatibility; nothing auto-flips.
- Routes anchors through the native LTX `VideoConditionByKeyframeIndex` primitive (`Wan2GP/models/ltx2/ltx_core/conditioning/types/keyframe_cond.py`), delivered via `image_refs_paths` + `frames_positions` + `image_refs_strengths` and `video_prompt_type = "KFI"`. The per-segment pixel cross-fade bake is skipped entirely.
- Gated to LTX-2 distilled models only (same gate as `ltx_control`/`ltx_hybrid`).
- Anchors-only: rejects `videos`, `mode`, `audio`, `control_strength`, `canny_intensity`, `depth_contrast`. Requires ≥2 anchors.
- Default strength policy, applied at parse time when no caller strengths are provided, sorted globally by `frame_position`:
  - **2 anchors** (first + last): `[1.0, 1.0]` — both hard. Matches the LTX community "Inplace / Keyframer" FF+LF pattern.
  - **3+ anchors** (first + middle(s) + last): `[1.0, 0.0, ..., 0.0, 1.0]` — hard on start/end, `0.0` on middles so the text prompt drives the transition. Matches the "AddGuide / Sequencer" pattern for middle frames; prevents the known Inplace-style middle-frame stutter.
- All-or-nothing override. If *any* anchor in the payload specifies an explicit `strength`, the default rewrite is skipped entirely and all payload-literal values pass through. This is the power-user escape hatch for non-default strength patterns (e.g., `0.8`/`0.9` on first/last for visually "weird" pairs, or a non-zero middle for mild adherence).
- Prompt convention. Authoring convention per LTX community consensus: describe the first frame, the transition, and the last frame in one prompt (e.g., *"Woman standing in a field, turns and walks toward camera, now close-up on her face"*). Hard anchors surface bad FF/LF pairs as jump cuts; explicit transition text prevents that.

Known gaps (out of scope for `ltx_anchor`, left as follow-ups):

- Upscaler-stage re-injection of anchors (LTX 2.3 is known to lose anchor fidelity through the upscaler pass).
- Identity drift across anchors. Layer in an ID-LoRA / IP-adapter if needed.
- Real-GPU empirical validation before promoting `ltx_anchor` to the default kind.

## Short Version

LTX travel now supports two parallel paths:

- **`kind = "ltx_control"`** (default, unchanged): segmented travel + per-segment pixel guide clip + `video_prompt_type = "VG"`. Best for control-video guidance (pose/depth/canny via IC-LoRA) where the guide video is the real signal. Start/end anchors are baked into the pixel timeline with cross-fades.
- **`kind = "ltx_anchor"`** (new, opt-in): native LTX keyframe conditioning. Anchors flow through `image_refs_paths` / `frames_positions` / `image_refs_strengths` to the KFI path and are injected as `VideoConditionByKeyframeIndex` tokens with a position-encoded frame index. No pixel bake.

What we still do **not** do for LTX:

- SVI-style continuation
- `video_source` prefix-window continuation
- Arbitrary per-frame image conditioning for *non-anchor* frames

So the best mental model is now:

- **`ltx_control`** = segmented travel + per-segment control video (good for pose/depth/canny)
- **`ltx_anchor`** = segmented travel + native LTX keyframe anchors (good for FF/MF/LF without structure guidance)
- **LTX-Desktop** = native LTX generation primitives, retake, and IC-LoRA workflows

## What Our Code Does

### 1. Travel guidance is parsed as an orchestrator-level contract

`travel_guidance` is parsed into a `TravelGuidanceConfig` with kinds `none`, `vace`, `ltx_control`, and `uni3c`.

Relevant code:

- `source/core/params/travel_guidance.py`
  - `from_payload(...)`
  - `needs_ic_lora()`
  - `get_preprocessor_type()`
  - `to_segment_payload(...)`

Important implication:

- `ltx_control` is treated as a **guidance video mode**, not as a native LTX multi-image-anchor mode.

### 2. The orchestrator computes segment overlap against a stitched timeline

The orchestrator computes stitched segment offsets, then checks whether each segment overlaps any configured guidance video range.

Relevant code:

- `source/task_handlers/travel/orchestrator.py`
  - `_calculate_segment_stitched_offsets(...)`
  - `_segment_has_travel_guidance_overlap(...)`
  - `_build_segment_travel_guidance_payload(...)`

Important implication:

- this is timeline-aware
- but the result is still a **segment-local control payload**
- segments with no overlap explicitly receive `{"kind": "none"}`

### 3. LTX segments do not get VACE-style context-frame inflation

For subsequent segments, context frames are only inflated for sequential `vace` travel.

Relevant code:

- `source/task_handlers/travel/orchestrator.py`
  - segment frame target calculation near `travel_mode == "vace"`

Important implication:

- LTX travel is not treated as overlap-based continuation
- segment length for LTX remains the base segment length after quantization

### 4. The guide builder materializes a segment-local control clip

If `travel_guidance.kind == "ltx_control"`, the guide builder bypasses the normal guide-video creation path.

Relevant code:

- `source/task_handlers/travel/guide_builder.py`
  - `_build_local_travel_guidance_video(...)`
  - `_create_ltx_control_guide_video(...)`
  - `create_guide_video(...)`

The actual behavior is:

1. If the orchestrator already produced a shared guidance clip, use that plus `_frame_offset`.
2. Otherwise, build a local composite guidance clip for the current segment only.
3. Extract exactly `total_frames_for_segment` frames from that clip.
4. Re-encode those frames as the control video passed to the segment.

Important implication:

- this is **multi-frame control video slicing**
- it is **not** first/middle/last anchor injection
- it is **not** latent continuation from a previous LTX segment

### 5. Segment processing forces `VG` for LTX control

When `travel_guidance.is_ltx_control` is present, the segment processor forces `video_prompt_type = "VG"`.

Relevant code:

- `source/task_handlers/travel/segment_processor.py`
  - `create_video_prompt_type(...)`

Important implication:

- we are telling the LTX path to consume a guide/control video
- we are not mapping segment anchors into native image frame slots

### 6. IC-LoRA is auto-injected for pose/depth/canny

The orchestrator auto-injects the union IC-LoRA for LTX control modes that need it.

Relevant code:

- `source/task_handlers/travel/orchestrator.py`
  - `_auto_inject_travel_guidance_lora(...)`

Important implication:

- our LTX travel control path relies on a control-video + LoRA pattern
- that is much closer to an IC/control workflow than to an anchor-window continuation workflow

### 7. The only true multi-frame anchor continuation path in this repo is SVI, and LTX does not use it

The SVI path uses:

- `image_refs_paths` for anchor image(s)
- `video_source` for predecessor prefix frames
- overlap accounting in `video_length`

Relevant code:

- `source/task_handlers/travel/orchestrator.py`
  - disables `use_svi` when `model_name` is `ltx2`
- `source/task_handlers/tasks/task_registry.py`
  - `_apply_svi_config(...)`

Important implication:

- this is the only path here that behaves like true multi-frame continuation
- it is Wan-specific
- LTX does not use it

## What LTX-Desktop Does

## 1. The standard local LTX video generation path supports native image conditioning inputs

`LTX-Desktop` has a shared `ImageConditioningInput(path, frame_idx, strength)` type and passes those entries down into the LTX pipeline.

Relevant code:

- `backend/api_types.py`
  - `ImageConditioningInput`
- `backend/services/ltx_pipeline_common.py`
  - `DistilledNativePipeline.__call__(...)`
  - `image_conditionings_by_replacing_latent(...)`

Important implication:

- the underlying LTX pipeline can condition at arbitrary frame indices
- this is the native primitive closest to "first/middle/last anchors"

### 2. But the regular desktop I2V UI only uses one image at frame 0

The normal video generation path in `LTX-Desktop` converts a single uploaded image into:

- `ImageConditioningInput(path=..., frame_idx=0, strength=1.0)`

Relevant code:

- `backend/handlers/video_generation_handler.py`

Important implication:

- the base product flow is still a single-image-start conditioning flow
- even though the backend primitive is more general

### 3. Duration is converted to `8k+1` frame counts

`LTX-Desktop` computes:

- `num_frames = ((duration * fps) // 8) * 8 + 1`

Relevant code:

- `backend/handlers/video_generation_handler.py`
  - `_compute_num_frames(...)`

Important implication:

- `4s @ 24fps` becomes `97` frames
- that is a frame-quantization rule, not a statement about effective new motion under anchors or continuation

### 4. Retake is a different multi-frame workflow

`LTX-Desktop` includes a retake flow that:

- loads the full source video
- VAE-encodes the full source video
- uses `TemporalRegionMask(start_time, end_time, fps)` to restrict where regeneration happens

Relevant code:

- `backend/handlers/retake_handler.py`
- `backend/services/retake_pipeline/ltx_retake_pipeline.py`

Important implication:

- this is a **region-editing** workflow
- it is not a segment stitcher
- it is not first/middle/last anchor sequencing either
- but it is genuinely multi-frame because the full source video is the conditioning substrate

### 5. IC-LoRA in LTX-Desktop is closer to our `ltx_control` path than its basic I2V path is

`LTX-Desktop`'s IC-LoRA flow:

- preprocesses every input frame into a full control video
- caches that control video
- optionally accepts `req.images` with frame-specific `frame` indices
- calls the LTX IC-LoRA pipeline with both `images` and `video_conditioning`

Relevant code:

- `backend/handlers/ic_lora_handler.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`

Important implication:

- this is the closest external analog to our `ltx_control` travel implementation
- both systems rely on full control-video conditioning
- `LTX-Desktop` also exposes the native image-frame conditioning primitive explicitly

## Sense-Check: How Our Implementation Compares

### Where our implementation matches the spirit of LTX-Desktop

1. We are using a real multi-frame conditioning object for LTX travel.
   In our case that object is a per-segment control clip.

2. We are closer to `LTX-Desktop`'s IC-LoRA/control-video workflows than to its simple I2V workflow.

3. We already respect the model's `8N+1` temporal quantization rule through segment frame quantization.

### Where our implementation does not match LTX-Desktop's native capabilities

1. We do not expose or use native LTX per-frame image conditioning in travel.
   We do not currently translate travel anchors into `ImageConditioningInput(frame_idx=...)`.

2. We do not have an LTX-native equivalent of a first/middle/last anchor workflow.
   Our "multi-frame" concept is a control video, not multiple anchor images.

3. We do not have an LTX retake-style partial video regeneration path in travel orchestration.
   Our orchestration model is segment generation + stitching.

4. We do not have LTX continuation via predecessor latent/prefix video.
   That machinery exists only in the Wan SVI path.

## Conclusion

If the question is:

- "Do we have multi-frame conditioning for LTX?"

then the answer is:

- **yes**, but in the form of per-segment control clips

If the question is:

- "Do we implement LTX multi-frame anchors the way LTX-Desktop or native LTX primitives allow?"

then the answer is:

- **no**

The biggest gap is this:

- `LTX-Desktop`'s backend has a native `images + frame_idx` conditioning primitive.
- our travel LTX path does not currently map travel anchors into that primitive.

So if we want true first/middle/last LTX anchors in travel, the missing work is not in stitching. The missing work is a new segment payload layer that expresses:

- multiple anchor images
- target frame indices
- strengths

and passes those through to the LTX generation boundary instead of collapsing everything into a control video.

## Recommended Next Step

If we want to align more closely with LTX-native behavior, the next design to prototype is:

1. keep the current `ltx_control` control-video path for pose/depth/canny/video guidance
2. add a separate LTX-native anchor mode for travel, backed by:
   - `images=[{path, frame_idx, strength}, ...]`
3. use that mode for first/middle/last or sparse anchor workflows
4. keep SVI out of the design, because it is Wan-specific and conceptually different

That would let us support both:

- control-video-driven LTX travel
- anchor-image-driven LTX travel

without pretending they are the same thing.
