# Processing

Processing is everything WanGP does around generation to prepare inputs, control what the model produces, or refine media after it exists.

Use this guide when you are deciding how to prepare a Control Video, build masks, use reference images, extend a video, or apply late postprocessing.

## At A Glance

| Stage | When It Happens | What It Does |
| --- | --- | --- |
| Preprocessing | Before generation | Converts source media into model-ready inputs such as pose, depth, masks, control frames, or cleaned references. |
| Generation-time processing | During generation | Uses temporal, spatial, mask, reference, and sliding-window controls to decide what the model should change or preserve. |
| Postprocessing | After generation | Improves or transforms completed images, videos, or audio with upscaling, remuxing, audio tools, or late media processing. |

## Before Generation: Preprocessing

Preprocessing changes what the model receives as conditioning. It does not produce the final media by itself; it prepares the inputs that guide generation.

Common preprocessing jobs:

- Resize and resample inputs to the model frame rate and output resolution.
- Extract pose, depth, canny, raw-frame, mask, or other control information from a Control Video.
- Prepare Video Masks for masked-area or non-masked-area workflows.
- Prepare reference images before inserting people, objects, landscapes, or exact frames.
- Build injected reference frames for temporal inpainting or spatial outpainting.
- Prepare source media for video continuation or long-video sliding windows.

### Control Videos

Control Video processing tells the model what structure, motion, or area to follow.

- **Keep Unchanged:** use the Control Video as the source structure without extracting a new annotation.
- **Transfer Human Motion:** extract pose or Open Pose style motion from a source video.
- **Depth / Canny / Similar Structure Controls:** extract scene structure while letting the prompt change appearance.
- **Inpainting:** mark areas that should be regenerated.
- **Outpainting:** add missing space around existing frames.

If a Video Mask is provided, the selected process can apply only to the masked area or only to the non-masked area. This lets you keep a subject while changing the background, replace one person among several people, or combine different effects inside and outside the mask.

WanGP previews processed Control Video and Video Mask thumbnails before generation. Use `--save-masks` when you need to inspect the full generated control media.

### Masks

Masks decide where a process applies.

- Black areas are usually preserved.
- White areas are usually processed.
- Some model families support colored masks to identify several people or objects independently.

For person or object replacement, a stable mask across frames is usually more important than a perfect mask on one frame.

Matanyone is useful for creating Video Masks:

1. Load the source video.
2. Select the face, person, object, or region in the first frame.
3. Validate the mask.
4. Generate the matting video.
5. Export the source video and mask back into the Control Video and Video Mask inputs.

Mask tips:

- Use negative point prompts to remove unwanted parts from a selection.
- Use sub masks when one selection is too hard to define cleanly.
- Use mask expansion or shrinking when the generated replacement needs more or less room than the original subject.

### Reference Images

Reference images can represent a subject, object, landscape, style source, or exact frame position. Their role should be clear in the prompt.

Use foreground cleanup for:

- People or objects that should be inserted into a generated scene.
- References where the original background should not influence the output.
- Character or object swaps where a clean foreground improves identity and placement.

Avoid foreground cleanup for:

- Landscape, room, or setting references.
- Background images that should define the environment.
- Injected frames that should appear as complete frames at a specific position.

If automatic cleanup is not precise enough, use the Image version of Matanyone to create a cleaner foreground mask.

## During Generation: Temporal And Spatial Processing

Generation-time processing changes what the model adds, keeps, or regenerates while it is producing the output.

### Temporal Processing

Temporal processing controls frames over time.

- **Temporal Outpainting:** adds frames before or after existing media. It is implicit when you continue a Source Video or provide a Control Video shorter than the requested output.
- **Temporal Inpainting:** fills missing frames between existing frames.
- **Injected Reference Frames:** places specific images at chosen frame positions so the model can fill the gaps between them.
- **Frames To Keep:** preserves selected Control Video frames and regenerates the others.

When a model supports positioned frames, frame positions are counted from the first frame of the active timeline. In sliding-window workflows, this usually means the first frame of the first window.

### Spatial Processing

Spatial processing controls image regions or canvas size.

- **Spatial Outpainting:** adds new content to the top, bottom, left, or right of existing frames.
- **Inpainting:** replaces masked image regions.
- **Mask Expansion / Shrinking:** adjusts the editable area when the replacement subject is larger or smaller than the original target.
- **Reference Image Placement:** uses injected frames or reference images to guide what appears in newly generated areas.

Spatial outpainting targets the resolution you request. If the requested resolution is the same as the source media, the original content may lose detail because canvas space is being reallocated. For outpainting, a higher output resolution is often useful.

### Long Videos Workflows

Sliding windows let WanGP generate longer videos by merging multiple generation windows.

The longer the video, the more likely quality is to degrade over time: artifacts can accumulate and character identity may progressively drift. The effect is usually less visible when the generated video mostly preserves unmodified control-video content. If the model supports end frames, a useful way to limit drift is to prepare several end-frame images in advance with an image-generation model and assign them to successive windows. These visual anchors steer later windows back toward the intended characters, composition, and appearance.

A reliable way to create related end frames is to generate one strong master image first, then use an image-editing model several times on that same master image to produce the planned keyframes. Ask each edit for the pose, camera angle, action, environment, or composition required at the end of its matching video window while preserving the original character identity and style. Always branch each edit from the master image rather than editing the previous keyframe in sequence, because chained edits can accumulate visual drift before video generation even begins.

Some uncommon models, including MiniMax H3 and VACE, can also re-inject reference images or frames between windows. Supplying suitable intermediate references gives the model recurring visual anchors and can restore character identity, appearance, or composition before drift becomes too noticeable. The available Reference Images or Injected Frames controls and their placement rules depend on the selected model.

There are two ways to build a multi-window video:

- **Plan all windows in one generation:** provide multiple prompts in advance, with one prompt for each window. This usually gives the best technical quality because WanGP keeps the windows in memory and performs the final merge and video encoding only once. The tradeoffs are higher RAM usage and the need to decide the sequence in advance.
- **Continue an existing video repeatedly:** use each completed video as the source for another continuation. This supports improvisation because the next prompt can be chosen after reviewing the latest result. However, every continuation reads an already encoded video and produces another encoded result, so repeated lossy re-encoding can slightly degrade quality at each stage.

Avoid making a window larger than the model can handle reliably. Larger windows take longer and require more VRAM, and every model has a practical or declared frame limit. On the other hand, a reasonably long window gives the model more room to preserve motion and character identity within that window. Choose the longest window that remains reliable for the selected model and available hardware rather than automatically using either the maximum or a very short window.

When sliding windows are enabled, positional arguments such as injected-frame positions or frames to keep are usually relative to the first frame of the first window. This lets you change the sliding-window size without recalculating all frame positions.

If you continue from a Source Video, the Source Video is treated as the first window. The overlap size controls how many frames from that source or previous window are used to condition the next generated segment.

How sliding windows work:

- Each window uses the matching time segment of the Control Video.
- Example: 0-4s control video -> first window, 4-8s -> second window.
- Overlap management helps smooth transitions between windows.

Generated frame count formula:

```text
Generated Frames = [Nb Windows - 1] x [Window Size - Overlap - Discard] + Window Size
```

Multi-line prompts can be used when the model supports assigning one prompt line per sliding window. If there are more windows than prompt lines, the last prompt line repeats.

Choose how window prompts are separated with **How to Process each Line of the Text Prompt**:

- **Each Line Will be used for a new Sliding Window of the same Video Generation:** every non-empty line starts a new window prompt.
- **Each Paragraph Separated by an Empty line will be used for a new Sliding Window of the same Video Generation:** every paragraph separated by an empty line starts a new window prompt. Use this choice when an individual window prompt contains line breaks of its own.

Each window prompt may begin with optional `[/...]` commands. Use `[/overlap=9]` to select a different overlap for that transition, or `[/overlap]` to restore the model's default overlap. On text-to-video-capable models, `[/overlap=0]` creates a hard cut by preventing the preceding window from conditioning the new one; `[/new_shot]` is its more explicit alias. Different window lengths can be requested with `[/duration=121]`, `[/duration=5s]`, or `[/duration=20%]`. When duration commands are present, they define the window schedule and predicted final length instead of treating the UI's Number of Frames as a strict final cap; windows without an explicit duration consume the remaining requested frames, up to the configured Sliding Window Size. Commands can be combined, for example `[/duration=4s,/new_shot]`.

Sliding-window settings:

- **Window Size:** choose a duration that fits the motion and model quality.
- **Overlap Frames:** use enough overlap for continuity, but not so much that blur propagates.
- **Discard Last Frames:** remove weak final frames from each window when a model tends to blur at the end.
- **Add Overlapped Noise:** may help some workflows, but can reduce quality in others.

## After Generation: Postprocessing

Postprocessing is applied after media exists. It can run immediately after generation or later from the Media Info / Late Post Processing area.

Use postprocessing when the content is good but the final media needs a technical or finishing pass.

Common postprocessing jobs:

- Spatial upsampling for images or videos.
- Video frame interpolation or temporal upsampling.
- Audio postprocessing, voice replacement, or soundtrack remuxing.
- Cropping, resizing, or importing existing media for a later workflow.
- Applying a system postprocessing model through MediaFlow.

Late postprocessing is useful when you want to improve existing media without regenerating the original scene.

## Checklists

Before generation:

- Preview processed Control Video and Video Mask thumbnails.
- Confirm masks apply to the intended region.
- Check that reference images match the role described in the prompt.
- Use `--save-masks` when the preview is not enough.

Before a long sliding-window run:

- Test a short window first.
- Confirm frame positions are measured from the intended timeline start.
- Pick enough overlap for continuity.
- Avoid excessive overlap when blur propagates.

After generation:

- Check that outputs are visually usable, not only valid files.
- Use late postprocessing when resolution, audio, or motion needs refinement.
- Keep the original generation output when trying multiple postprocessing passes.

## Troubleshooting

If windows are inconsistent:

- Increase overlap frames.
- Keep prompts consistent across windows.
- Reduce discard frames if too much useful content is lost.
- Add overlapped noise only if it helps the specific workflow.

If results are blurry:

- Reduce excessive overlap.
- Increase discarded end frames for models that blur at window ends.
- Use higher-quality reference images or control videos.
- Verify that preprocessing did not resize the useful region too aggressively.

If masks do not behave as expected:

- Inspect the preview thumbnails.
- Save masks and control videos with `--save-masks`.
- Rebuild the mask with cleaner selections or sub masks.
- Expand or shrink the mask when the replacement subject needs different space.
