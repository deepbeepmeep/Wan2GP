VACE_INFOS = """
# VACE ControlNet Guide

VACE is a ControlNet-style Wan model for Video-to-Video and Reference-to-Video generation. It can inject images into output videos, animate characters, perform masked edits, continue existing videos, transfer motion, and preserve scene structure while changing style or content.

## Overview

With VACE you can:

- Inject people or objects into scenes.
- Animate characters from a motion or pose source.
- Perform video inpainting and outpainting.
- Continue existing videos.
- Transfer motion from one video to another.
- Change the style of scenes while preserving their structure.

## Getting Started

### Model Selection

1. Select a VACE model such as `Vace 1.3B`, `Vace 14B`, or a derived VACE model.
2. VACE works best with short windows. For long videos, use sliding windows from the Processing guide.

Derived VACE models, such as VACE FusioniX, can be combined with accelerator LoRAs such as Causvid when the model supports them.

### Control Video

The Control Video is the source material that contains the visual instructions. VACE expects hints in the Control Video about the intended operation: replacing an area, converting an Open Pose wireframe into human motion, colorizing an area, transferring depth, and similar controls.

For example, an area colored grey, value 127, is treated as an inpainting region and replaced by content from the prompt and/or reference images. If the Control Video contains an Open Pose wireframe, VACE can turn that pose into a generated person based on the prompt and reference images.

You can build a VACE-formatted Control Video with the official VACE annotator tools, or let WanGP generate it from your normal source media.

WanGP needs:

- **Control Video:** the original source video, not already processed by an external annotator.
- **Control Video Process:** the process to apply, such as `Transfer Human Motion`, `Depth`, `Inpainting`, or `Keep Unchanged`.
- **Area Processed:** whether the process applies to the whole video, the masked area, or the non-masked area.

If you target an area, provide a Video Mask. You can create masks with the embedded Mask Generator / Matanyone workflow.

WanGP previews the processed Control Video and Video Mask in the Generation Preview area. Launch with `--save-masks` when you need to inspect the complete generated control media.

### Reference Images

Reference Images let you inject people, objects, landscapes, or exact frames into a video. You can also force images to appear at specific frame numbers.

Describe injected people or objects explicitly in the prompt so VACE can connect the references to the generated video.

## VACE Control Video And Mask Format

The Video Mask decides where the Control Video is interpreted.

- Black mask areas are kept as-is.
- White mask areas are processed.
- If there is no Video Mask, the selected VACE process applies to the whole video.

The content of the Control Video decides what process VACE sees in the selected area:

- Grey, value 127, means the area is replaced by new content from the prompt or image references.
- Open Pose wireframes are converted into an animated person.
- Multiple shades of grey can represent image depth, letting VACE generate new content at similar depth.

There are more VACE representations. Refer to the official VACE documentation for the full mapping.

## Example 1: Replace A Person While Keeping The Background

1. Select `Control Video Process` = `Transfer human pose` and `Area processed` = `Masked area`.
2. In Mask Generator, load the source video and mask the person to replace.
3. Click `Export to Control Video Input and Video Mask Input`.
4. In VACE, set Reference Image to `Inject Landscapes / People / Objects` and upload one or more pictures of the new person.
5. Generate.

This also works with several people if you mask several targets. Use `Expand / Shrink Mask` when the replacement subject needs more or less room.

## Example 2: Change The Background Behind Characters

1. Select `Control Video Process` = `Inpainting` and `Area processed` = `Non Masked area`.
2. In Mask Generator, create a mask for the people you want to keep.
3. Click `Export to Control Video Input and Video Mask Input`.
4. Generate.

If you use `Depth` instead of `Inpainting`, the new background can keep geometry closer to the original control video.

## Example 3: Outpaint A Video And Inject A Character

1. Select `Control Video Process` = `Keep Unchanged`.
2. In `Control Video Outpainting in Percentage`, enter the desired expansion amount, such as `40` for Left.
3. In Reference Image, select `Inject Landscapes / People / Objects` and upload a person reference.
4. Prompt the new action, such as `a person is coming from the left`.
5. Generate.

## Creating Face / Object Replacement Masks

Matanyone can generate the Video Mask that is combined with the Control Video.

1. Load your video in Matanyone.
2. Click the face or object in the first frame.
3. Validate the mask with `Set Mask`.
4. Generate a copy of the control video and a new mask video with `Generate Video Matting`.
5. Export to VACE with `Export to Control Video Input and Video Mask Input`.

Advanced tips:

- Use negative point prompts to remove unwanted areas from the mask.
- Use sub masks when one selection is difficult to define cleanly.

## Recommended Settings

- **Skip Layer Guidance:** turn on with the default configuration for better results when using original CFG-based VACE models. This is not useful with FusioniX or Causvid-style no-CFG workflows.
- **Long Prompts:** describe the whole scene, especially background elements not provided by reference images.
- **Steps:** use at least 15 steps for good quality with original VACE, or 30+ for best quality. FusioniX and accelerator LoRAs often work with about 8-10 steps.

## External Resources

- **GitHub:** https://github.com/ali-vilab/VACE/tree/main/vace/gradios
- **User Guide:** https://github.com/ali-vilab/VACE/blob/main/UserGuide.md
- **Preprocessors:** official tools for preparing VACE materials.

## Troubleshooting

### Poor Quality Results

1. Use longer, more detailed prompts.
2. Enable Skip Layer Guidance when the model supports CFG.
3. Increase the number of steps.
4. Check reference image quality.
5. Ensure the mask targets the intended area.

### Memory Issues

1. Use VACE 1.3B instead of 14B.
2. Reduce video length or resolution.
3. Enable quantization.

## Tips For Best Results

1. **Detailed Prompts:** describe everything in the scene, especially elements not in reference images.
2. **Quality Reference Images:** use high-resolution, well-lit reference images.
3. **Proper Masking:** take time to create precise masks with Matanyone.
4. **Consistent Lighting:** match lighting between reference images and the intended scene.
"""
