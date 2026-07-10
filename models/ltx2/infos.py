LTX2_INFOS = """
# LTX2 Workflows

## What The Model Can Do

- Text to video with soundtrack: write a cinematic prompt and LTX2 generates both the video and an audio track.
- Image or video continuation: provide a Start Image, End Image, or Video to Continue to keep identity, framing, or motion continuity.
- Control Video / Frames Injection: guide the new video with motion, structure, raw frames, HDR conversion, outpainting, or injected reference frames.
- Inpainting: 22B can regenerate masked regions of a Control Video while preserving the unmasked video context.
- Ingredients Reference Sheet: 22B can use one composite reference-sheet image with the Ingredients IC-LoRA to keep characters, props, and location consistent.
- EditAnything variants: provide a source/control video plus one reference image to add or edit a subject in the video.

## Text To Image Mode

LTX2 image generation is implemented by generating a short video internally and keeping only the first frame.

Image quality, reference identity, and control-image adherence can often be improved from the Quality tab with `Generate more frames to preserve Reference Image Identity / Control Image Information or improve`.

Some IC-LoRAs, such as the union-control LoRA used by pose, depth, and canny control, need at least 9 frames so the model has more than one temporal latent. If image mode uses these controls and only one latent was requested, WanGP automatically expands the internal generation to 9 frames and still returns only the first frame.

## Control Video Processes

- `No Video Process`: the Control Video is not used as a visual guide.
- `Transfer Human Motion`: extracts body pose/motion from the Control Video.
- `Transfer Human Motion With Pose Alignment`: extracts pose and aligns it to your Start Image, Video to Continue, or background reference.
- `Transfer Depth`: follows the depth and scene layout of the Control Video.
- `Transfer Canny Edges`: follows strong edges and outlines from the Control Video.
- `LTX2 Raw Format / Control Video for Ic Lora`: uses the Control Video frames directly. Use this for IC-LoRA style control, to provide the video to be outpainted, and for `Generate Audio based on Control Video`.
- `Inpaint Masked Area`: 22B only. Uses the Control Video plus a Video Mask to regenerate the masked area. This mode requires `Control Video Strength` set to `1` and `Unmasked Area Strength` set to `0`; the unmasked area is preserved by the inpainting workflow.
- `Ingredients Reference Sheet`: 22B only. Duplicates one uploaded reference-sheet image as the IC-LoRA guide video; use a clean composite sheet on a white background, with black separator lines between individual pieces and without text.
- `Convert SDR to HDR (IC-LoRA)`: 22B only. Converts an SDR Control Video toward HDR output.
- `Inject Frames`: places selected Reference Images at exact frame positions. In `Positions of Injected Frames`, `1` means the first frame and `L` means the last frame of a sliding-window segment.

## Audio Options

- `Generate Video & Soundtrack based on Text Prompt`: no audio file is needed. The prompt drives both the visuals and the generated soundtrack.
- `Generate Video based on Soundtrack and Text Prompt`: upload an Audio Prompt to guide timing, rhythm, speech, or sound events. If you leave it blank, WanGP uses null audio so the model is not driven by a real soundtrack. When the audio covers the generated window, it is normally reused as the final output audio.
- `Generate Video based on Control Video + its Audio Track and Text Prompt`: the audio track is extracted from the Control Video and used like the soundtrack prompt. This requires a Control Video with an audio track. When that track covers the generated window, it is normally reused as the final output audio.
- `Generate Audio based on Control Video and Text Prompt`: generates audio from a raw Control Video plus the text prompt. It requires `LTX2 Raw Format / Control Video for Ic Lora` and is not compatible with pose/depth/canny/HDR/outpainting/mask/injected-frame modes.
- `Generate Video based on Reference Voice (ID-LoRA) and Text Prompt`: available when the ID-LoRA option is listed. Provide a reference voice/audio sample and describe the on-camera speaker and speech in the prompt.
- `Prompt Audio Strength`: controls how strongly the uploaded soundtrack/audio prompt affects the generated result.
- `Ignore Background Music`: removes or reduces background music/noise from the audio prompt before using it for conditioning.
- `Postprocess Remux Audio`: after generation, you can replace or reuse the final soundtrack with a Custom Soundtrack, MMAudio, or the Control Video audio track.

## New Content Generation

If the Control Video or Control Audio has a shorter duration than the number of frames to generate, LTX2 will complete with new video and audio content.

## Control Video & Audio Timing With Video Continuation

When you use an Audio Prompt or the Control Video audio track, WanGP slices the audio per sliding window so each window receives the matching part of the soundtrack.

Sliding Window overlap frames are used for continuity and then removed from the final stitched video. Their matching audio is also used only to help the next window start smoothly.

If you continue from a Video to Continue, the alignment dropdown decides where Control Video, Control Audio, and Positioned Frames start on the timeline:

- `Aligned to the beginning of the Source Video`: frame/time 0 means the first frame of the source video you are continuing. Use this when your Control Video, audio prompt, or positioned frame numbers include the original source video at the beginning.
- `Aligned to the beginning of the First Window of the new Video Sample`: frame/time 0 means the first newly generated part after the source video. Use this when your Control Video, soundtrack, or positioned frame numbers are only meant for the continuation.

Example: if you continue a 4 second source video and your soundtrack starts with the new action, choose `Aligned to the beginning of the First Window of the new Video Sample`. If your soundtrack starts with those same 4 source seconds before the new action, keep `Aligned to the beginning of the Source Video`.

# Advanced

## Changing System LoRA 

LTX2 automatically adds some system LoRAs when a feature needs them. Each system LoRA comes with predefined LoRA Multipliers that may adjust to the context. 
You can change the LoRA used for a system process and the LoRA multipliers that will used for this LoRA.
A selected LoRA will be automatically used instead of a system LoRA if it contains in its name a text sequence signature.

Recognized system LoRA signatures:
- `distilled-lora`: distilled stage LoRA used by dev models for two-phase, Distilled 8 Steps, HQ/res2s, and some ID-LoRA cases.
- `union-control`: IC-LoRA used by Pose, Pose Alignment, Depth, and Canny control.
- `ic-lora-hdr`: HDR IC-LoRA used by 22B HDR output.
- `ic-lora-outpaint`: outpainting IC-LoRA used by 22B legacy spatial outpainting.
- `in-outpainting`: inpainting/outpainting IC-LoRA used by 22B mask-based inpainting and new outpainting.
- `ic-lora-ingredients`: Ingredients IC-LoRA used by the 22B Ingredients Reference Sheet process.
- `id-lora-celebvhq`: ID-LoRA used by the reference voice workflow.

To change one of these, manually select a LoRA whose filename contains the same recognized signature, then set its multipliers in the LoRAs tab. Because your selected LoRA has the recognized signature, WanGP skips the automatic default and uses your selected one instead.

Use a single number for one weight, such as `0.7`. Use `phase1;phase2` for two-phase generation, such as `0;1`, `1;0`, or `0.25;0.5`.


Examples:

```text
Select: custom-distilled-lora-v2.safetensors
Multiplier: 0.3;0.7
Filename rule: it contains "distilled-lora"
Result: the dev model uses your custom distilled LoRA instead of the default distilled LoRA and and it will apply your choices of LoRA multipliers.
```

```text
Select: ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
Multiplier: 0.6;0.2
Result: Pose/Depth/Canny control uses your manual phase weights instead of the automatic union-control weight.
```

```text
Select: my-ic-lora-outpaint-experiment.safetensors
Multiplier: 0.8
Filename rule: it contains "ic-lora-outpaint"
Result: 22B outpainting uses your outpaint LoRA instead of the built-in one.
```


```text
Select: my-id-lora-celebvhq-voice.safetensors
Multiplier: 1;0
Filename rule: it contains "id-lora-celebvhq"
Result: the reference voice workflow uses your ID-LoRA file and weight.
```
"""

LTX2_MSR_INFOS = """
# LTX2 Multiple Subject Reference

This model is configured for the LiconStudio Multiple Subject Reference LoRA. It uses reference images only; Control Video, pose, depth, canny, HDR, outpainting, masks, and frame injection are disabled for this workflow.

## Reference Image Order

Use one of the `MSR Reference Images` modes:

- `Background + Up to 4 Subjects`: upload 2 to 5 images. Image 1 is the background or scene reference; images 2 to 5 are the subject or object references to preserve.
- `Subjects / Objects only`: upload 1 to 4 subject or object references on a plain white or neutral background.

In `Background + Up to 4 Subjects` mode, WanGP internally reorders the references before denoising so the MSR LoRA receives the subject references first and the background reference last, matching the upstream MSR convention. In `Subjects / Objects only` mode, WanGP keeps the uploaded reference order unchanged.

## Reference Image Preparation

Subject or object references work best on a plain white background. If your non-background references are not already isolated on white, keep `Remove Background behind MSR Subjects / Objects` enabled so WanGP removes the subject background and places it on white.

Character sheets are recommended for character references: use an image that shows the same character from several points of view, poses, or close-up/detail angles. This gives MSR more identity and clothing information than a single portrait.

Use the text prompt to describe how the referenced subjects should appear together in the referenced environment.
"""
