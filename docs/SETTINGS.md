# WanGP Settings

WanGP generation settings are JSON-serializable values consumed by `wgp.py` and by the Python API in `shared/api.py`.

The baseline schema lives in `models/_settings.json`. Model defaults in `defaults/*.json` and `finetunes/*.json` override those values, then handler code can update or hide settings according to the selected model definition. In practice, an exported settings file is the safest template for a specific model.

## Model Selection

| Setting | Type | Meaning |
| --- | --- | --- |
| `model_type` | string | Model definition id, for example `ltx2_22B_distilled`. Required for API tasks. |
| `model_mode` | string, number, or null | Optional model-specific mode selected from `model_def["model_modes"]`. Some image edit and TTS models use it for edit method, language, speaker, or scheduler variants. |
| `settings_version` | number | Saved settings schema version. WanGP uses this to migrate older saved settings in `fix_settings(...)`. |
| `client_id` | string | Optional API correlation id. Required only when an API caller wants returned in-memory artifacts for a specific submitted task. |

## Prompt Settings

| Setting | Type | Meaning |
| --- | --- | --- |
| `prompt` | string | Main text prompt, lyrics, speech text, or task instruction. |
| `negative_prompt` | string | Negative prompt when the model supports it. Ignored or removed for models with `no_negative_prompt`. |
| `alt_prompt` | string | Model-specific secondary prompt, such as music caption, emotion instruction, or alternate conditioning prompt. |
| `prompt_enhancer` | string flags | Prompt enhancer mode. Empty disables enhancement unless explicitly enabled elsewhere. See flag details below. |
| `multi_prompts_gen_type` | string flags | How multiline prompts are split into queue tasks or sliding-window prompts. See flag details below. |
| `multi_images_gen_type` | integer | For models that can treat multiple images as text prompts. `0` generally means one queue task per image; other values are model/UI dependent. |
| `custom_settings` | object or null | Model-specific settings keyed by custom setting id. WanGP supports the first six custom setting slots in the UI. |

## Output Shape

| Setting | Type | Meaning |
| --- | --- | --- |
| `image_mode` | integer | `0` for video/audio-generation mode, `1` for image mode, `2` for image inpainting mode. The available tabs depend on `image_outputs`, `v2i_switch_supported`, and `inpaint_support` in the model definition. |
| `resolution` | string | Output size as `WIDTHxHEIGHT`, for example `1280x720`. |
| `batch_size` | integer | Number of images for image-output models. For some special image models this may have model-specific meaning. Video and audio paths usually force one sample per repeat. |
| `video_length` | integer | Requested output frames. For audio-only models this is usually `0`; duration may come from `duration_seconds`. |
| `duration_seconds` | number | Requested duration for audio models or models exposing `duration_slider`. |
| `pause_seconds` | number | Pause inserted between multi-speaker or multi-sentence audio segments when supported. |
| `force_fps` | string | Output FPS policy. Empty uses the model default. Accepts `auto`, `control`, `source`, or a numeric string such as `24`. |
| `repeat_generation` | integer | Number of generated videos/audio files per prompt. Image models use `batch_size` instead. |
| `output_filename` | string | Optional output file name override. |

## Core Sampling

| Setting | Type | Meaning |
| --- | --- | --- |
| `seed` | integer | Random seed. `-1` means random. |
| `num_inference_steps` | integer | Denoising or generation step count when the model exposes inference steps. |
| `sample_solver` | string | Optional scheduler/solver selected from `model_def["sample_solvers"]`. |
| `flow_shift` | number | Flow matching shift scale when the model exposes `flow_shift`. |
| `temperature` | number | Sampling temperature for audio/TTS models that expose it. |
| `top_p` | number | Nucleus sampling limit for supported audio/TTS models. |
| `top_k` | integer | Top-k sampling limit for supported audio/TTS models. `0` usually disables top-k. |

## Guidance

| Setting | Type | Meaning |
| --- | --- | --- |
| `guidance_phases` | integer | Number of active CFG/guidance phases when a model supports phased guidance. |
| `model_switch_phase` | integer | Phase where a multi-submodel pipeline switches model, when applicable. |
| `switch_threshold` | integer | Phase 1 to 2 switch step/threshold, or model switch threshold for multi-submodel pipelines. |
| `switch_threshold2` | integer | Phase 2 to 3 switch step/threshold. |
| `guidance_scale` | number | CFG guidance for phase 1. |
| `guidance2_scale` | number | CFG guidance for phase 2. |
| `guidance3_scale` | number | CFG guidance for phase 3. |
| `audio_guidance_scale` | number | Audio guidance scale when `audio_guidance` is exposed by the model. |
| `embedded_guidance_scale` | number | Embedded guidance scale for models with embedded guidance. |
| `alt_guidance_scale` | number | Alternate-condition guidance scale for models exposing `alt_guidance`. |
| `alt_scale` | number | Alternate-condition strength for models exposing `alt_scale`. |
| `audio_scale` | number | Audio prompt/source strength when the model exposes `audio_scale_name`. |
| `control_net_weight` | number | Primary control weight for model families with control modules. |
| `control_net_weight2` | number | Secondary control weight when the model exposes two control weights. |
| `control_net_weight_alt` | number | Alternate control weight, for example Lynx or another named side condition. |

## Image And Video Inputs

| Setting | Type | Meaning |
| --- | --- | --- |
| `image_prompt_type` | string flags | Start/end/source continuation mode. See flag details below. |
| `image_start` | image or list | Start image(s) for image-to-video or image-conditioned generation. |
| `image_end` | image or list | End image(s) when the model supports end frames. |
| `image_refs` | image or list | Reference images selected by `video_prompt_type` flags such as `I`, `K`, `F`, or `J`. |
| `image_refs_relative_size` | integer | Relative internal size for reference images on models exposing `any_image_refs_relative_size`. |
| `remove_background_images_ref` | integer | Background-removal mode for reference images. Usually `0` off, `1` auto/on, with older values migrated by `fix_settings`. |
| `frames_positions` | string | Positions for `F` positioned-frame references. Syntax is model-specific but usually frame indexes or ranges. |
| `image_guide` | image | Control image used in image mode. WanGP maps this into `video_guide` internally for generation. |
| `image_mask` | image | Image mask used in image inpainting/control modes. |
| `video_source` | video | Source video for continuation (`image_prompt_type` `V`) or post-processing edit tasks. |
| `keep_frames_video_source` | string | Truncation for source video after resampling. Empty keeps all; negative truncates from the end. |
| `input_video_strength` | number | Strength of the source/start input when the selected model exposes this setting. |
| `video_prompt_type` | string flags | Main control/reference/mask mode. See flag details below. |
| `video_guide` | video | Control video for video mode, or control image after internal mapping in image mode. |
| `keep_frames_video_guide` | string | Truncation for control video after resampling. Empty keeps all. |
| `denoising_strength` | number | Control strength for guide-based generation. Visible when `G` is present in `video_prompt_type`. |
| `masking_strength` | number | Masked-control duration/strength. Used when a mask is active or the model always enables mask strength. |
| `video_mask` | video | Video mask for video control/inpainting. |
| `mask_expand` | integer | Expands or shrinks mask area for mask-enabled guide modes. |
| `custom_guide` | file | Model-specific guide file, for example trajectory or camera data. |
| `video_guide_outpainting` | string | Four-part outpainting expansion percentages encoded as `top;bottom;left;right`, or `#` for disabled/default. |
| `video_guide_outpainting_ratio` | string | Preset target ratio for guide outpainting, for example `16:9`; empty means manual expansion. |
| `min_frames_if_references` | integer | Image-mode workaround for v2i-capable models: generate one or more frames to preserve reference identity. Values above `1000` mean always generate that many frames minus `1000`. |

## Audio Inputs

| Setting | Type | Meaning |
| --- | --- | --- |
| `audio_prompt_type` | string flags | Audio conditioning/source mode. See flag details below. |
| `audio_guide` | audio | Primary audio prompt, source, reference voice, or soundtrack depending on model. |
| `audio_guide2` | audio | Secondary audio reference for two-speaker, emotion, or timbre modes. |
| `audio_source` | audio | External audio used by post-processing mode `custom`. |
| `speakers_locations` | string | Speaker ranges for multi-speaker audio/video models, for example `0:45 55:100`. |
| `seedvc_voice_sample` | audio | Reference voice for one-speaker SeedVC voice replacement. |
| `seedvc_voice_sample2` | audio | Second reference voice for two-speaker SeedVC voice replacement. |

## Acceleration And Cache

| Setting | Type | Meaning |
| --- | --- | --- |
| `skip_steps_cache_type` | string | Step-skipping cache type. Empty disables. Known values are `tea` and `mag` when the model exposes TeaCache or MagCache. |
| `skip_steps_multiplier` | number | Cache speed/skip multiplier. The meaning is model/cache dependent. |
| `skip_steps_start_step_perc` | integer | Percentage of denoising steps before cache skipping starts. |
| `temporal_upsampling` | string | Temporal post-processing mode. Current core values include `rife2` and `rife4`. |
| `spatial_upsampling` | string | Spatial post-processing mode. Empty disables; values include `vae1`, `vae2`, `lanczos<scale>`, and registered edit upsamplers such as FlashVSR. |
| `flashvsr_topk_ratio` | number | FlashVSR top-k ratio when FlashVSR post-processing is active. |
| `film_grain_intensity` | number | Film grain amount. `0` disables. |
| `film_grain_saturation` | number | Saturation of generated film grain. |
| `RIFLEx_setting` | integer | RIFLEx long-video policy. `0` auto, `1` always on, `2` always off. |
| `override_profile` | number | Per-task MMGP profile override. `-1` uses the normal selected profile. |
| `override_attention` | string | Per-task attention backend override. Empty uses global auto/default selection. |

## Post-Processing Audio

| Setting | Type | Meaning |
| --- | --- | --- |
| `postprocess_audio` | string | Audio post-processing mode. Empty disables. Known values are `custom`, `mmaudio`, `control`, `seedvc`, `seedvc2`, and for audio edit tasks `remove_background`. |
| `MMAudio_prompt` | string | Positive prompt for MMAudio soundtrack generation. |
| `MMAudio_neg_prompt` | string | Negative prompt for MMAudio soundtrack generation. |

## Advanced Sampling

| Setting | Type | Meaning |
| --- | --- | --- |
| `perturbation_switch` | integer | Enables perturbation/skip-layer guidance mode when supported. Model definitions provide the displayed choices. |
| `perturbation_layers` | list[int] | Transformer layers affected by perturbation. |
| `perturbation_start_perc` | integer | Start percentage for perturbation. |
| `perturbation_end_perc` | integer | End percentage for perturbation. |
| `apg_switch` | integer | Adaptive Projected Guidance. Usually requires guidance greater than 1. |
| `cfg_star_switch` | integer | CFG Star switch for models exposing `cfg_star`. |
| `cfg_zero_step` | integer | CFG Zero layer cutoff. `-1` disables. |
| `NAG_scale` | number | Normalized Attention Guidance scale. |
| `NAG_tau` | number | NAG tau value. |
| `NAG_alpha` | number | NAG alpha value. |
| `self_refiner_setting` | integer | Self-refiner mode. `0` disables; other values select the normalization strategy. |
| `self_refiner_plan` | list or string | Self-refiner step/iteration rules. WanGP normalizes this through `normalize_self_refiner_plan`. |
| `self_refiner_f_uncertainty` | number | Self-refiner uncertainty threshold. |
| `self_refiner_certain_percentage` | number | Self-refiner certainty skip percentage. |
| `motion_amplitude` | number | Motion acceleration/intensity multiplier for models exposing motion amplitude. |

## Sliding Window

| Setting | Type | Meaning |
| --- | --- | --- |
| `sliding_window_size` | integer | Number of frames in each sliding-window generation chunk. |
| `sliding_window_overlap` | integer | Frames overlapped between consecutive windows. |
| `sliding_window_color_correction_strength` | number | Color matching strength between windows. |
| `sliding_window_overlap_noise` | integer | Noise injected in overlap frames to reduce blur. |
| `sliding_window_discard_last_frames` | integer | Discarded tail frames per window. |
| `keep_intermediate_sliding_windows` | integer | Saved in server config, but exported settings may include it for traceability. Controls whether intermediate windows are retained. |

## LoRAs

| Setting | Type | Meaning |
| --- | --- | --- |
| `activated_loras` | list[string] | LoRA file names, URLs, or cached local entries. WanGP resolves them under the model's LoRA directory. |
| `loras_multipliers` | string | Per-LoRA multipliers. Space/newline separated; lines starting with `#` are ignored. Phase multipliers use semicolon syntax such as `0;1`. |

## Flag Settings

WanGP stores several UI selections as compact strings of letters. These are not free-form tags. Each model definition supplies allowed choices through keys such as `image_prompt_types_allowed`, `guide_preprocessing`, `mask_preprocessing`, `guide_custom_choices`, `image_ref_choices`, `audio_prompt_type_sources`, and `prompt_enhancer_def`.

### `image_prompt_type`

`image_prompt_type` describes the source/start/end media for the main generation path.

| Flag | Meaning |
| --- | --- |
| empty | Text-only/new generation. |
| `S` | Use `image_start` as starting image(s). |
| `E` | Use `image_end` as end image(s), if the selected source mode supports end frames. |
| `V` | Continue from `video_source`. |
| `L` | Continue from the last generated video. |
| `T` | Model definition sentinel for text/new-video availability. The stored value is usually empty rather than `T`. |

Only flags listed in `model_def["image_prompt_types_allowed"]` survive settings migration.

### `video_prompt_type`

`video_prompt_type` is the main control, mask, reference-image, and guide mode string. Model handlers build the visible UI choices, so not every flag is legal for every model.

| Flag | Meaning |
| --- | --- |
| `V` | A control video or control image is used (`video_guide` in video mode, `image_guide` in image mode). |
| `G` | Guide/denoise against the control media. Enables `denoising_strength`. |
| `U` | Keep/use the guide unchanged, identity/raw format, or bypass preprocessing depending on the model choice. |
| `A` | A mask is active. Uses `video_mask` in video mode or image mask data in image mode. |
| `N` | Non-masked-area variant when paired with mask choices such as `NA`, `XNA`, `YNA`, `WNA`, or `ZNA`. |
| `P` | Pose preprocessing. |
| `O` | Aligned pose preprocessing. |
| `D` | Depth preprocessing. |
| `E` | Canny edge preprocessing. |
| `S` | Scribble/shapes preprocessing. |
| `L` | Optical flow preprocessing. |
| `C` | Grayscale/recolor preprocessing. |
| `M` | Inpaint-mask preprocessing. |
| `B` | Face movement/control preprocessing. |
| `H` | Bounding-box preprocessing. |
| `X` | Inpaint outside the selected mask area. |
| `Y` | Depth outside the selected mask area. |
| `W` | Shapes/scribble outside the selected mask area. |
| `Z` | Flow outside the selected mask area. |
| `I` | Reference image(s) are used through `image_refs`. |
| `K` | Landscape, background, main subject, or first reference image role, depending on the model's `image_ref_choices`. |
| `F` | Positioned-frame reference images. Uses `frames_positions`. |
| `J` | Style image role used by Flux USO-style definitions. |
| `T` | Align control media/audio/positioned frames to the first generated sliding-window sample instead of the beginning of the source video. |
| `Q` | Hidden/model-forced special option used by specific handlers. Do not set manually unless the model definition selects it. |
| `&` | LTX-2 HDR-output option in the IC-LoRA control choices. |

The same letter can have slightly different labels in different handlers. For example, `V` means "Control Video" in video mode and "Control Image" in image mode. Always prefer a value from the selected model's exposed choices instead of composing a string by hand.

### `audio_prompt_type`

`audio_prompt_type` combines model-specific source choices with shared audio option flags.

The source part is defined by `model_def["audio_prompt_type_sources"]`. Common patterns are:

| Flag | Common meaning |
| --- | --- |
| empty | Text-only audio/video generation, or no audio source. |
| `A` | Primary audio source, reference voice, soundtrack, or source audio. |
| `B` | Secondary audio reference, second speaker, emotion reference, or timbre reference. |
| `X` | Auto-separate two speakers from one primary audio source. |
| `C` | Treat two audio sources as consecutive, played in a row. |
| `P` | Treat two audio sources as parallel speaker/audio tracks. |
| `K` | Use the control video's audio track as the audio prompt. Requires a control video mode. |
| `O` | Output/generated-audio selector for models that otherwise reuse the input audio as the soundtrack. |
| `F` | Use the full audio guide instead of per-window audio slices. |
| `1`, `2` | Model-specific source modes. For example, LTX-2 uses `1` inside `A1OF` for an ID-LoRA reference voice path and `2` for audio generation from control video plus text. |

Shared option flags can be appended:

| Flag | Meaning |
| --- | --- |
| `V` | Remove or ignore background music/noise for the audio prompt. |
| `L` | Allow video length to continue beyond the audio end when the model supports it. |
| `N` | Normalize audio volumes. Usually meaningful when both `A` and `B` are present. |
| `Y` | SeedVC one-speaker voice replacement option. |
| `Z` | SeedVC two-speaker voice replacement option. |
| custom | A one-letter model-defined option from `audio_prompt_type_custom_option`, when present. |

### `prompt_enhancer`

`prompt_enhancer` selects what the prompt enhancer receives and which model-specific instruction set it uses.

| Flag | Meaning |
| --- | --- |
| empty | Disabled. |
| `T` | Use the existing text prompt. |
| `I` | Include the start/source image when the selected model supports image-aware enhancement. |
| `K` | Enable "Think" mode for supported prompt enhancer backends. |
| `1` | Use the model's alternate enhancer instruction set, often a relayed/dialogue prompt mode. |

Examples: `T`, `TI`, `T1`, and `TI1`. Older saved values `I` and `IK` are migrated to `TI` and `TIK`.

### `multi_prompts_gen_type`

| Value | Meaning |
| --- | --- |
| `G` | Each non-empty line creates a new generation queue task. |
| `PG` | Each paragraph separated by an empty line creates a new generation queue task. |
| `W` | Each line becomes a prompt for a sliding window of the same video. |
| `PW` | Each paragraph becomes a prompt for a sliding window of the same video. |
| `FG` | Full prompt. All lines belong to one prompt. |

If the selected model does not support sliding windows, `W` modes are migrated to queue-task modes.

## API Metadata About Models

Each in-memory model definition now includes a `metadata` object inferred after handler initialization:

```json
{
  "metadata": {
    "model_type": "ltx2_22B_distilled",
    "family": "ltx2",
    "family_label": "LTX-2",
    "base_model_type": "ltx2_22B",
    "finetune": false,
    "main_output": ["video"],
    "outputs": ["video", "audio"],
    "inputs": ["text", "image", "video", "audio"],
    "media_inputs": {
      "image": {
        "start": true,
        "end": true,
        "reference": true,
        "single_reference": false,
        "multiple_references": true,
        "background": false,
        "injected_frames": false,
        "control": false,
        "mask": false
      },
      "video": {
        "continue": true,
        "last": false,
        "control": true,
        "mask": false
      },
      "audio": {
        "prompt": true,
        "output": true
      }
    },
    "capabilities": {
      "text_to_video": true,
      "image_to_video": true,
      "audio_output": true
    },
    "setting_values": {
      "image_prompt_type": {"allowed": "TVSE"},
      "video_prompt_type": {"image_ref_choices": {"choices": []}}
    }
  }
}
```

Inference rules:

- `finetune` is true when the model definition comes from `finetunes/`.
- `main_output` is `["audio"]` for `audio_only` models.
- `main_output` includes `image` for `image_outputs` models.
- `main_output` includes both `image` and `video` for models with `v2i_switch_supported` or `inpaint_support`; this includes Vace image generation.
- otherwise `main_output` defaults to `["video"]`.
- `outputs` starts from `main_output`, then adds `audio` for models with intrinsic audio output or audio-track generation (`returns_audio`), such as LTX-2, OVI, InfiniteTalk, Fantasy, LongCat Avatar, Hunyuan Avatar/Custom Audio, Magi Human, and Cosmos3. It does not include audio created later by `postprocess_audio`.
- `inputs` always includes `text`.
- `inputs` includes `audio` when `any_audio_prompt` is true.
- `inputs` includes `image` when start images, image references, alternate guide references, or inpainting are supported.
- `inputs` includes `video` when source-video or control-video choices are supported.
- `media_inputs.image` describes optional image attachment roles: start image, end image, one or more reference images, background image (`K` choices), injected frames (`F` choices), control image, and mask image.
- `media_inputs.video` describes optional video roles: continuation/source video, last generated video continuation, control video, and mask video.
- `media_inputs.audio.prompt` describes audio input support. It is not audio output; output audio is indicated by `outputs` containing `audio` or `capabilities.audio_output`.
- `capabilities` provides common text/image/video/audio workflow booleans derived from the model definition.
- `setting_values` exposes normalized allowed values for agent-facing settings, especially letter-flag settings such as `image_prompt_type`, `video_prompt_type`, `audio_prompt_type`, `prompt_enhancer`, and model-specific selectors such as `model_mode`.
