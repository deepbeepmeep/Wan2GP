---
name: wangp-agent
description: "Use when an agent needs to operate WanGP: discover available model capabilities, choose a model, inspect accepted inputs and setting values, build settings, run generation through the MCP server or Python API, poll jobs, cancel jobs, and return generated media artifact paths."
---

# WanGP Agent

## Tool Choice

Prefer the WanGP MCP server when its tools are available. Use the in-process Python API when working inside this repository or when MCP is not connected. Use the CLI only as a fallback for existing queue JSON/ZIP files or one-off smoke tests.

Python API bootstrap:

```python
from shared.api import init

session = init(console_output=False)
```

MCP server command for local clients:

```bash
python wgp.py --mcp --config <config dir> --output-dir <output dir>
```

Use `python -m shared.mcp_server --root <WanGP repo> --output-dir <output dir>` only when a client needs the lower-level adapter entrypoint. `wgp.py --mcp` is preferred because it preserves normal WanGP CLI/config behavior.

## Discovery Workflow

1. List candidate models before generating.
   For a user-facing model name, call MCP `wangp_search_models(query=...)` first. Python: `session.list_model_metadata(query=...)`. Use `wangp_list_models` only for an already-filtered category or a deliberately paged survey; never request the complete unfiltered list to locate one named model. Useful filters: `name`, `family`, `base_model_type`, `finetune`, `model_type`, `main_output`, `inputs`. String filters accept case-insensitive `*` and `?` glob patterns, for example `name="*Krea 2*"` or `model_type="krea2_*"`.
2. Pick the model from `metadata.capabilities`, `metadata.media_inputs`, `metadata.inputs`, `metadata.main_output`, and `metadata.outputs`.
3. Inspect the selected model with `wangp_get_model_schema` for its compact capability summary, frame limits, prompt guidance, and sliding-window support. If exact request parameters or their choices remain unclear, call `wangp_get_model` for the model's parameter declarations; its embedded default-value block is intentionally omitted.
4. Fetch defaults separately when building a raw generation request, and modify only the few settings needed for the request. Preserve model-specific flags unless the user explicitly supplied an exact supported value.
   MCP: `wangp_get_default_settings`. Python: `session.get_default_settings(model_type)`.
5. If the request involves LoRAs, call `wangp_list_loras(model_type)` and copy returned identifiers exactly into `activated_loras`, with corresponding values in `loras_multipliers`.
6. Generate, then return artifact paths and any structured errors.

Read `wangp://docs/settings` when the task involves model selection, prompts, output dimensions, sampling, guidance, media inputs, acceleration or cache options, post-processing, sliding windows, LoRAs, or API setting metadata. For only `image_prompt_type`, `video_prompt_type`, or `audio_prompt_type`, prefer the smaller `wangp://docs/settings/prompt-flags` resource. Read resources through the client's standard MCP resource-reading capability. WanGP also infers compatible `S`, `E`, and `V` image-source flags, the first declared reference-image mode containing `I`, and declared `A`/`B` audio-source modes when corresponding media fields are supplied without their flags.

For post-processing, call `wangp_postprocess(media_id=<media_id>)` without `process` to discover compatible spatial upsampling and refiner operations (for instance face correction), temporal upscaling, soundtrack, voice replacement, and audio editing. Then call it again with an exact returned process id and parameters; the operation runs through WanGP's normal generation queue.

For direct media utilities, call `wangp_toolbox()` without an action for a compact action list, then pass one action without `arguments` for its exact schema, then call it with `arguments`. It includes frame/video/audio extraction, transcription, resize/crop, muting, soundtrack replacement, video merging, color-frame creation, media details, and documentation lookup. Use `media_id` values returned by `wangp_list_gallery` for media arguments. Direct server filesystem paths are accepted only when the server was explicitly started with filesystem-read permission.

## Media Input Rules

Use `metadata.media_inputs.image` to decide which image attachments can be supplied:

- `start`: set `image_start` and include `S` in `image_prompt_type`.
- `end`: set `image_end` and include `E` in `image_prompt_type`; combine it with the source flag when needed, for example `SE`.
- `reference`: set `image_refs` and use a model-exposed `video_prompt_type` choice containing `I`.
- `single_reference`: use a `video_prompt_type` choice containing `I` and provide exactly one `image_refs` item.
- `multiple_references`: use a `video_prompt_type` choice containing `I` and provide multiple `image_refs` items.
- `background`: set `image_refs` and use a model-exposed `video_prompt_type` choice containing `K`, normally together with `I`, such as `KI`.
- `injected_frames`: set `image_refs` and `frames_positions`, then use a model-exposed `video_prompt_type` choice containing `F`.
- `control`: set `image_guide` and use a model-exposed `video_prompt_type` choice containing `V`; preserve any accompanying preprocessing flags in that choice.
- `mask`: set `image_mask` and use a model-exposed `video_prompt_type` choice containing `A`; preserve any accompanying mask/control flags in that choice.

Use `metadata.media_inputs.video` for `video_source`, `video_guide`, and `video_mask`. Use `metadata.media_inputs.audio.prompt` for audio prompt files and never treat it as audio output. Audio output is indicated by `metadata.outputs` containing `audio` or `metadata.capabilities.audio_output`.

Use `wangp_list_gallery` to discover compact summaries of existing session media; set `selected_only=true` when only the live visual/audio selections are needed. Pass its `media_id` directly in generation, post-processing, and toolbox media fields. Call `wangp_get_media_settings(media_id=...)` only when the media's full generation settings are needed; for media outside the Gallery, its mutually exclusive `path` input is available only when filesystem reads are enabled. Media IDs already observed by the MCP session remain resolvable while their files exist even if WanGP has trimmed their rows from the visible Gallery; remembered records have `in_gallery: false`. Reuse those IDs instead of recreating the media. For remote HTTP servers, use `wangp_create_gallery_upload` to obtain a short-lived PUT URL; a successful upload registers and selects the item in the Visual or Audio Gallery. Use `wangp_create_gallery_download(media_id=...)` for a short-lived GET URL when the user needs the resulting file locally. These transfer tools are not available over stdio.

## Long Video Workflows

A long video is any requested sequence that exceeds what the selected model should generate in one window, or that is deliberately built from several shots or continuation calls. The model must create it as multiple conditioned segments rather than one uninterrupted generation. Planning those segments matters because artifacts and character-identity drift can accumulate, transitions depend on overlap and anchors, larger windows cost more time and VRAM, and repeated continuation introduces additional lossy video encoding.

For a model whose `metadata.outputs` contains `video`, call `wangp_get_model_schema` and use `metadata.frames_maximum` as the suggested maximum frame count for one generation window or one continuation call. Treat a requested `video_length` greater than this value as a long-video workflow that needs an explicit window plan. Keep each window at or below `metadata.frames_maximum`; when `metadata.sliding_window` is true, one `video_length` request may span multiple windows.

The MCP/API frontier accepts `video_length` as a seconds string such as `"10s"`, so do not calculate frames manually. Set numeric `force_fps` when the user requests a specific FPS; otherwise WanGP uses `metadata.fps` and snaps the duration to the nearest valid frame count for the selected model.

Prefer one planned sliding-window generation when the complete sequence is known. Start from the filtered defaults, set `video_length` to the intended total, and keep `sliding_window_size` at or below `metadata.frames_maximum`. Set `multi_prompts_gen_type` to `W` for one non-empty prompt line per window or `PW` for one blank-line-separated paragraph per window. With `PW`, never put a blank line inside one window: keep every line and labeled section belonging to that window adjacent with single newlines, and use exactly one blank line only between complete windows. Before calling `wangp_generate`, split the prompt mentally on blank lines and verify that the resulting paragraph count equals the intended window count and, when used, the `image_end` count. Prefix a window prompt with optional commands when needed:

- `[/duration=121]`, `[/duration=5s]`, or `[/duration=20%]` selects that window's contributed output length. When duration commands are present, they define the window schedule and predicted total instead of treating `video_length` as a strict final cap.
- `[/overlap=9]` overrides the transition overlap; `[/overlap]` restores the model default.
- `[/overlap=0]` or `[/new_shot]` creates a hard cut when `metadata.capabilities.text_to_video` is true.
- Combine commands when useful, for example `[/duration=4s,/new_shot]`.

When `metadata.media_inputs.image.end` is true, strongly prefer planned end-frame anchors for long videos. This is one of the most effective ways to counter progressive identity, appearance, and composition drift: instead of allowing each window to inherit every error from the previous one, an end frame steers it back toward a prepared target.

- Generate the anchor images first with a suitable image model, keeping character identity, clothing, environment, style, and intended composition consistent with the corresponding video prompts.
- Prefer a master-image workflow: generate one high-quality reference image, select an image-edit model, and run one independent edit for every planned keyframe or end frame. Always give every edit the same master-image `media_id` rather than the previous edited output; request only the pose, action, camera, environment, or composition needed for that window. Collect the edited output media IDs in chronological order.
- Provide the ordered anchors through `image_end`, with one end frame for each planned window, and include `E` in `image_prompt_type`. Retain `S` when also using `image_start`, producing `SE`.
- Align each window prompt with its matching end-frame target so the motion leads naturally toward that image instead of fighting it.
- If end frames are unavailable but `metadata.media_inputs.image.injected_frames` is true, use intermediate anchors through `image_refs`, set their locations with `frames_positions`, and select a model-exposed `video_prompt_type` value containing `F`.

Use repeated continuation when the next scene should be chosen after reviewing the latest output. For each continuation, pass the latest generated video as `video_source`, include `V` in `image_prompt_type`, keep the newly generated portion within `metadata.frames_maximum`, and submit another `wangp_generate` job. The returned video already contains the source plus its generated continuation, so use that latest combined output for the next call and do not merge the same source segment again. This workflow permits improvisation but repeated video decoding and encoding can gradually reduce quality.

For transition behavior, RAM and quality tradeoffs, frame-count formulas, and human-facing controls, read the **Sliding Windows For Long Videos** section of `wangp://docs/processing`.

## Generation

MCP generation is asynchronous by default:

```json
{
  "source": {
    "model_type": "example_model",
    "prompt": "A concise prompt",
    "image_mode": 0,
    "_api": {"return_media": true}
  }
}
```

Poll with `wangp_get_job(job_id)`. Use `wangp_cancel_job(job_id)` if the user asks to stop. For multiple requests in a row, keep using the same MCP server or API session so model/runtime caches stay warm.

Some MCP clients expose tool return dictionaries as JSON text content instead of `structuredContent`. If `structuredContent` is empty, parse the first text content item as JSON before treating the call as failed.

Python generation:

```python
result = session.run_task(settings)
paths = result.generated_files
errors = [str(error) for error in result.errors]
```

Only request `_api.return_media`, `_api.return_video_uint8`, or `_api.return_audio` when the agent actually needs in-memory tensors/audio; artifact paths are usually enough.

## Practical Guardrails

Read the prompt-flag resource instead of composing unfamiliar flag strings by hand. Keep prompt enhancer off unless the user explicitly asks for prompt expansion. Through MCP, prefer media IDs returned by `wangp_list_gallery` and use server filesystem paths only when the server explicitly permits them. For Python or CLI calls, resolve paths relative to the caller workspace or pass absolute paths. If validation or generation fails, surface the structured error instead of silently changing model or media inputs.

When writing settings JSON for WanGP, use UTF-8 without BOM. On Windows, set `PYTHONIOENCODING=utf-8` or keep report JSON ASCII-safe (`ensure_ascii=True`) if printing MCP event payloads to the console; progress text can contain Unicode characters.
