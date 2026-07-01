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
   MCP: `wangp_list_models`. Python: `session.list_model_metadata(...)`. Useful filters: `family`, `base_model_type`, `finetune`, `model_type`, `main_output`, `inputs`.
2. Pick the model from `metadata.capabilities`, `metadata.media_inputs`, `metadata.inputs`, `metadata.main_output`, and `metadata.outputs`.
3. Inspect `setting_values` before setting flag fields such as `image_prompt_type`, `video_prompt_type`, `audio_prompt_type`, `model_mode`, `sample_solver`, or `prompt_enhancer`.
4. Fetch defaults/schema and modify the few settings needed for the request.
   MCP: `wangp_get_model_schema` or `wangp_get_default_settings`. Python: `session.get_model_schema(model_type)` or `session.get_default_settings(model_type)`.
5. Generate, then return artifact paths and any structured errors.

## Media Input Rules

Use `metadata.media_inputs.image` to decide which image attachments can be supplied:

- `start`: `image_start`
- `end`: `image_end`
- `reference`: `image_refs`
- `single_reference`: one `image_refs` item is expected
- `multiple_references`: multiple `image_refs` items may be useful
- `background`: background image role, selected through `video_prompt_type` values containing `K`
- `injected_frames`: positioned-frame references, selected through `video_prompt_type` values containing `F` plus `frames_positions`
- `control`: `image_guide`
- `mask`: `image_mask`

Use `metadata.media_inputs.video` for `video_source`, `video_guide`, and `video_mask`. Use `metadata.media_inputs.audio.prompt` for audio prompt files and never treat it as audio output. Audio output is indicated by `metadata.outputs` containing `audio` or `metadata.capabilities.audio_output`.

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

Prefer exact values exposed in `setting_values` over composing flag strings by hand. Keep prompt enhancer off unless the user explicitly asks for prompt expansion. When supplying file paths, resolve them relative to the caller workspace or pass absolute paths. If validation or generation fails, surface the structured error instead of silently changing model or media inputs.

When writing settings JSON for WanGP, use UTF-8 without BOM. On Windows, set `PYTHONIOENCODING=utf-8` or keep report JSON ASCII-safe (`ensure_ascii=True`) if printing MCP event payloads to the console; progress text can contain Unicode characters.
