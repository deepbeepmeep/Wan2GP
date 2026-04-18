# WanGP Python API

`shared/api.py` provides a lightweight in-process wrapper over WanGP's existing generation path.

The main goal is to let third-party code call WanGP directly, keep the last loaded model alive across requests, receive structured progress updates, and still capture the same stdout/stderr output that would normally go to the console.

This same API can be used directly from a WanGP PlugIn and uses WanGP Web Queue to process the requested Jobs of called from a Third Party App (for instance https://github.com/deepbeepmeep/LTX-Desktop-WanGP).

**Please note that use of the WanGP API is subject to the WanGP Terms and Conditions. Any product that integrates WanGP should clearly disclose that it uses WanGP in both its user interface and its documentation.**

## Quick Start

The WanGP API consumes mostly *WanGP Settings*. In order to get the format of the settings, just launch the Web interface of WanGP, pick the model you want to use and fill the UI with settings you want to use, then click the *Export Settings* button at the bottom.

```python
from pathlib import Path

from shared.api import init

session = init(
    root=Path(r"C:\WanGP"),
    cli_args=["--attention", "sdpa", "--profile", "4"],
)

settings = {
    "model_type": "ltx2_22B_distilled",
    "prompt": "Cinematic shot of a neon train entering a rainy station",
    "resolution": "1280x704",
    "num_inference_steps": 8,
    "video_length": 97,
    "duration_seconds": 4,
    "force_fps": 24,
}

job = session.submit_task(settings)

for event in job.events.iter(timeout=0.2):
    if event.kind == "progress":
        progress = event.data
        print(progress.phase, progress.progress, progress.current_step, progress.total_steps)
    elif event.kind == "preview":
        preview = event.data
        if preview.image is not None:
            preview.image.save("preview.png")
    elif event.kind == "stream":
        line = event.data
        print(f"[{line.stream}] {line.text}")

result = job.result()
if result.success:
    print(result.generated_files)
else:
    for error in result.errors:
        print(error.message)
```

## Main Entry Points

- `init(...) -> WanGPSession`
  - Creates a reusable session and eagerly loads the runtime.
  - This is the normal entrypoint for standalone / CLI-style third-party integrations.
- `WanGPSession.submit(source, callbacks=None) -> SessionJob`
  - Starts a job from a settings dict, a manifest list, or a saved `.json` / `.zip` file.
- `WanGPSession.submit_task(settings, callbacks=None) -> SessionJob`
  - Preferred single-task entrypoint.
- `WanGPSession.submit_manifest(settings_list, callbacks=None) -> SessionJob`
  - Batch entrypoint for multiple tasks.
- `SessionJob.result() -> GenerationResult`
  - Waits for completion and returns a structured result object.
- `SessionJob.cancel()`
  - Requests cancellation of the active generation.

### When `init(...)` Is Needed

- Standalone / CLI-style third-party app or Python script:
  - yes, call `init(...)`
- WanGP plugin tab receiving an injected `api_session` in `create_config_ui(...)`:
  - no, do not call `init(...)`
  - use the injected `api_session` directly
- Low-level in-process integration inside WanGP with direct access to the live Gradio `state`:
  - possible with `init(webui_state=state, ...)`
  - this is an advanced/internal path, not the normal plugin entrypoint

## `init(...)` Parameters

```python
session = init(
    root=Path(r"C:\WanGP"),
    config_path=Path(r"C:\WanGP\wgp_config.json"),  # optional
    output_dir=Path(r"C:\WanGP\outputs_override"),  # optional
    callbacks=MyCallbacks(),                        # optional
    cli_args=["--attention", "sdpa"],              # optional
    console_output=True,                           # optional, default=True
    console_isatty=True,                           # optional, default=True
    webui_state=None,                              # optional
)
```

- `root`
  - Path to the WanGP installation folder.
  - Example: `C:\WanGP`

- `config_path`
  - Optional path to `wgp_config.json`.
  - If omitted, WanGP uses `C:\WanGP\wgp_config.json`.
  - This must point to a file named `wgp_config.json`.

- `output_dir`
  - Optional override for generated outputs.
  - If omitted, WanGP uses the output paths defined in the config file.

- `callbacks`
  - Optional callback object. See the callback section below.

- `cli_args`
  - Optional WanGP startup flags.
  - Example: `["--attention", "sdpa", "--profile", "4"]`

- `console_output`
  - Enables or disables writing WanGP stdout/stderr to the real console.
  - Default: `True`
  - The stream object always receives a copy of stdout/stderr, regardless of this setting.

- `console_isatty`
  - Controls the TTY capability reported by the API's console capture wrapper.
  - Default: `True`
  - Keep this enabled if you want tqdm or other terminal-style progress output to behave like a live console stream even when WanGP is called from another Python process.

- `webui_state`
  - Optional live WanGP WebUI state dictionary.
  - When provided, submitted tasks target the existing WanGP Gradio queue instead of calling the headless generation path directly.
  - This is mainly a low-level internal / advanced integration hook.
  - Normal WanGP plugins should use the injected `api_session` instead of calling `init(...)` themselves.
  - In this mode, `output_dir` is ignored and `on_stream(...)` is not expected because the WebUI queue path is probe-based rather than stdout-capture based.

### WebUI Queue Mode

This is the WebUI-backed queue mode used by WanGP plugin integrations.

If you are implementing a low-level in-process integration and explicitly need to bind the API to WanGP's live Gradio queue yourself, pass the current `state` dict to `init(...)`:

```python
from shared.api import init

session = init(webui_state=state, console_output=False)
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "A cinematic rainy street at night",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
    }
)
```

This mode reuses WanGP's existing WebUI queue loader server-side, then probes the live queue by `client_id` until the task is admitted, completed, fails, or is cancelled.

Differences versus standalone / CLI-style mode:

- third-party apps normally use plain `init(...)` without `webui_state`
- this WebUI-backed mode is the WanGP-side queue path
- low-level direct use still calls `init(...)`, but with `webui_state=state`
- jobs run through WanGP's existing Gradio queue instead of the headless direct generation path
- `output_dir` is ignored
- `on_stream(...)` / `job.events` console lines are not the primary signal because this path is probe-based
- the API banner is intentionally not printed in this mode

### Plugin Tab API Object

For WanGP plugins, a WebUI-backed session can be injected directly into the tab constructor.

If a plugin tab constructor accepts one extra positional argument, WanGP now passes a session-shaped API object there:

```python
def create_config_ui(self, api_session):
    ...
```

That object behaves like a normal `WanGPSession`. In a WanGP plugin tab, the WebUI queue integration stays inside the injected session object, so plugin code can keep using the same `submit_task(...)`, `job.result()`, and `job.cancel()` pattern as CLI code.

Plugin-specific difference:

- in a plugin tab, you normally do **not** call `init(...)`
- WanGP creates and injects the WebUI-backed session object for you
- plugin code should treat `api_session` as the API entrypoint
- this is the normal WebUI/API path for WanGP plugins

Minimal plugin pattern:

```python
def create_config_ui(self, api_session):
    active_job = {"job": None}

    def start_demo(progress=gr.Progress(track_tqdm=False)):
        class DemoCallbacks:
            ratio = 0.0

            def on_status(self, status):
                status = str(status or "").strip()
                if status:
                    progress(self.ratio, desc=status)

            def on_progress(self, update):
                self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
                progress(self.ratio, desc=str(getattr(update, "status", "") or "Generating..."))

        job = api_session.submit_task(settings, callbacks=DemoCallbacks())
        active_job["job"] = job
        result = job.result()
        return result.generated_files[0] if result.success and result.generated_files else gr.update()

    def cancel_demo():
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()
```

In a plugin tab, callback methods like `on_status(...)` and `on_progress(...)` can safely update a local `gr.Progress(...)` while the job itself still runs through WanGP's main WebUI queue.

Useful `SessionJob` handles for plugin authors:

- `job.result(timeout=None) -> GenerationResult`
  - Waits for completion and returns generated files, errors, and any requested returned media artifacts.
- `job.cancel()`
  - Requests cancellation of the current job.
- `job.done`
  - `True` once the job has fully finished.
- `job.events`
  - Event stream for progress, preview, status, output, and completion events.
- `job.cancel_requested`
  - `True` after cancellation was requested.

Useful `GenerationResult` fields:

- `result.generated_files`
  - Output paths collected from WanGP's normal gallery / save path handling.
- `result.errors`
  - Structured generation errors. Runtime failures do not raise from `submit_task(...)`; they appear here.
- `result.artifacts`
  - Optional returned media payloads requested through `_api`.
- `result.cancelled`
  - Optional returned if job was cancelled / aborted.

Useful `GeneratedArtifact` fields:

- `artifact.path`
  - Saved output path for that artifact when WanGP produced one.
- `artifact.media_type`
  - `"video"`, `"image"`, or `"audio"`.
- `artifact.video_tensor_uint8`
  - Optional returned video tensor in WanGP's native post-decode layout: `[C, F, H, W]`, `uint8`.
- `artifact.audio_tensor`
  - Optional returned audio tensor / array when requested.
- `artifact.audio_sampling_rate`
  - Sampling rate associated with `artifact.audio_tensor` when present.
- `artifact.fps`
  - Output FPS associated with `artifact.video_tensor_uint8` when present.

### Getting Outputs In Memory

By default, the API gives you output file paths in `result.generated_files`.

If you also want the generated media directly in memory, request it explicitly:

- pass `_api={"return_media": True}` in the submitted task settings
- then read the returned payloads from `result.artifacts`

Important notes:

- `result.generated_files` and `result.artifacts` are not mutually exclusive
  - WanGP still saves normal output files
  - the same completed task can also return in-memory media through `result.artifacts`
- `result.artifacts` is ordered like the submitted tasks for the tasks that actually returned media
- for video, the returned tensor layout is `[C, F, H, W]` with `dtype=uint8`
- for audio, use `artifact.audio_tensor` together with `artifact.audio_sampling_rate`

Minimal example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
        "_api": {"return_media": True},
    }
)

result = job.result()

artifact = result.artifacts[0]
video_tensor = artifact.video_tensor_uint8   # [C, F, H, W], uint8
audio_tensor = artifact.audio_tensor         # optional
audio_sr = artifact.audio_sampling_rate      # optional
saved_path = artifact.path                   # optional saved file path
```

If you only care about the in-memory result, you can ignore `result.generated_files` and work directly from `result.artifacts`.

## Accepted Input Shapes

Relative attachment paths are normalized to absolute paths when the job is submitted.

- For direct settings dictionaries and `.json` settings files, the base is the API caller's current working directory at submit time.
- For `.zip` queue files, WanGP keeps the queue bundle behavior and resolves bundled media from the extracted queue contents.
- A few WanGP string-like fields are normalized for convenience. For example, `force_fps` may be passed as `24` or `"24"`.

### Targeting A Frame Range Inside A Video File

For media inputs that accept a file path, WanGP also supports a virtual-media suffix so you can target only part of a source file without extracting an intermediate clip first.

Syntax:

- `path/to/file.ext|start_frame=123,end_frame=456`

Notes:

- `start_frame` is zero-based
- `end_frame` is inclusive
- the underlying source file stays the same; WanGP just decodes the requested frame range
- this is especially useful for `video_guide`, `video_mask`, and similar video-input fields

Optional audio-track targeting is also supported:

- `path/to/file.ext|start_frame=123,end_frame=456,audio_track_no=2`

Example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "video_prompt_type": "VG",
        "video_guide": r"F:\ALIENS_t01.mkv|start_frame=57542,end_frame=57782",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
    }
)
```

If you want to build that string programmatically, use `shared.utils.virtual_media.build_virtual_media_path(...)`.

### Optional API Meta Settings

`submit_task(settings, ...)` also accepts a reserved `_api` dictionary inside `settings`. This is API metadata, not a normal WanGP generation setting.

Current keys:

- `_api={"return_media": True}`
  - Requests returned media artifacts in `result.artifacts`.
  - Video outputs return `artifact.video_tensor_uint8` when WanGP can expose a contiguous `uint8` tensor.
  - Audio outputs return `artifact.audio_tensor` and `artifact.audio_sampling_rate` when available.

Example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
        "_api": {"return_media": True},
    }
)

result = job.result()
artifact = result.artifacts[0]
video_tensor = artifact.video_tensor_uint8
```

### Single Task

For single-task use, the intended input is the task settings dictionary itself:

```python
settings = {
    "model_type": "qwen_image_20B",
    "prompt": "A red bicycle parked in front of a bakery",
    "resolution": "1024x1024",
    "num_inference_steps": 4,
    "image_mode": 1,
}

job = session.submit_task(settings)
```

### Manifest

`submit_manifest(...)` accepts a list of settings dictionaries:

```python
settings_list = [
    {
        "model_type": "qwen_image_20B",
        "prompt": "A quiet library at sunrise",
        "resolution": "1024x1024",
        "num_inference_steps": 4,
        "image_mode": 1,
    },
    {
        "model_type": "qwen_image_20B",
        "prompt": "A rainy alley with neon signs",
        "resolution": "1024x1024",
        "num_inference_steps": 4,
        "image_mode": 1,
    },
]

job = session.submit_manifest(settings_list)
```

### Saved Queue / Settings File

`submit(...)` also accepts:

- a `.json` settings file path
- a `.zip` saved queue path

Example:

```python
job = session.submit(Path(r"C:\WanGP\my_queue.zip"))
```

## Streaming Events

Each job exposes `job.events`, a `SessionStream`.

The stream yields `SessionEvent` objects:

```python
SessionEvent(
    kind="progress",
    data=ProgressUpdate(...),
    timestamp=1710000000.0,
)
```

Known `kind` values:

- `started`
  - Job accepted and session processing started.
- `progress`
  - Structured progress update.
- `preview`
  - RGB preview update.
- `stream`
  - One stdout/stderr line.
- `status`
  - WanGP status message.
- `info`
  - WanGP informational message.
- `output`
  - Raw output refresh event from WanGP.
- `refresh_models`
  - Raw model-refresh event from WanGP.
- `completed`
  - Final `GenerationResult`.
- `error`
  - One `GenerationError` record.

## Returned Objects

### `GenerationResult`

Returned by `job.result()`:

```python
GenerationResult(
    success=False,
    generated_files=[
        r"C:\WanGP\outputs\clip_001.mp4",
    ],
    errors=[
        GenerationError(
            message="Task 2 failed validation",
            task_index=2,
            task_id=2,
            stage="validation",
        ),
    ],
    total_tasks=3,
    successful_tasks=2,
    failed_tasks=1,
)
```

Fields:

- `success: bool`
  - `True` only when every submitted task completed without error.
- `generated_files: list[str]`
  - Absolute paths to every file generated by the job, including partial-success runs.
- `errors: list[GenerationError]`
  - Structured error records collected during the run.
- `total_tasks: int`
  - Number of tasks submitted in the job.
- `successful_tasks: int`
  - Number of tasks that completed successfully.
- `failed_tasks: int`
  - Number of tasks that failed or were cancelled.

`job.result()` does not raise generation-task failures. Instead, inspect `result.success` and `result.errors`.

### `GenerationError`

Delivered through `error` events, `on_error(...)`, and `GenerationResult.errors`:

```python
GenerationError(
    message="Task 2 did not complete successfully",
    task_index=2,
    task_id=2,
    stage="generation",
)
```

Fields:

- `message: str`
  - Human-readable error message.
- `task_index: int | None`
  - One-based task index when the error is associated with a specific task.
- `task_id: Any`
  - Task identifier from the manifest when available.
- `stage: str | None`
  - Error stage such as `validation`, `generation`, `cancelled`, or `runtime`.

### `ProgressUpdate`

Delivered through `progress` events and `on_progress(...)`:

```python
ProgressUpdate(
    phase="inference",
    status="Prompt 1/1 | Denoising | 7.2s",
    progress=54,
    current_step=4,
    total_steps=8,
    raw_phase="Denoising",
    unit=None,
)
```

Fields:

- `phase: str`
  - Normalized phase. Typical values:
  - `loading_model`
  - `encoding_text`
  - `inference`
  - `decoding`
  - `downloading_output`
  - `cancelled`
- `status: str`
  - Human-readable status string produced by WanGP.
- `progress: int`
  - Estimated percentage from `0` to `100`.
- `current_step: int | None`
  - Current inference step when available.
- `total_steps: int | None`
  - Total inference steps when available.
- `raw_phase: str | None`
  - Original WanGP phase label before normalization.
- `unit: str | None`
  - Optional progress unit if WanGP provides one.

### `PreviewUpdate`

Delivered through `preview` events and `on_preview(...)`:

```python
PreviewUpdate(
    image=<PIL.Image.Image image mode=RGB size=800x200>,
    phase="inference",
    status="Prompt 1/1 | Denoising",
    progress=54,
    current_step=4,
    total_steps=8,
)
```

Fields:

- `image: PIL.Image.Image | None`
  - RGB preview image generated from WanGP's latent preview payload.
- `phase`, `status`, `progress`, `current_step`, `total_steps`
  - Same interpretation as `ProgressUpdate`.

### `StreamMessage`

Delivered through `stream` events and `on_stream(...)`:

```python
StreamMessage(
    stream="stdout",
    text="New video saved to Path: C:\\WanGP\\outputs\\clip_001.mp4",
)
```

Fields:

- `stream: str`
  - Usually `stdout` or `stderr`.
- `text: str`
  - One redirected line of console output.

### `SessionEvent`

Generic event wrapper:

```python
SessionEvent(
    kind="stream",
    data=StreamMessage(stream="stdout", text="Model loaded"),
    timestamp=1710000000.0,
)
```

Fields:

- `kind: str`
  - Event type.
- `data: Any`
  - Payload object for that event.
- `timestamp: float`
  - Event creation time.

## Callback Object

You can pass a callback object to `init(...)` / `WanGPSession(...)` as the session default, or pass one directly to `submit(...)`, `submit_task(...)`, or `submit_manifest(...)` for a specific job.

Supported callback methods:

- `on_progress(progress_update)`
  - Called when WanGP emits a structured progress update.
  - Use this for progress bars, step counters, and status text.

- `on_preview(preview_update)`
  - Called when a preview image is available.
  - Use this when you want live RGB preview frames during inference.

- `on_stream(stream_message)`
  - Called for every redirected stdout/stderr line.
  - This is the programmatic equivalent of watching the terminal output.

- `on_status(text)`
  - Called for WanGP status messages.
  - Use this if you want coarse status without parsing full progress objects.

- `on_info(text)`
  - Called for informational messages.

- `on_output(data)`
  - Called for raw WanGP output refresh events.
  - This is a low-level hook and is usually not needed by third-party integrations.

- `on_complete(result)`
  - Called when the job finishes.
  - Receives a `GenerationResult`.

- `on_error(error)`
  - Called each time WanGP reports a task or runtime error.
  - Receives a `GenerationError`.

- `on_event(session_event)`
  - Generic catch-all event hook.
  - Called alongside the specific callback above, not instead of it.

Example:

```python
class Callbacks:
    def on_progress(self, progress):
        print("progress:", progress.progress, progress.phase)

    def on_preview(self, preview):
        if preview.image is not None:
            preview.image.save("latest_preview.png")

    def on_stream(self, line):
        print(f"[{line.stream}] {line.text}")

    def on_complete(self, result):
        print("success:", result.success)
        print("generated:", result.generated_files)

    def on_error(self, error):
        print("error:", error.message)
```

Full signature example:

```python
from shared.api import GenerationError, GenerationResult, PreviewUpdate, ProgressUpdate, SessionEvent, StreamMessage


class VerboseCallbacks:
    def on_progress(self, progress: ProgressUpdate) -> None:
        print("progress", progress.progress, progress.current_step, progress.total_steps)

    def on_preview(self, preview: PreviewUpdate) -> None:
        print("preview", preview.phase, preview.image.size if preview.image is not None else None)

    def on_stream(self, line: StreamMessage) -> None:
        print(line.stream, line.text)

    def on_status(self, text: str) -> None:
        print("status", text)

    def on_info(self, text: str) -> None:
        print("info", text)

    def on_output(self, data: object) -> None:
        print("output", data)

    def on_complete(self, result: GenerationResult) -> None:
        print("success", result.success)
        print("files", result.generated_files)

    def on_error(self, error: GenerationError) -> None:
        print("error", error.stage, error.task_index, error.message)

    def on_event(self, event: SessionEvent) -> None:
        print("event", event.kind)
```

## Cancellation

```python
job = session.submit_task(settings)
job.cancel()
```

Cancellation is cooperative and forwards WanGP's normal abort signal to the active model. A cancelled run completes with `result.success == False` and a cancellation entry in `result.errors`.
