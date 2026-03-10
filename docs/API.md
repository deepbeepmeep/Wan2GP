# WanGP Python API

`shared/api.py` provides a lightweight in-process wrapper over WanGP's existing generation path.

The main goal is to let third-party code call WanGP directly, keep the last loaded model alive across requests, receive structured progress updates, and still capture the same stdout/stderr output that would normally go to the console.

## Quick Start

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
- `WanGPSession.submit(source) -> SessionJob`
  - Starts a job from a settings dict, a manifest list, or a saved `.json` / `.zip` file.
- `WanGPSession.submit_task(settings) -> SessionJob`
  - Preferred single-task entrypoint.
- `WanGPSession.submit_manifest(settings_list) -> SessionJob`
  - Batch entrypoint for multiple tasks.
- `SessionJob.result() -> GenerationResult`
  - Waits for completion and returns a structured result object.
- `SessionJob.cancel()`
  - Requests cancellation of the active generation.

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

## Accepted Input Shapes

Relative attachment paths are normalized to absolute paths when the job is submitted.

- For direct settings dictionaries and `.json` settings files, the base is the API caller's current working directory at submit time.
- For `.zip` queue files, WanGP keeps the queue bundle behavior and resolves bundled media from the extracted queue contents.
- A few WanGP string-like fields are normalized for convenience. For example, `force_fps` may be passed as `24` or `"24"`.

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

You can pass a callback object to `init(...)` or `WanGPSession(...)`.

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
