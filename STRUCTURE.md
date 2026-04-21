# Project Structure

Queue-based video generation system built on [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

## Core Files

```
├── pyproject.toml               # Canonical uv project metadata
├── uv.lock                      # Locked Python 3.10 dependency graph
├── requirements.txt             # Rollback ballast during uv rollout
├── worker.py                    # Main worker - polls DB, claims tasks, routes to handlers
├── headless_model_management.py # Thin facade re-exporting HeadlessTaskQueue
├── headless_wgp.py              # Thin facade re-exporting WanOrchestrator
├── heartbeat_guardian.py        # Process-level heartbeat monitoring
```

## Runtime Model

- Runtime commands now assume `uv sync --locked --python 3.10` from the repo root on every launch.
- `uv run --python 3.10 ...` is the supported execution path for the worker and debug tools.
- Local Linux installs are validated against Ubuntu 22.04 as the baseline. Ubuntu 24.04+ requires Python 3.10 packages from deadsnakes before sync can succeed.
- `Wan2GP/` is a git submodule pinned to `banodoco/Wan2GP`. Any upstream requirement change should land via a pointer bump plus the matching root metadata update before regenerating `uv.lock`.
- Rollback stays repo-based: there is no runtime pip fallback on the uv branch. First-migration failures restore the timestamped `venv.pre-uv-*` or `.venv.pre-uv-*` backup, while release rollback means reverting to the pre-uv revision that still uses `requirements.txt`.

## source/ Package

Organized into 5 top-level groups:

### source/core/ — Infrastructure
- `db/` — Database operations (Supabase): config, task claim/completion/status/dependencies, edge helpers
- `log/` — Structured logging: core logger, database log sink, safe formatting, timing
- `params/` — Typed parameter handling: TaskConfig, LoRAConfig, phase_config, structure guidance
- `constants.py` — Shared constants
- `platform_utils.py` — Platform-specific utilities (ALSA suppression, headless env)

### source/media/ — Media Processing
- `video/` — Video operations: ffmpeg, crossfade, color matching, frame extraction, hires, VACE frames
- `structure/` — Structure guidance video: generation, download, preprocessing, segment tracking
- `visualization/` — Debug visualizations: comparison grids, layouts, timeline
- `vlm/` — Vision-language model: prompt generation, image prep

### source/models/ — Model Integration
- `wgp/` — WanOrchestrator: parameter resolution, model ops, LoRA setup, generation strategies, patches
- `comfy/` — ComfyUI integration
- `lora/` — LoRA download, path resolution, collision handling
- `model_handlers/` — Task-type-specific handlers (QwenHandler for 7 Qwen image task types)

### source/task_handlers/ — Task Orchestration
- `tasks/` — Task routing: registry, type definitions, DB→GenerationTask conversion
- `queue/` — HeadlessTaskQueue: task queue, task processor, download ops, lifecycle
- `travel/` — Multi-image travel: orchestrator, segment processor, SVI config, chaining, stitch
- `join/` — Video clip joining: orchestrator, generation, final stitch, VLM enhancement
- `worker/` — Worker utilities: heartbeat, fatal error handling, debug printing
- Top-level handlers: `edit_video_orchestrator.py`, `magic_edit.py`, `inpaint_frames.py`, `extract_frame.py`, `rife_interpolate.py`, `create_visualization.py`

### source/utils/ — Shared Utilities
- `download_utils.py` — File/image download helpers
- `frame_utils.py` — Frame manipulation (pose, keypoint detection)
- `resolution_utils.py` — Resolution parsing and grid snapping
- `prompt_utils.py` — Prompt validation and formatting
- `mask_utils.py` — Mask generation for VACE
- `output_path_utils.py` — Output directory management
- `lora_validation.py` — LoRA file validation

### source/db_operations.py — Database Facade
Thin facade with `_ConfigProxy` for runtime config propagation. Delegates to `source/core/db/`.

## External

- `Wan2GP/` — Git submodule pointing at `banodoco/Wan2GP`; edit via fork PR + pointer bump, not in-place
- `debug/` — CLI tool for investigating tasks/workers (`python -m debug`)
- `scripts/` — Standalone utilities (test task creation, LoRA rank conversion)

## Data Flow

```
DB → worker.py → TaskRegistry → HeadlessTaskQueue → WanOrchestrator → wgp.py → Files
```

1. **worker.py** polls tasks table, claims work
2. **TaskRegistry** routes task_type to the appropriate handler
3. **HeadlessTaskQueue** manages model loading, LoRA downloads, and task processing
4. **WanOrchestrator** maps parameters, calls WGP
5. **wgp.py** performs generation, writes to outputs/
6. Results flow back through the chain to update DB

## Database

| Column | Purpose |
|--------|---------|
| `id` | UUID primary key |
| `task_type` | e.g., `vace`, `travel_orchestrator`, `t2v` |
| `dependant_on` | Optional FK forming execution DAG |
| `params` | JSON payload |
| `status` | `Queued` → `In Progress` → `Complete`/`Failed` |
| `output_location` | Final output path/URL |

## Configuration

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Service role key |
| `--db-type` | `sqlite` (default) or `supabase` |
| `--debug` | Keep temp folders, extra logs |
