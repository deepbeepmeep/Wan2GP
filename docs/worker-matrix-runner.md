# Worker Matrix Runner

`scripts/run_worker_matrix.py` runs a curated set of worker-facing smoke cases and writes artifacts for each case under `artifacts/worker-matrix/`.

## What It Covers

- Real `worker.py -> process_single_task()` execution with real task routing and parameter conversion
- For direct-queue generation cases, a fake immediate queue captures the submitted generation task after real payload conversion, then returns a synthetic output. This catches routing and parameter-shape regressions without requiring GPU execution.
- Local-only handler cases (e.g. frame extraction, visualization) can also be added and will run end-to-end without a fake queue.
- Per-case logs, inputs, outputs, and summaries

## Basic Usage

Run the default suite:

```bash
python scripts/run_worker_matrix.py
```

List available cases:

```bash
python scripts/run_worker_matrix.py --list-cases
```

Run only selected cases:

```bash
python scripts/run_worker_matrix.py --case-id qwen_image_style_db_task --case-id z_image_turbo_db_task
```

Filter by tag, task type, or family:

```bash
python scripts/run_worker_matrix.py --tag qwen
python scripts/run_worker_matrix.py --task-type extract_frame
python scripts/run_worker_matrix.py --family direct_queue
```

Rerun only failed cases from a previous summary:

```bash
python scripts/run_worker_matrix.py --failed-from artifacts/worker-matrix/<run>/summary.json
```

Stop on first failure:

```bash
python scripts/run_worker_matrix.py --stop-on-failure
```

Include non-default cases:

```bash
python scripts/run_worker_matrix.py --all-cases
```

Override manifest or output directory:

```bash
python scripts/run_worker_matrix.py --manifest path/to/custom.json --output-dir path/to/output
```

## Case Definitions and DB Snapshots

Cases are defined in `scripts/worker_matrix_cases.json`. Each case references a builder function in the script, a task type, a family, tags, and an `expected_outcome` (`"success"` or `"failure"`). Cases can also declare `requires` (e.g. `ffmpeg`) and are auto-skipped when requirements are missing.

Several cases load frozen DB task payloads from `scripts/worker_matrix_db_snapshots.json`. Snapshots contain real parameter shapes captured from production tasks, with `__FIXTURE_PATH__` and `__FIXTURE_URL__` placeholder tokens that get substituted with local test fixtures at runtime.

## Artifacts

Each run creates a timestamped directory:

```text
artifacts/worker-matrix/<timestamp>/
```

Top-level files:

- `selected_cases.json`
- `summary.json`
- `summary.md`
- `failures.txt`
- `rerun_failed.sh`

Per-case files:

- `case_definition.json`
- `prepared_input.json`
- `captured_generation_task.json` for fake-queue direct cases
- `worker.log`
- `result.json`
- `traceback.txt` when a case crashes
- generated local outputs

## Current Locked Cases

- `vace_cocktail_2_2_basic`
- `individual_travel_segment_wan22_i2v_baseline_2_2_2`
- `ltx2_22b_basic`
- `ltx2_22b_distilled_basic`
- `z_image_turbo_db_task`
- `z_image_turbo_i2i_db_task`
- `qwen_image_style_db_task`
- `qwen_image_edit_db_task`

## Notes

- The runner is intentionally serial. It is optimizing for readable failures, not throughput.
- The locked suite is centered on DB-backed Wan 2.2, Qwen, and Z-Image tasks, plus an LTX 2.3 representative `ltx2_22B` case.
- Every case writes a live `worker.log` plus explicit `[WORKER_MATRIX][CASE_START]`, `[CASE_END]`, and `[CASE_EXCEPTION]` markers so failures are easy to spot during and after the run.
- A case passes when its `actual_outcome` matches its `expected_outcome`. This lets you lock in expected-failure cases as regression guards.
