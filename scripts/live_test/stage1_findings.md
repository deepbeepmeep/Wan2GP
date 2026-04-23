# Stage 1 Findings

## Live schema / auth probes

- `user_api_tokens` repo evidence points to a plain `token` column, not `jti_hash`.
  - Migration [reigh-app/supabase/migrations/20250713000008_fix_user_api_tokens_structure.sql](/Users/peteromalley/Documents/reigh-workspace/reigh-app/supabase/migrations/20250713000008_fix_user_api_tokens_structure.sql:10) drops `jti_hash`, recreates `idx_user_api_tokens_token`, and makes `verify_api_token()` match on `token`.
  - Generated types [reigh-app/src/integrations/supabase/types.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/src/integrations/supabase/types.ts:1314) expose `user_api_tokens.Row` with `token`, `user_id`, `label`, `created_at`, and no `jti_hash`.
  - Edge auth lookup [reigh-app/supabase/functions/_shared/auth.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/supabase/functions/_shared/auth.ts:136) resolves PATs with `.select("user_id").eq("token", token).single()`.
- Worker claim auth can be forced onto the PAT path while still keeping the service-role key available for heartbeat RPCs.
  - `_resolve_worker_db_client_key()` in [reigh-worker/source/runtime/worker/server.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/runtime/worker/server.py:47) returns the PAT when `WORKER_DB_CLIENT_AUTH_MODE=worker`; service-role auth is only chosen when `WORKER_DB_CLIENT_AUTH_MODE=service`.
  - Heartbeat guardian auth resolves through edge-token helpers in [reigh-worker/source/task_handlers/worker/heartbeat_utils.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/task_handlers/worker/heartbeat_utils.py:13), so injecting `SUPABASE_SERVICE_ROLE_KEY` remains useful even when task claims stay PAT-backed.
- Probe script added at [reigh-worker/scripts/live_test/stage1_probe.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/live_test/stage1_probe.py:1).
  - It loads `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` from the worker repo `.env` if needed.
  - It requires `REIGH_LIVE_TEST_TOKEN` from env, and falls back to an interactive hidden prompt if stdin is a TTY so the token does not need to be written to disk.
  - It probes live schema shape by confirming `select=token` works and `select=jti_hash` fails, resolves the live token to exactly one `user_id`, then dumps non-`live_test` queued / in-progress tasks for that user.

## No `startup_phase` writers in worker source

- `rg -n "startup_phase" reigh-worker/source` returned zero matches.
- The only writer found in this workspace is the orchestrator startup template:
  - [reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:45) documents the metadata merge.
  - [reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:87) sets `meta['startup_phase']`.
- Consequence for later tasks: readiness must key off heartbeat freshness plus dwell, not `startup_phase` or `ready_for_tasks`.
  - `ready_for_tasks` exists only as derived metadata in [reigh-worker-orchestrator/gpu_orchestrator/worker_state.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/worker_state.py:160) and is explicitly still TODO-gated for promotion logic at [worker_state.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/worker_state.py:260).

## RunPod lifecycle / caller sequence

- `RunpodLifecycleMixin.spawn_worker()` exact signature is `spawn_worker(self, worker_id: str, worker_env: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]` in [reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py:28).
  - It calls `create_pod_and_wait(...)` with `gpu_type_id`, `image_name`, `name`, `network_volume_id`, `volume_mount_path`, `disk_in_gb`, `container_disk_in_gb`, `min_vcpu_count`, `min_memory_in_gb`, `public_key_string`, `env_vars`, and `template_id` at [lifecycle.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py:92).
  - On success it returns a dict containing both `worker_id` and distinct `runpod_id` at [lifecycle.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py:113).
- `RunpodLifecycleMixin.start_worker_process()` exact signature is `start_worker_process(self, runpod_id: str, worker_id: str, has_pending_tasks: bool = False) -> bool` in [reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/lifecycle.py:148).
- Caller sequence is explicitly split across the control loop and worker-capacity phase:
  - The main control loop reaches scaling at [reigh-worker-orchestrator/gpu_orchestrator/control_loop.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control_loop.py:128) and invokes `_execute_scaling(...)` at [control_loop.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control_loop.py:131).
  - `_execute_scaling()` decides to spawn and calls `_spawn_worker(...)` at [reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py:115).
  - `_spawn_worker()` generates `worker_id = self.runpod.generate_worker_id()` at [worker_capacity.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py:188).
  - It creates the workers row first via `await self.db.create_worker_record(worker_id, self.runpod.gpu_type)` at [worker_capacity.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py:190).
  - It provisions the pod separately with `result = self.runpod.spawn_worker(worker_id)` at [worker_capacity.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py:193).
  - It only calls `self.runpod.start_worker_process(pod_id, worker_id, has_pending_tasks=(queued_count > 0))` afterward, and only when `final_status == "active"` plus `auto_start_worker_process` are both true, at [worker_capacity.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/control/phases/worker_capacity.py:233).
- Accepted-tradeoff confirmation for later `B-fresh-takeover`: `spawn_worker()` does not itself call `start_worker_process()`, and the identifiers stay distinct (`worker_id` vs `runpod_id`).

## RunPod client defaults captured from source

- `RUNPOD_GPU_TYPE` default: `"NVIDIA GeForce RTX 4090"` in [reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:28).
- `RUNPOD_WORKER_IMAGE` default: `"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"` in [client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:29).
- `RUNPOD_TEMPLATE_ID` default: `"runpod-torch-v240"` in [client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:33).
- Storage / RAM defaults:
  - Volume mount path `/workspace`, disk `50 GB`, container disk `50 GB`, min vCPU `8`, min memory `32 GB`, max task wait `5 minutes` at [client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:35).
  - Storage fallback list `["Peter", "EU-NO-1", "EU-CZ-1", "EUR-IS-1"]` at [client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:42).
  - RAM fallback tiers `[72, 60, 48, 32, 16]` with `RUNPOD_RAM_TIER_FALLBACK=true` by default at [client.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/client.py:44).

## Template install / launch facts

- Startup template install path:
  - Repo selection / clone happens at [reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:167). It prefers `/workspace/Reigh-Worker`, falls back to `/workspace/Headless-Wan2GP`, otherwise clones `https://github.com/banodoco/Reigh-Worker.git` into `/workspace/Reigh-Worker`.
  - Apt bootstrap is `apt-get update` plus `apt-get install -y python3.10-venv ffmpeg git curl wget` at [worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:200).
  - `uv` bootstrap / sync is `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `"$UV_BIN" sync --locked --python 3.10 --extra cuda124` at [worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:283).
- Direct worker launch shape is captured at [worker_startup.template.sh](/Users/peteromalley/Documents/reigh-workspace/reigh-worker-orchestrator/gpu_orchestrator/runpod/worker_startup.template.sh:355):
  - `"$UV_BIN" run --python 3.10 --extra cuda124 python worker.py`
  - `--supabase-url "$SUPABASE_URL"`
  - `--supabase-access-token "$SUPABASE_SERVICE_ROLE_KEY"`
  - `--worker "$WORKER_ID"`
  - `--wgp-profile 1`

## Canonical `run_worker.py` argv order

- The app-side launch builder is the authoritative source for the supervisor path:
  - `buildWorkerLaunchLine()` in [reigh-app/src/shared/components/SettingsModal/commandUtils.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/src/shared/components/SettingsModal/commandUtils.ts:58) emits `python run_worker.py`.
  - Flag order is pinned as `--reigh-access-token`, optional `--debug`, `--wgp-profile`, then `--idle-release-minutes` at [commandUtils.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/src/shared/components/SettingsModal/commandUtils.ts:60).
  - The Linux install/run wrapper uses `uv run --python 3.10 --extra <cuda> ...` at [commandUtils.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/src/shared/components/SettingsModal/commandUtils.ts:69) and `uv sync --locked --python 3.10 --extra <cuda>` at [commandUtils.ts](/Users/peteromalley/Documents/reigh-workspace/reigh-app/src/shared/components/SettingsModal/commandUtils.ts:91).

## LTX model pick

- Source evidence says several LTX IDs are available:
  - Task-type default `ltx2 -> ltx2_19B` is in [reigh-worker/source/task_handlers/tasks/task_types.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/task_handlers/tasks/task_types.py:101).
  - The worker matrix already exercises `ltx2_22B` and `ltx2_22B_distilled` in [reigh-worker/scripts/run_worker_matrix.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/run_worker_matrix.py:655).
  - Distilled LTX-2 models are the ones that unlock `ltx_control`, `ltx_hybrid`, and `uni3c` travel-guidance kinds per [reigh-worker/source/core/params/travel_guidance.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/core/params/travel_guidance.py:36), because non-distilled LTX returns only `{"none"}`.
  - `ltx2_22B_distilled` is a first-class model definition with `ltx2_pipeline: "distilled"` in [reigh-worker/Wan2GP/defaults/ltx2_22B_distilled.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/Wan2GP/defaults/ltx2_22B_distilled.json:3).
- Recommendation for the later travel-orchestrator LTX case: use `ltx2_22B_distilled`.
  - Rationale: it is supported by the checked-in model registry and is the safest LTX choice if the travel path ends up touching any non-`none` guidance mode.

## Fixture inventory for the later live matrix

- The five requested worker-matrix cases exist, each with `case_definition.json`, `prepared_input.json`, `captured_generation_task.json`, and `result.json`.
  - `qwen_image_basic`: [case_definition.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/qwen_image_basic/case_definition.json:1), [prepared_input.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/qwen_image_basic/prepared_input.json:1)
  - `qwen_image_edit_basic`: [case_definition.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/qwen_image_edit_basic/case_definition.json:1), [prepared_input.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/qwen_image_edit_basic/prepared_input.json:1)
  - `z_image_turbo_i2i_basic`: [case_definition.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/z_image_turbo_i2i_basic/case_definition.json:1), [prepared_input.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T135006000511Z/z_image_turbo_i2i_basic/prepared_input.json:1)
  - `qwen_image_style_db_task`: [case_definition.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T141738351426Z/qwen_image_style_db_task/case_definition.json:1), [prepared_input.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T141738351426Z/qwen_image_style_db_task/prepared_input.json:1)
  - `wan22_i2v_individual_segment`: [case_definition.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T140127343314Z/wan22_i2v_individual_segment/case_definition.json:1), [prepared_input.json](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/artifacts/worker-matrix/20260316T140127343314Z/wan22_i2v_individual_segment/prepared_input.json:1)
- Snapshot highlights from the prepared inputs:
  - `qwen_image_basic` params: `seed`, `prompt`, `resolution`, `num_inference_steps`, `task_id`.
  - `qwen_image_edit_basic` params add `image`.
  - `z_image_turbo_i2i_basic` params include `image_url` and `denoising_strength`.
  - `qwen_image_style_db_task` params include `style_reference_image`, `subject_reference_image`, `style_reference_strength`, `subject_strength`, and `subject_description`.
  - `wan22_i2v_individual_segment` params include `model_name`, `parsed_resolution_wh`, `individual_segment_params`, `orchestrator_details`, and `input_image_paths_resolved`.

## Task-type verification

- [reigh-worker/source/task_handlers/tasks/task_types.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/task_handlers/tasks/task_types.py:26) includes:
  - `z_image_turbo_i2i` in `WGP_TASK_TYPES`
  - `qwen_image_style` and `qwen_image_2512` in both `WGP_TASK_TYPES` / `DIRECT_QUEUE_TASK_TYPES`
  - `z_image_turbo_i2i` maps to the `_i2i` model entry at [task_types.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/source/task_handlers/tasks/task_types.py:116)

## Travel templates to reuse later

- `scripts/create_test_task.py` already contains reusable template payloads for:
  - `travel_orchestrator` at [reigh-worker/scripts/create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:324)
  - `qwen_image_style` at [create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:472)
  - `individual_travel_segment` via the `uni3c_basic` template at [create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:26)
- `scripts/preview/fixtures.py` already wraps those templates into preview-ready task payloads:
  - `travel_orchestrator` builder at [reigh-worker/scripts/preview/fixtures.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/preview/fixtures.py:55)
  - `individual_travel_segment` builder at [fixtures.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/preview/fixtures.py:109)
  - `qwen_image` builder at [fixtures.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/preview/fixtures.py:122)

## Default anchor-image picks for later config

- Proposed default anchor image A:
  - `https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg`
  - Reason: already used repeatedly as a travel start image in [reigh-worker/scripts/create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:84) and [create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:107), so it is already in the repo’s happy-path fixtures.
- Proposed default anchor image B:
  - `https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg`
  - Reason: paired with anchor A throughout the individual-travel templates at [create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:85) and [create_test_task.py](/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/create_test_task.py:108), making it the lowest-risk default pair for the first harness pass.

## Live probe execution note

- I attempted to run `python3 scripts/live_test/stage1_probe.py` interactively from `reigh-worker/` with the provided live token supplied through the hidden prompt.
- The script reached the network call and failed read-only with:
  - `ERROR: Network error for /rest/v1/user_api_tokens: <urlopen error [Errno 8] nodename nor servname provided, or not known>`
- Interpretation: the probe logic is in place, but this sandbox currently cannot resolve / reach Supabase, so the live `user_id` resolution and stray-task dump remain blocked at execution time rather than by missing code.
