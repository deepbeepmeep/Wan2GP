# Wan2GP Rebase Runbook

## 1. Cadence

Rebase `banodoco/Wan2GP` monthly per SD-007. Also trigger an out-of-band rebase when upstream ships a new minor version that touches `models/ltx2/`, shared runtime code, plugin surfaces, or any Sprint 3/Sprint 4 carry path.

## 2. Pre-Flight

Start every cycle by recording or verifying a checkpoint tag in `reigh-worker` using the `wan2gp-mig-sprint-N-baseline` naming pattern. Sprint 4 already has `wan2gp-mig-sprint-4-baseline` per the migration brief.

Before touching the fork or the submodule pointer:

- Confirm the current triage source of truth in [docs/wan2gp-triage.csv](wan2gp-triage.csv).
- Review the current upstream draft inventory in [scripts/sprint4/upstream_prs/](../scripts/sprint4/upstream_prs/).
- Note the Sprint 4 audit finding that the original Section 2b missing-item list became a planning smell: the current submodule already contains `res2s.py`, `ltx2_22B_distilled_1_1.json`, prompt-enhancer work, and `self_refiner`, so any new upstream work should start from a fresh diff review rather than from the stale design-doc assumptions.

## 3. Fork Rebase Steps

1. Update the long-running fork from `deepbeepmeep/Wan2GP` head in the fork repository, not inside `reigh-worker`.
2. Review the hot zones from the triage CSV before merging, especially `models/ltx2/`, shared runtime, plugin integration, and preset/config drifts.
3. Resolve conflicts in the fork while preserving explicitly carried banodoco behavior such as `clear_conditioning` and any still-local Sprint carry items.
4. Refresh the Sprint 4 upstream draft inventory if the literal `upstreamable` CSV themes or file memberships changed.

## 4. Submodule Bump in reigh-worker

1. Update the `Wan2GP/` submodule pointer to the reviewed fork commit.
2. Confirm the mount path stays exactly `reigh-worker/Wan2GP/`.
3. Re-check that `source/runtime/wgp_ports/vendor_imports.py` callers still rely on the absolute `Wan2GP/` path rather than a repo-root fallback.
4. Review `source/models/wgp/wgp_patches.py` for any local carry that should move into the fork or be re-validated after the bump.

## 5. Post-Bump Verification

Run the full Sprint 4 verification set before asking the orchestrator to commit or push anything.

### Full 8-file Regression Suite

`uv run --python 3.10 pytest tests/test_travel_guidance_config.py tests/test_travel_ltx_vpt.py tests/test_ltx_hybrid_travel.py tests/test_ltx_hybrid_vgkfi.py tests/test_travel_payload_contracts.py tests/test_travel_orchestrator_terminal_gating.py tests/test_travel_guidance_ic_lora_override.py tests/test_task_registry_ic_lora_dedup.py`

If `uv` hits the darwin-arm64 `decord==0.6.0` wheel gap, rerun the same files with `.venv/bin/python -m pytest ...`.

### Strict Path-Contract Smoke

`.venv/bin/python -c "import os, sys; from source.core.runtime_paths import ensure_wan2gp_on_path, get_wan2gp_path; ensure_wan2gp_on_path(); assert os.path.abspath(str(get_wan2gp_path())) in [os.path.abspath(p) for p in sys.path]; assert os.path.isfile('Wan2GP/models/qwen/qwen_handler.py'); assert os.path.isfile('Wan2GP/models/qwen/qwen_main.py'); assert os.path.isfile('Wan2GP/shared/utils/loras_mutipliers.py'); assert os.path.isfile('Wan2GP/shared/utils/self_refiner.py'); assert os.path.isfile('Wan2GP/models/ltx2/ltx_core/tools.py'); print('path-contract smoke OK')"`

### Patch-Lifecycle Smoke, Executor Form

`uv run --python 3.10 pytest tests/test_wgp_patch_context_contracts.py`

The required executor-side anchor is `tests/test_wgp_patch_context_contracts.py::test_apply_all_wgp_patches_records_ltx2_runtime_fork_markers`, which is the executor-safe stub form of the Sprint 4 bootstrap verification.

### Patch-Lifecycle Smoke, Orchestrator Form

`scripts/sprint4/patch_lifecycle_smoke.py` is the real bootstrap check. It must be run on the orchestrator or another Linux environment with the full Wan2GP bootstrap dependencies available. Do not treat the darwin-arm64 executor as authoritative for this script.

## 6. Rollback

Rollback is orchestrator-only.

1. Reset the fork or submodule pointer back to the last approved baseline.
2. Restore `reigh-worker` to the matching `wan2gp-mig-sprint-N-baseline` tag.
3. If the failure is isolated to runtime patch consolidation, clear or roll back the registered runtime patch context before retrying.
4. Re-run the strict path-contract smoke and the 8-file regression suite before reopening the bump.

## 7. Drift CI Interpretation

`.github/workflows/wan2gp-drift.yml` is report-only. Non-zero drift does not fail the job by itself; only Git command failures should fail the run.

Read the `$GITHUB_STEP_SUMMARY` output in this order:

- Confirm the pinned submodule SHA matches the reviewed fork commit.
- Compare the upstream head SHA and the `Commits behind upstream` count to decide whether the monthly cadence slipped.
- Review the `Dirty tracked files excluding wgp_config.json` line. `source/models/wgp/orchestrator.py` rewrites `Wan2GP/wgp_config.json` during worker boot, so the workflow deliberately excludes that file to avoid false-positive drift alerts.
- If any other tracked file appears in the dirty-file list, treat that as actionable drift and investigate before the next cadence window closes.

## 8. Sprint 5 Followup

Multi-context runtime-patch isolation, meaning `context_id` semantics beyond registry bookkeeping, is **not** covered by Sprint 4. This is the deferred item from the iter-3 FLAG-007 / scope-v3 decision.

If multi-context production runtime patches become necessary, follow-up work must do one of these:

- Make `context_id` drive a per-context runtime view, likely via copy-on-write against `runtime.models_def`.
- Forbid same-model runtime patches under differing contexts and fail fast when a second context attempts it.

Concrete reproducer outline for Sprint 5:

1. Bootstrap a shared runtime with one `ltx2_*` model definition in `runtime.models_def`.
2. Apply `apply_runtime_model_definition_patch(..., context_id='ctx.a')` for that model.
3. Apply the same runtime patch again under `context_id='ctx.b'` against the same runtime object.
4. Roll back only `ctx.b`.
5. Observe that `ctx.b` captured a snapshot of already-patched state, so its rollback closure restores the patched value rather than the true pre-patch value. That is why registry bookkeeping alone is insufficient.

Do not add same-model cross-context runtime patching to production until one of the two resolution paths above is implemented.
