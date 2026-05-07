# Canary Readiness Package: canary-readiness-sprint11b-final

- Created at: `2026-05-07T00:12:48.459999+00:00`
- Source report dir: `/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/dual_run_compare/reports`
- Output dir: `/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/canary_readiness/reports`
- Source reports: 1
- Exit code: `1`

## Source Reports
- `wgp-self-repeat-0b-2026-05-05-deferral` from `/Users/peteromalley/Documents/reigh-workspace/reigh-worker/scripts/dual_run_compare/reports/wgp-self-repeat-0b-2026-05-05-deferral.json` (14 routes)

## Sections
### Prerequisite Evidence
- Status: `green`
- Evidence refs:
  - `wgp-self-repeat-0b-2026-05-05-deferral.json`
### Active Non-RayWorker Smoke
- Status: `red`
- video_enhance: observation status must be completed/green/succeeded/pass
- image-upscale: observation status must be completed/green/succeeded/pass
- animate_character: observation status must be completed/green/succeeded/pass
- flux_klein_edit: observation status must be completed/green/succeeded/pass
- Evidence refs:
  - `video_enhance:NO_GO_missing_live_video_enhance`
  - `image-upscale:NO_GO_missing_live_image-upscale`
  - `animate_character:NO_GO_missing_live_animate_character`
  - `flux_klein_edit:NO_GO_missing_live_flux_klein_edit`
### Soak
- Status: `red`
- mixed_pools: scenario status is fail
- concurrent_claims: scenario status is fail
- selector_flip_in_flight: scenario status is fail
- worker_kill_restart: scenario status is fail
- cold_warm_cache: scenario status is fail
- disk_near_full: scenario status is fail
- Evidence refs:
  - `mixed_pools:UNKNOWN`
  - `concurrent_claims:UNKNOWN`
  - `selector_flip_in_flight:UNKNOWN`
  - `worker_kill_restart:UNKNOWN`
  - `cold_warm_cache:UNKNOWN`
  - `disk_near_full:UNKNOWN`
### Dashboards
- Status: `red`
- Dashboard export validation is blocked by the Sprint 11A DatabaseClient import prerequisite.
- Static dashboard evidence is paused/no-go context only; it is not a green live dashboard export.
- Evidence refs:
  - `reigh-worker-orchestrator/docs/sprint11b-dashboard-evidence.redacted.json`
  - `reigh-worker-orchestrator/docs/sprint11b-dashboard-evidence.md`
### Alerts
- Status: `green`
- Evidence refs:
  - `reigh-worker-orchestrator/config/alerts/section11-canary.yaml`
  - `reigh-worker-orchestrator/tasks/canary-rollback-orchestrator.md#output-divergence`
  - `reigh-worker-orchestrator/tasks/canary-rollback-app.md#non-rayworker-route-smoke`
### Rollback Exercise
- Status: `red`
- Rollback exercise status must be pass.
- Evidence refs:
  - `reigh-worker-orchestrator/tasks/canary-rollback-orchestrator.md`
### Go/No-Go
- Status: `red`
- dashboard_import_prerequisite:NO_GO
- active_live_cohort:UNKNOWN
- selector_namespace_version:UNKNOWN
- hold_windows:UNKNOWN
- non_rayworker_live_observations:UNKNOWN
- shadow_isolation:UNKNOWN
- operator_signoff:UNKNOWN
- Evidence refs:
  - `sprint11b-evidence-manifest.json`
  - `reigh-worker-orchestrator/docs/sprint11b-dashboard-evidence.redacted.json`

## Go/No-Go
- Decision: `no_go`
