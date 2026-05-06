# Sprint 10 Canary Rollback Draft - Worker

## Purpose

Rollback worker runtime behavior to stable WGP routing and capture placeholder-safe worker evidence. This is a draft PR artifact for the worker repo; real live/staging task IDs and secrets must be supplied at execution time, not committed.

Required placeholders:

```bash
export SUPABASE_URL="<SUPABASE_URL>"
export SUPABASE_SERVICE_ROLE_KEY="<SUPABASE_SERVICE_ROLE_KEY>"
export WORKER_ID="<worker-id-placeholder>"
export STABLE_SELECTOR_NAMESPACE="production"
export STABLE_SELECTOR_VERSION="<stable-selector-version-or-empty>"
export STABLE_WORKER_POOL="gpu-wgp-production"
export ROUTE_KEYS_JSON='["travel_segment","image__model-z_image_turbo","video__model-wan_t2v"]'
```

## Rollback Steps

### 1. Start or restart workers with the stable WGP contract

```bash
export REIGH_BACKEND=wgp
export WORKER_BACKEND=wgp
export REIGH_WORKER_PROFILE=1
export WGP_PROFILE=1
export REIGH_WORKER_POOL="$STABLE_WORKER_POOL"
export WORKER_POOL="$STABLE_WORKER_POOL"
export REIGH_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export ROUTE_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export REIGH_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
export ROUTE_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
```

```bash
python -m source.runtime.worker.server
```

Expected evidence:

```json
{
  "worker_backend": "wgp",
  "worker_profile": "1",
  "worker_pool": "gpu-wgp-production",
  "selector_namespace": "production",
  "selector_version": "<stable-selector-version-or-empty>",
  "claim_enabled": true
}
```

### 2. Verify stable task-count and claim request shape

```bash
python - <<'PY'
from source.core.db.task_claim import check_task_counts_supabase

ok = check_task_counts_supabase()
print({"task_counts_gate_ok": bool(ok)})
PY
```

```bash
python - <<'PY'
from source.runtime.worker.health_labels import queryable_telemetry_labels

labels = queryable_telemetry_labels("<worker-id-placeholder>")
print(labels)
PY
```

Expected evidence:

```json
{
  "task_counts_gate_ok": true,
  "health_labels": {
    "selector_namespace": "production",
    "selector_version": "<stable-selector-version-or-empty>",
    "worker_backend": "wgp",
    "worker_pool": "gpu-wgp-production"
  }
}
```

### 3. Verify no VibeComfy canary worker remains claimable

```bash
curl --fail-with-body -sS \
  "$SUPABASE_URL/rest/v1/workers?worker_pool=neq.$STABLE_WORKER_POOL&worker_backend=eq.vibecomfy&status=in.(ready,idle,active)&select=id,status,metadata" \
  -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
| tee /tmp/canary-rollback-vibecomfy-workers.json
```

Expected evidence:

```json
{
  "vibecomfy_canary_claimable_workers": 0,
  "allowed_remaining_workers": "active in-flight drain only"
}
```

### 4. Capture route parity evidence after rollback

```bash
python - <<'PY' | tee /tmp/canary-rollback-runtime-metrics.json
import json
from pathlib import Path
from scripts.dual_run_compare.runtime_metrics import normalize_live_runtime_observations

observations = json.loads(Path("/tmp/redacted-live-evidence.json").read_text())
print(json.dumps(normalize_live_runtime_observations(observations), indent=2, sort_keys=True))
PY
```

Expected evidence:

```json
{
  "routes": {
    "<route_key>": {
      "status": "green",
      "worker_backend": "wgp",
      "selector_namespace": "production",
      "source_ref": {
        "kind": "redacted_live_observation",
        "path": "/tmp/redacted-live-evidence.json"
      }
    }
  }
}
```

## Sprint 11A Validation Harness Rollback

Sprint 11A did not change worker production runtime behavior because no
route-specific live proof existed and no production patch boundary was selected.
Only the worker live-validation harness gained route-specific execution controls
for `z_image_turbo`. If that validation harness blocks rollback operations or
confuses canary evidence collection, revert the harness changes while keeping
the production worker route contract on WGP.

Harness rollback target:

```bash
git revert <commit-that-added-sprint-11a-live-test-route-harness>
```

Stable rollback environment for reruns:

```bash
export REIGH_BACKEND=wgp
export WORKER_BACKEND=wgp
export REIGH_WORKER_PROFILE=1
export WGP_PROFILE=1
export REIGH_WORKER_POOL="$STABLE_WORKER_POOL"
export WORKER_POOL="$STABLE_WORKER_POOL"
export REIGH_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export ROUTE_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export REIGH_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
export ROUTE_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
export REIGH_WORKER_CONTRACT_VERSION=1
```

Route-specific WGP rollback rerun for the only current VibeComfy-supported
candidate:

```bash
python -m scripts.live_test.main \
  --variant fresh \
  --wgp-rollback \
  --selector-namespace "$STABLE_SELECTOR_NAMESPACE" \
  --selector-version "$STABLE_SELECTOR_VERSION" \
  --worker-contract-version 1 \
  --worker-profile default \
  --route-key z_image_turbo
```

Expected evidence:

```json
{
  "route_key": "z_image_turbo",
  "worker_backend": "wgp",
  "worker_pool": "gpu-wgp-production",
  "selector_namespace": "production",
  "selector_version": "<stable-selector-version-or-empty>",
  "worker_contract_version": 1,
  "wgp_selectable": true
}
```

Route-stale behavior remains the orchestrator-owned drain-first policy: after
the emergency selector flip back to WGP, active stale-route workers may drain
in-flight tasks, but idle stale-route VibeComfy workers should be terminated or
left unclaimable. Do not use aggregate task counts as route rollback proof;
use route-specific claim/completion evidence or selected-pool route totals.

## Alert Runbook Anchor

`#output-divergence`: if `canary_output_divergence` remains nonzero, keep VibeComfy canary scaled to zero, keep stable WGP selected, and attach the failed route key plus redacted source refs to the readiness package.
