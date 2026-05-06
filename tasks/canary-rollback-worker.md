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

## Alert Runbook Anchor

`#output-divergence`: if `canary_output_divergence` remains nonzero, keep VibeComfy canary scaled to zero, keep stable WGP selected, and attach the failed route key plus redacted source refs to the readiness package.
