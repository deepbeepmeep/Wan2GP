# Sprint 11B Canary Rollback Runbook - Worker

## Purpose

Rollback a Sprint 11B route cohort to the stable WGP worker contract and capture redacted evidence for the readiness package. Roll back by route cohort and route key, not by a broad backend switch.

Current mechanical state for this runbook:

- Active live cohort: `UNKNOWN`.
- Selector namespace/version: `UNKNOWN`.
- Hold windows for Cohorts A/B/E: `UNKNOWN`.
- Dashboard validation prerequisite: `NO_GO` until `reigh-worker-orchestrator/scripts/dashboard.py` imports without `NameError: name 'DatabaseClient' is not defined`.
- Shadow isolation: `UNKNOWN`; output-divergence auto-rollback stays disabled unless isolated shadow evidence proves no completion, billing, upload, or user-visible side effects.
- Rollback PR references: `ROLLBACK_PR_WORKER_PLACEHOLDER`, replace when mechanically verified.

## Required Placeholders

```bash
export SUPABASE_URL="<SUPABASE_URL>"
export SUPABASE_SERVICE_ROLE_KEY="<SUPABASE_SERVICE_ROLE_KEY>"
export STABLE_SELECTOR_NAMESPACE="production"
export STABLE_SELECTOR_VERSION="<stable-selector-version-or-empty>"
export STABLE_WORKER_BACKEND="wgp"
export STABLE_WORKER_PROFILE="1"
export STABLE_WORKER_POOL="gpu-wgp-production"
export CANARY_SELECTOR_NAMESPACE="<canary-selector-namespace-or-UNKNOWN>"
export CANARY_SELECTOR_VERSION="<canary-selector-version-or-UNKNOWN>"
export CANARY_WORKER_POOL="<gpu-vibecomfy-canary-pool-or-UNKNOWN>"
export ROLLBACK_ROUTE_KEYS_JSON='["<route-key-1>","<route-key-2>"]'
export ROLLBACK_PR_WORKER="ROLLBACK_PR_WORKER_PLACEHOLDER"
```

Use the mechanically verified route universe from `reigh-worker/scripts/dual_run_compare/migration-thresholds.yaml` (`version: 0B-2026-05-05`) when filling `ROLLBACK_ROUTE_KEYS_JSON`. If the live cohort is still unknown, set the readiness evidence to `PAUSED` and keep all promoted route keys on WGP.

## Emergency Selector Flip Back To WGP

The emergency rollback is a selector flip for the affected route keys back to WGP. Do not change unrelated route cohorts and do not use an all-backends switch as rollback proof.

Patch the selector source of truth used by live ops. If it is exposed through Supabase REST, use the route-key scoped shape below; otherwise run the equivalent owner-approved selector command and preserve the same evidence fields.

```bash
python - <<'PY'
import json
import os

route_keys = json.loads(os.environ["ROLLBACK_ROUTE_KEYS_JSON"])
print(json.dumps({
    "selector_flip": {
        "route_keys": route_keys,
        "selected_backend": "wgp",
        "selector_namespace": os.environ["STABLE_SELECTOR_NAMESPACE"],
        "selector_version": os.environ["STABLE_SELECTOR_VERSION"],
        "worker_pool": os.environ["STABLE_WORKER_POOL"],
        "worker_profile": os.environ["STABLE_WORKER_PROFILE"],
        "rollback_pr": os.environ["ROLLBACK_PR_WORKER"],
    }
}, indent=2, sort_keys=True))
PY
```

Expected redacted evidence:

```json
{
  "status": "PAUSED",
  "rollback_action": "route_selector_flip_to_wgp",
  "route_keys": ["<route-key-1>"],
  "selector_namespace_before": "<canary-selector-namespace-or-UNKNOWN>",
  "selector_version_before": "<canary-selector-version-or-UNKNOWN>",
  "selector_namespace_after": "production",
  "selector_version_after": "<stable-selector-version-or-empty>",
  "selected_backend_after": "wgp",
  "worker_pool_after": "gpu-wgp-production",
  "worker_profile_after": "1",
  "rollback_pr": "ROLLBACK_PR_WORKER_PLACEHOLDER",
  "source_ref": {
    "kind": "redacted_operator_log",
    "id": "<replace-with-log-or-PR-ref>"
  }
}
```

## Restart Stable WGP Workers

Start or restart WGP workers with the stable route contract:

```bash
export REIGH_BACKEND="$STABLE_WORKER_BACKEND"
export WORKER_BACKEND="$STABLE_WORKER_BACKEND"
export REIGH_WORKER_PROFILE="$STABLE_WORKER_PROFILE"
export WGP_PROFILE="$STABLE_WORKER_PROFILE"
export REIGH_WORKER_POOL="$STABLE_WORKER_POOL"
export WORKER_POOL="$STABLE_WORKER_POOL"
export REIGH_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export ROUTE_SELECTOR_NAMESPACE="$STABLE_SELECTOR_NAMESPACE"
export REIGH_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
export ROUTE_SELECTOR_VERSION="$STABLE_SELECTOR_VERSION"
export REIGH_WORKER_CONTRACT_VERSION=1
```

```bash
python -m source.runtime.worker.server
```

Expected redacted evidence:

```json
{
  "worker_backend": "wgp",
  "worker_profile": "1",
  "worker_pool": "gpu-wgp-production",
  "selector_namespace": "production",
  "selector_version": "<stable-selector-version-or-empty>",
  "worker_contract_version": 1,
  "claim_enabled": true
}
```

## Stale VibeComfy Worker Drain

After the selector flip, active stale-route VibeComfy workers may drain only in-flight work that was already claimed before the flip. Idle or ready stale-route VibeComfy workers must be made unclaimable or terminated. Do not let stale workers claim new tasks for the rolled-back route keys.

```bash
curl --fail-with-body -sS \
  "$SUPABASE_URL/rest/v1/workers?worker_backend=eq.vibecomfy&worker_pool=eq.$CANARY_WORKER_POOL&status=in.(ready,idle,active)&select=id,status,worker_pool,worker_backend,metadata" \
  -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
| tee /tmp/sprint11b-vibecomfy-stale-workers.redacted.json
```

Expected drain evidence:

```json
{
  "route_keys": ["<route-key-1>"],
  "stale_vibecomfy_workers": {
    "active_draining": 0,
    "idle_terminated_or_unclaimable": 0,
    "new_claims_allowed": false
  },
  "drain_policy": "active pre-flip claims may finish; idle/ready stale VibeComfy workers terminate or remain unclaimable"
}
```

## Verify WGP Claim Contract

Verify task counts and worker labels after the route selector flip:

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

print(queryable_telemetry_labels("<worker-id-placeholder>"))
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

## Capture Route Evidence

Attach a redacted evidence document to the Sprint 11B readiness package. It must not include service-role keys, bearer tokens, raw production task payloads, private media URLs, RunPod tokens, or unredacted user IDs.

```json
{
  "environment": "production",
  "observed_at": "<ISO-8601>",
  "status": "PAUSED",
  "dashboard_import_prerequisite": "blocked",
  "rollback_action": "route_selector_flip_to_wgp",
  "route_keys": ["<route-key-1>"],
  "selected_backend": "wgp",
  "selector_namespace": "production",
  "selector_version": "<stable-selector-version-or-empty>",
  "worker_backend": "wgp",
  "worker_pool": "gpu-wgp-production",
  "completion_evidence": {
    "status": "UNKNOWN",
    "reason": "live completion evidence not mechanically verified"
  },
  "billing_evidence": {
    "status": "UNKNOWN",
    "reason": "live billing evidence not mechanically verified"
  },
  "source_ref": {
    "kind": "redacted_readiness_evidence",
    "path": "<readiness-manifest-path>"
  },
  "redaction": {
    "secrets": "redacted",
    "task_ids": "redacted"
  }
}
```

## Output Divergence

Alert anchor: `#output-divergence`.

If `canary_output_divergence` is nonzero, keep the affected route keys selected on WGP and attach the failed route key plus redacted source refs to the readiness package. Auto-rollback based on output divergence is allowed only when the comparison came from isolated shadow runs with no completion, billing, upload, or user-visible side effects. If isolation is absent, mark the route `NO_GO` for output-divergence auto-rollback and keep divergence sampled/offline only.
