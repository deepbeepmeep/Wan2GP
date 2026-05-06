from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


DEFAULT_MAX_AGE = timedelta(hours=24)
REQUIRED_FIELDS = (
    "environment",
    "observed_at",
    "task_id",
    "task_type",
    "route_key",
    "selector_namespace",
    "selector_version",
    "status",
    "completion_evidence",
    "billing_evidence",
    "source_ref",
    "redaction",
)
ALLOWED_ENVIRONMENTS = frozenset({"staging", "canary", "live", "production", "prod"})
REDACTION_STATUSES = frozenset({"redacted", "none_needed"})
SECRET_MARKERS = (
    "authorization:",
    "bearer ",
    "service_role",
    "service-role",
    "supabase_service_role",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "github_pat_",
)
SECRET_PREFIXES = ("sk-proj-", "sk-live-", "sk-ant-")


class EvidenceValidationError(ValueError):
    """Raised when live/staging evidence cannot satisfy canary readiness gates."""


def validate_observation(
    observation: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, Any]:
    errors = observation_errors(observation, now=now, max_age=max_age)
    if errors:
        raise EvidenceValidationError("; ".join(errors))
    return dict(observation)


def validate_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[dict[str, Any]]:
    return [validate_observation(observation, now=now, max_age=max_age) for observation in observations]


def observation_errors(
    observation: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if not _present(observation.get(field)):
            errors.append(f"missing required field: {field}")

    if not _present(observation.get("worker_backend")) and not _present(observation.get("runtime")):
        errors.append("missing required field: worker_backend or runtime")

    environment = observation.get("environment")
    if environment == "fixture" or observation.get("fixture_only") is True:
        errors.append("fixture-only observations cannot satisfy live/staging evidence")
    elif isinstance(environment, str) and environment not in ALLOWED_ENVIRONMENTS:
        errors.append(f"unsupported environment: {environment}")

    observed_at = _parse_observed_at(observation.get("observed_at"))
    if observed_at is None:
        errors.append("observed_at must be an ISO-8601 timestamp")
    else:
        reference_time = now or datetime.now(timezone.utc)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        if observed_at < reference_time - max_age:
            errors.append("observation is stale")
        if observed_at > reference_time + timedelta(minutes=5):
            errors.append("observed_at is too far in the future")

    if not _evidence_mapping(observation.get("completion_evidence")):
        errors.append("completion_evidence must be a non-empty object")
    if not _evidence_mapping(observation.get("billing_evidence")):
        errors.append("billing_evidence must be a non-empty object")
    if not _evidence_mapping(observation.get("source_ref")):
        errors.append("source_ref must be a non-empty object")

    redaction = observation.get("redaction")
    if not isinstance(redaction, Mapping):
        errors.append("redaction must be an object")
    else:
        status = redaction.get("status")
        secret_scan = redaction.get("secret_scan")
        if status not in REDACTION_STATUSES:
            errors.append("redaction.status must be redacted or none_needed")
        if secret_scan != "passed":
            errors.append("redaction.secret_scan must be passed")

    secret_paths = _secret_paths(observation)
    if secret_paths:
        errors.append("unredacted secret-bearing observations are not allowed: " + ", ".join(secret_paths))

    return errors


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != {} and value != []


def _parse_observed_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _evidence_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _secret_paths(value: Any, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_path = f"{path}.{key}"
            if _secretish(str(key)) and _present(child) and not _redacted_placeholder(child):
                paths.append(key_path)
            paths.extend(_secret_paths(child, key_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            paths.extend(_secret_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str) and _secretish(value):
        paths.append(path)
    return paths


def _secretish(value: str) -> bool:
    normalized = value.lower()
    return normalized.startswith(SECRET_PREFIXES) or any(marker in normalized for marker in SECRET_MARKERS)


def _redacted_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"[redacted]", "<redacted>", "redacted", "***"}
