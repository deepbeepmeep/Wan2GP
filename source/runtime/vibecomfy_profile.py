"""Numeric VibeComfy memory-profile protocol helpers.

This module is intentionally worker-local.  The worker runs on Python 3.10 in
some environments, while VibeComfy owns the concrete profile mapping in its own
package.  Keep this boundary to numeric CLI protocol data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROCESS_DEFAULT_PROFILE = -1
VALID_MEMORY_PROFILES = frozenset((1, 2, 3, 4, 5))


@dataclass(frozen=True)
class VibeComfyProfileProtocol:
    """One-run VibeComfy profile protocol payload."""

    memory_profile: int

    def to_cli_args(self) -> list[str]:
        return ["--memory-profile", str(self.memory_profile)]

    def to_dict(self) -> dict[str, int]:
        return {"memory_profile": self.memory_profile}


def _validate_numeric_profile(value: Any, *, field_name: str, allow_default: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer memory profile")
    if allow_default and value == PROCESS_DEFAULT_PROFILE:
        return value
    if value not in VALID_MEMORY_PROFILES:
        allowed = "1, 2, 3, 4, 5"
        if allow_default:
            allowed = f"{PROCESS_DEFAULT_PROFILE}, {allowed}"
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def resolve_memory_profile(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> int | None:
    """Resolve worker process default plus one-task override to a numeric profile.

    ``None`` means no VibeComfy profile flag should be emitted.  ``-1`` is only
    accepted for ``override_profile`` and means "use the process default".
    """

    resolved_default = None
    if process_default is not None:
        resolved_default = _validate_numeric_profile(
            process_default,
            field_name="process_default",
            allow_default=False,
        )

    if override_profile is None or override_profile == PROCESS_DEFAULT_PROFILE:
        return resolved_default

    return _validate_numeric_profile(
        override_profile,
        field_name="override_profile",
        allow_default=False,
    )


def build_profile_protocol(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> VibeComfyProfileProtocol | None:
    """Return one-run VibeComfy CLI protocol data for the selected profile."""

    resolved = resolve_memory_profile(
        process_default=process_default,
        override_profile=override_profile,
    )
    if resolved is None:
        return None
    return VibeComfyProfileProtocol(memory_profile=resolved)


def build_memory_profile_cli_args(
    *,
    process_default: int | None,
    override_profile: int | None = PROCESS_DEFAULT_PROFILE,
) -> list[str]:
    """Return ``vibecomfy run`` CLI args for the selected memory profile."""

    protocol = build_profile_protocol(
        process_default=process_default,
        override_profile=override_profile,
    )
    if protocol is None:
        return []
    return protocol.to_cli_args()
