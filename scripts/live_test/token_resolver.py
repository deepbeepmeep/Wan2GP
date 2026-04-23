"""Token resolution helpers for the live-test harness."""

from __future__ import annotations


class TokenResolutionError(RuntimeError):
    """Raised when a PAT cannot be resolved to exactly one user."""


def resolve_token_to_user_id(db, token: str) -> str:
    """Resolve a PAT to the owning user ID and require a single match."""
    token_value = str(token).strip()
    if not token_value:
        raise TokenResolutionError("Token is empty")

    result = (
        db.supabase.table("user_api_tokens")
        .select("user_id")
        .eq("token", token_value)
        .execute()
    )
    rows = list(result.data or [])
    if len(rows) != 1:
        raise TokenResolutionError(
            f"Expected exactly one user_api_tokens row for the provided token, found {len(rows)}"
        )

    user_id = rows[0].get("user_id")
    if not user_id:
        raise TokenResolutionError("Matched token row did not include user_id")
    return str(user_id)


__all__ = ["TokenResolutionError", "resolve_token_to_user_id"]
