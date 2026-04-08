"""Idle-release configuration and tracking for the runtime worker."""

from __future__ import annotations

import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class IdleReleaseConfig:
    idle_minutes: float
    grace_seconds: float
    is_service_mode: bool


class IdleReleaseTracker:
    def __init__(
        self,
        config: IdleReleaseConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self.onboarded_at: float | None = None
        self.last_successful_empty_poll_at: float | None = None

    def mark_onboarded(self) -> None:
        self.onboarded_at = self._clock()

    def record_empty_poll(self) -> None:
        if self.last_successful_empty_poll_at is None:
            self.last_successful_empty_poll_at = self._clock()

    def record_claim(self) -> None:
        self.last_successful_empty_poll_at = None

    def should_release(self) -> bool:
        if self.config.idle_minutes <= 0:
            return False
        if self.config.is_service_mode:
            return False
        if self.onboarded_at is None:
            return False

        now = self._clock()
        if (now - self.onboarded_at) < self.config.grace_seconds:
            return False
        if self.last_successful_empty_poll_at is None:
            return False
        return (now - self.last_successful_empty_poll_at) >= (self.config.idle_minutes * 60)


def add_cli_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--idle-release-minutes", type=float, default=15.0)
    parser.add_argument("--idle-onboarding-grace-seconds", type=float, default=60.0)
    return parser


def is_service_mode(client_key) -> bool:
    auth_mode = os.environ.get("WORKER_DB_CLIENT_AUTH_MODE", "").strip().lower()
    if auth_mode == "service":
        return True
    service_keys = {
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        os.environ.get("SUPABASE_SERVICE_KEY"),
    }
    service_keys.discard(None)
    service_keys.discard("")
    return client_key in service_keys


def config_from_cli(cli_args: Namespace, *, client_key) -> IdleReleaseConfig:
    return IdleReleaseConfig(
        idle_minutes=cli_args.idle_release_minutes,
        grace_seconds=cli_args.idle_onboarding_grace_seconds,
        is_service_mode=is_service_mode(client_key),
    )
