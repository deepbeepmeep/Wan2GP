from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


THRESHOLD_VERSION = "0B-2026-05-05"
SCHEMA_VERSION = 1

EXPECTED_METRIC_KEYS: tuple[str, ...] = (
    "image_phash_normalized_hamming",
    "image_ssim",
    "image_pixel_dimensions",
    "image_format_container",
    "video_frame_count",
    "video_phash_mean",
    "video_phash_p95",
    "video_duration_ms",
    "video_fps",
    "video_audio_duration_ms",
    "latency_p95_wall_clock_ratio",
    "vram_peak_ratio",
    "error_oom_count",
    "canary_output_divergence_rate",
)

APPROVED_CALIBRATION_STATUSES: frozenset[str] = frozenset(
    {
        "green",
        "pending_calibration",
        "deferred_pending_sprint_0c_disk",
        "wgp_only",
        "owner_deferred",
    }
)

REQUIRED_METRIC_FIELDS: tuple[str, ...] = (
    "class",
    "metric",
    "comparator",
    "threshold",
    "unit",
    "failure_action",
)

DEFAULT_PATH = Path(__file__).with_name("migration-thresholds.yaml")


class ThresholdValidationError(ValueError):
    """Raised when the Sprint 0B threshold contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class RouteThreshold:
    route_key: str
    route: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Any]]
    calibration_status: str


@dataclass(frozen=True)
class Thresholds:
    path: Path
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None, *, strict: bool = False) -> "Thresholds":
        threshold_path = Path(path) if path is not None else DEFAULT_PATH
        try:
            loaded = yaml.safe_load(threshold_path.read_text())
        except FileNotFoundError as exc:
            raise ThresholdValidationError(f"threshold YAML not found: {threshold_path}") from exc
        if not isinstance(loaded, Mapping):
            raise ThresholdValidationError("threshold YAML must contain a mapping at the root")
        thresholds = cls(path=threshold_path, raw=loaded)
        thresholds.validate(strict=strict)
        return thresholds

    @property
    def metrics(self) -> Mapping[str, Mapping[str, Any]]:
        metrics = self.raw.get("defaults")
        return metrics if isinstance(metrics, Mapping) else {}

    @property
    def routes(self) -> Mapping[str, Mapping[str, Any]]:
        routes = self.raw.get("routes")
        return routes if isinstance(routes, Mapping) else {}

    @property
    def route_defaults(self) -> Mapping[str, Any]:
        defaults = self.raw.get("route_defaults")
        return defaults if isinstance(defaults, Mapping) else {}

    def for_route(self, route_key: str) -> RouteThreshold:
        if route_key not in self.routes:
            raise KeyError(f"unknown route key: {route_key}")
        route = self.routes[route_key]
        overrides = route.get("thresholds_override") or {}
        if not isinstance(overrides, Mapping):
            raise ThresholdValidationError(f"route {route_key} thresholds_override must be a mapping")

        merged = {key: dict(value) for key, value in self.metrics.items()}
        for metric_key, override in overrides.items():
            if metric_key not in merged:
                raise ThresholdValidationError(f"route {route_key} has unknown threshold override: {metric_key}")
            if not isinstance(override, Mapping):
                raise ThresholdValidationError(f"route {route_key} override {metric_key} must be a mapping")
            merged[metric_key].update(dict(override))

        status = route.get("calibration_status", self.route_defaults.get("calibration_status"))
        if not isinstance(status, str):
            raise ThresholdValidationError(f"route {route_key} calibration_status must be a string")
        return RouteThreshold(
            route_key=route_key,
            route=route,
            metrics=merged,
            calibration_status=status,
        )

    def validate(self, *, strict: bool = False) -> None:
        errors: list[str] = []
        if self.raw.get("version") != THRESHOLD_VERSION:
            errors.append(f"version must be {THRESHOLD_VERSION}")
        if self.raw.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        if "metrics" in self.raw:
            errors.append("metrics is a legacy threshold schema key; use metric_keys plus defaults")

        enum_values = self.raw.get("calibration_status_enum")
        if not isinstance(enum_values, list):
            errors.append("calibration_status_enum must be a list")
        elif set(enum_values) != APPROVED_CALIBRATION_STATUSES:
            errors.append("calibration_status_enum must exactly match the approved Sprint 0B statuses")

        errors.extend(self._validate_metrics())
        errors.extend(self._validate_routes(strict=strict))
        if strict:
            errors.extend(self._validate_manifest_sync())

        if errors:
            raise ThresholdValidationError("\n".join(errors))

    def _validate_metrics(self) -> list[str]:
        errors: list[str] = []
        metric_keys = self.raw.get("metric_keys")
        if not isinstance(metric_keys, list):
            errors.append("metric_keys must be a list")
        elif tuple(metric_keys) != EXPECTED_METRIC_KEYS:
            missing_from_list = [key for key in EXPECTED_METRIC_KEYS if key not in metric_keys]
            extra_in_list = [key for key in metric_keys if key not in EXPECTED_METRIC_KEYS]
            if missing_from_list:
                errors.append(f"metric_keys missing rows: {', '.join(missing_from_list)}")
            if extra_in_list:
                errors.append(f"metric_keys unknown rows: {', '.join(extra_in_list)}")
            if not missing_from_list and not extra_in_list:
                errors.append("metric_keys must preserve the approved row order")
        if not isinstance(self.raw.get("defaults"), Mapping):
            return errors + ["defaults must be a mapping"]
        metric_keys = tuple(self.metrics.keys())
        missing = [key for key in EXPECTED_METRIC_KEYS if key not in self.metrics]
        extra = [key for key in metric_keys if key not in EXPECTED_METRIC_KEYS]
        if missing:
            errors.append(f"defaults missing metric rows: {', '.join(missing)}")
        if extra:
            errors.append(f"defaults unknown metric rows: {', '.join(extra)}")
        for key in EXPECTED_METRIC_KEYS:
            metric = self.metrics.get(key)
            if not isinstance(metric, Mapping):
                continue
            missing_fields = [field for field in REQUIRED_METRIC_FIELDS if field not in metric]
            if missing_fields:
                errors.append(f"metric {key} missing fields: {', '.join(missing_fields)}")
        return errors

    def _validate_routes(self, *, strict: bool) -> list[str]:
        errors: list[str] = []
        default_status = self.route_defaults.get("calibration_status")
        if default_status not in APPROVED_CALIBRATION_STATUSES:
            errors.append(f"route_defaults calibration_status is not approved: {default_status}")
        if not isinstance(self.raw.get("routes"), Mapping):
            return errors + ["routes must be a mapping"]
        for route_key, route in self.routes.items():
            if not isinstance(route, Mapping):
                errors.append(f"route {route_key} must be a mapping")
                continue
            status = route.get("calibration_status", default_status)
            if status not in APPROVED_CALIBRATION_STATUSES:
                errors.append(f"route {route_key} has invalid calibration_status: {status}")
            overrides = route.get("thresholds_override", {})
            if overrides is None:
                overrides = {}
            if not isinstance(overrides, Mapping):
                errors.append(f"route {route_key} thresholds_override must be a mapping")
                continue
            unknown_overrides = [key for key in overrides if key not in EXPECTED_METRIC_KEYS]
            if unknown_overrides:
                errors.append(f"route {route_key} has unknown threshold overrides: {', '.join(unknown_overrides)}")
            if strict:
                required = ("cohort", "route_key_mode", "task_type", "media_class", "source_fixture_status")
                missing = [field for field in required if field not in route]
                if missing:
                    errors.append(f"route {route_key} missing fields: {', '.join(missing)}")
        return errors

    def _validate_manifest_sync(self) -> list[str]:
        golden_dir = self.path.parent / "golden"
        if not golden_dir.exists():
            return []
        manifest_dirs = {
            path.parent.name
            for path in golden_dir.glob("*/manifest.json")
            if path.is_file()
        }
        if not manifest_dirs:
            return []
        route_keys = set(self.routes)
        errors: list[str] = []
        missing_manifests = sorted(route_keys - manifest_dirs)
        orphan_manifests = sorted(manifest_dirs - route_keys)
        if missing_manifests:
            errors.append(f"routes missing golden manifests: {', '.join(missing_manifests)}")
        if orphan_manifests:
            errors.append(f"golden manifests without YAML routes: {', '.join(orphan_manifests)}")
        return errors
