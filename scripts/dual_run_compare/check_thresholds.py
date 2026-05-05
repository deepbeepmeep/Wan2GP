from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .thresholds import DEFAULT_PATH, ThresholdValidationError, Thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Sprint 0B migration thresholds.")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH),
        help="Path to migration-thresholds.yaml.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict route and golden-manifest synchronization validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        thresholds = Thresholds.load(Path(args.path), strict=args.strict)
    except ThresholdValidationError as exc:
        print(f"threshold validation failed:\n{exc}", file=sys.stderr)
        return 1
    print(
        f"threshold validation ok: version={thresholds.raw['version']} "
        f"schema_version={thresholds.raw['schema_version']} "
        f"metrics={len(thresholds.metrics)} routes={len(thresholds.routes)} "
        f"strict={args.strict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
