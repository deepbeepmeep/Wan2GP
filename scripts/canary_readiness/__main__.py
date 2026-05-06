from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.canary_readiness.package import (
    CANARY_READINESS_PREFIX,
    DEFAULT_OUTPUT_DIR,
    build_package,
    write_package,
)
from scripts.dual_run_compare.reporting import REPORTS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Sprint 10 canary readiness package.")
    parser.add_argument("--source-report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--package-id")
    return parser


def _default_package_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{CANARY_READINESS_PREFIX}{timestamp}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_id = args.package_id or _default_package_id()
    package = build_package(
        package_id=package_id,
        source_report_dir=args.source_report_dir,
        output_dir=args.output_dir,
    )
    json_path, markdown_path = write_package(package, output_dir=args.output_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return int(package["exit_policy"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

