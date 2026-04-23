"""CLI entrypoint for the live worker harness."""

from __future__ import annotations

import argparse

from scripts.live_test import config
from scripts.live_test.variant_fresh import run as run_variant_fresh
from scripts.live_test.variant_update import run as run_variant_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Reigh live worker harness.")
    parser.add_argument("--variant", choices=("fresh", "update"), required=True)
    parser.add_argument("--pod-id", help="Existing RunPod pod ID for update-mode takeover.")
    parser.add_argument(
        "--spawn-takeover",
        action="store_true",
        help="Spawn a fresh orchestrator-managed pod, then take it over with the local worker branch.",
    )
    termination_group = parser.add_mutually_exclusive_group()
    termination_group.add_argument("--no-terminate", dest="no_terminate", action="store_true")
    termination_group.add_argument("--terminate", dest="no_terminate", action="store_false")
    parser.set_defaults(no_terminate=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ref", default="main", help="Branch/ref to clone for Variant Fresh.")
    parser.add_argument("--wgp-profile", type=int, default=3)
    parser.add_argument("--timeout-image", type=int, default=config.TIMEOUT_IMAGE_SEC)
    parser.add_argument(
        "--timeout-travel-segment",
        type=int,
        default=config.TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC,
    )
    parser.add_argument(
        "--timeout-travel-orchestrator",
        type=int,
        default=config.TIMEOUT_TRAVEL_ORCHESTRATOR_SEC,
    )
    parser.add_argument("--anchor-image-a", default=config.ANCHOR_IMAGE_A_URL)
    parser.add_argument("--anchor-image-b", default=config.ANCHOR_IMAGE_B_URL)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Reserved for future matrix fan-out; current harness always runs serially.",
    )
    return parser


def _finalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.variant == "fresh":
        if args.pod_id or args.spawn_takeover:
            parser.error("--pod-id/--spawn-takeover are only valid with --variant update")
        if args.no_terminate is None:
            args.no_terminate = False
        return args

    if bool(args.pod_id) == bool(args.spawn_takeover):
        parser.error("update variant requires exactly one of --pod-id or --spawn-takeover")
    if args.no_terminate is None:
        args.no_terminate = True
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = _finalize_args(parser.parse_args(argv), parser)
    if args.variant == "fresh":
        return run_variant_fresh(args)
    return run_variant_update(args)


if __name__ == "__main__":
    raise SystemExit(main())
