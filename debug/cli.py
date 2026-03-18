#!/usr/bin/env python3
"""
Unified Debug Tool
==================

One tool to investigate tasks, workers, and system health.
Uses system_logs as the primary data source.

Usage:
    debug.py task <task_id>             # Investigate specific task
    debug.py worker <worker_id>         # Investigate specific worker
    debug.py tasks                      # Analyze recent tasks
    debug.py workers                    # List recent workers
    debug.py health                     # System health check
    debug.py orchestrator               # Orchestrator status

Options:
    --json                              # Output as JSON
    --hours N                           # Time window in hours
    --limit N                           # Limit results
    --logs-only                         # Show only logs timeline
    --debug                             # Show debug info on errors
"""

import sys
import argparse
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass

# Add project root to path so `import debug` works.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from debug.client import DebugClient
from debug.commands.registry import COMMAND_MODULES
from debug.commands._shared import run_with_error_boundary
from debug.options import (
    coerce_config_options,
    coerce_health_options,
    coerce_runpod_options,
    coerce_storage_options,
    coerce_task_options,
    coerce_tasks_options,
    coerce_worker_options,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified debugging tool for investigating tasks, workers, and system health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Task command
    task_parser = subparsers.add_parser("task", help="Investigate specific task")
    task_parser.add_argument("task_id", help="Task ID to investigate")
    task_parser.add_argument("--json", action="store_true", help="Output as JSON")
    task_parser.add_argument("--logs-only", action="store_true", help="Show only logs timeline")
    task_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Investigate specific worker")
    worker_parser.add_argument("worker_id", help="Worker ID to investigate")
    worker_parser.add_argument("--hours", type=int, default=24, help="Hours of history (default: 24)")
    worker_parser.add_argument("--json", action="store_true", help="Output as JSON")
    worker_parser.add_argument("--logs-only", action="store_true", help="Show only logs timeline")
    worker_parser.add_argument("--startup", action="store_true", help="Show startup logs only")
    worker_parser.add_argument("--check-logging", action="store_true", help="Check if worker is logging")
    worker_parser.add_argument("--check-disk", action="store_true", help="Check disk space via SSH")
    worker_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Tasks command
    tasks_parser = subparsers.add_parser("tasks", help="Analyze recent tasks")
    tasks_parser.add_argument("--limit", type=int, default=50, help="Number of tasks (default: 50)")
    tasks_parser.add_argument("--status", help="Filter by status")
    tasks_parser.add_argument("--type", help="Filter by task type")
    tasks_parser.add_argument("--worker", help="Filter by worker ID")
    tasks_parser.add_argument("--hours", type=int, help="Filter by hours")
    tasks_parser.add_argument("--json", action="store_true", help="Output as JSON")
    tasks_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Workers command
    workers_parser = subparsers.add_parser("workers", help="List recent workers")
    workers_parser.add_argument("--hours", type=int, default=2, help="Hours of history (default: 2)")
    workers_parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    workers_parser.add_argument("--json", action="store_true", help="Output as JSON")
    workers_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Health command
    health_parser = subparsers.add_parser("health", help="System health check")
    health_parser.add_argument("--json", action="store_true", help="Output as JSON")
    health_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Orchestrator command
    orch_parser = subparsers.add_parser("orchestrator", help="Orchestrator status")
    orch_parser.add_argument("--hours", type=int, default=1, help="Hours of history (default: 1)")
    orch_parser.add_argument("--json", action="store_true", help="Output as JSON")
    orch_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show system configuration")
    config_parser.add_argument("--explain", action="store_true", help="Show detailed explanations")

    # RunPod command
    runpod_parser = subparsers.add_parser("runpod", help="Check RunPod sync status")
    runpod_parser.add_argument("--terminate", action="store_true", help="Terminate orphaned pods")
    runpod_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    # Storage command
    storage_parser = subparsers.add_parser("storage", help="Check storage volume health")
    storage_parser.add_argument("--expand", type=str, help="Expand specified storage volume")
    storage_parser.add_argument("--debug", action="store_true", help="Show debug info on errors")

    return parser


def _build_options(args):
    mapping = vars(args)
    command = mapping.get("command")
    common = {"format": "json" if mapping.get("json") else "text", "debug": bool(mapping.get("debug", False))}
    if command == "task":
        return coerce_task_options({**common, "task_id": mapping.get("task_id"), "logs_only": mapping.get("logs_only")})
    if command == "worker":
        return coerce_worker_options({
            **common,
            "worker_id": mapping.get("worker_id"),
            "hours": mapping.get("hours"),
            "logs_only": mapping.get("logs_only"),
            "startup": mapping.get("startup"),
            "check_logging": mapping.get("check_logging"),
            "check_disk": mapping.get("check_disk"),
        })
    if command == "tasks":
        return coerce_tasks_options({
            **common,
            "limit": mapping.get("limit"),
            "status": mapping.get("status"),
            "type": mapping.get("type"),
            "worker": mapping.get("worker"),
            "hours": mapping.get("hours"),
        })
    if command == "health":
        return coerce_health_options(common)
    if command == "config":
        return coerce_config_options({"explain": mapping.get("explain")})
    if command == "runpod":
        return coerce_runpod_options({"terminate": mapping.get("terminate"), "debug": mapping.get("debug")})
    if command == "storage":
        return coerce_storage_options({"expand": mapping.get("expand"), "debug": mapping.get("debug")})
    return common


def _options_dict(options):
    if is_dataclass(options):
        return asdict(options)
    return dict(options)


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Create debug client
    try:
        client = DebugClient(debug=bool(getattr(args, "debug", False)))
    except Exception as e:
        if getattr(args, "json", False):
            print(json.dumps({"error": "Failed to initialize debug client", "detail": str(e)}))
        else:
            print(f"Failed to initialize debug client: {e}")
        return 1

    options = _build_options(args)

    # Route to appropriate command handler
    try:
        module = COMMAND_MODULES[args.command]
        options_dict = _options_dict(options)
        if args.command == "task":
            module.run(client, args.task_id, options_dict)
        elif args.command == "worker":
            module.run(client, args.worker_id, options_dict)
        else:
            module.run(client, options)
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        run_with_error_boundary(lambda: (_ for _ in ()).throw(e), error_message="Command failed", json_output=getattr(args, "json", False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
