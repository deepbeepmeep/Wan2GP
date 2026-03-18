"""Small adapter for specialized handler invocation."""

from __future__ import annotations

from source.task_handlers.tasks.dispatch_manifest import HANDLER_IMPORT_SPECS


def dispatch_specialized_task(
    *,
    task_type: str,
    context,
    handler_params: dict | None = None,
    task_id: str | None = None,
    resolve_handler,
):
    if task_type not in HANDLER_IMPORT_SPECS:
        raise ValueError(f"Unknown task type: {task_type}")

    handler = resolve_handler(task_type)
    payload = handler_params or {}

    if task_type == "travel_orchestrator":
        orch = context.for_orchestrator()
        orchestrator_task_id = task_id or payload.get("task_id") or orch.task_params_dict.get("task_id")
        return handler(
            task_params_from_db=payload if payload else getattr(orch, "task_params_dict", {}),
            main_output_dir_base=orch.main_output_dir_base,
            orchestrator_task_id_str=orchestrator_task_id,
            orchestrator_project_id=orch.project_id,
        )

    if task_type == "extract_frame":
        orch = context.for_orchestrator()
        return handler(payload, orch.main_output_dir_base, task_id)

    return handler(task_params_from_db=payload)


__all__ = ["dispatch_specialized_task"]
