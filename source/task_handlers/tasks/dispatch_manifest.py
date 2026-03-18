"""Canonical specialized-dispatch manifest."""

HANDLER_IMPORT_SPECS = {
    "travel_orchestrator": "source.task_handlers.travel.orchestrator.handle_travel_orchestrator_task",
    "extract_frame": "source.task_handlers.extract_frame.handle_extract_frame_task",
    "travel_segment": "source.task_handlers.travel.segments.segment_queue.handle_travel_segment_via_queue",
    "individual_travel_segment": "source.task_handlers.travel.segments.segment_queue.handle_travel_segment_via_queue",
}

__all__ = ["HANDLER_IMPORT_SPECS"]
