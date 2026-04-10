import os
from multiprocessing import Process, Queue

from source.core.log import headless_logger
from source.core.constants import BYTES_PER_MB
from source.core.db.edge.request import resolve_edge_auth_token
from source.runtime.worker.guardian import guardian_main

# Maximum number of log messages buffered in the heartbeat guardian queue
HEARTBEAT_LOG_QUEUE_MAX_SIZE = 1000


def resolve_guardian_auth_token(
    *,
    explicit_token: str | None = None,
    runtime_config=None,
    auth_scope: str = "worker",
) -> str:
    """Resolve the auth token used by the guardian heartbeat path."""
    if explicit_token:
        return explicit_token
    token = resolve_edge_auth_token(scope=auth_scope, runtime_config=runtime_config)
    if not token:
        raise RuntimeError("No guardian auth token available")
    return token


def start_heartbeat_guardian_process(worker_id: str, supabase_url: str, supabase_key: str):
    """
    Start bulletproof heartbeat guardian as a separate process.
    """
    log_queue = Queue(maxsize=HEARTBEAT_LOG_QUEUE_MAX_SIZE)

    config = {
        'worker_id': worker_id,
        'worker_pid': os.getpid(),
        'db_url': supabase_url,
        'api_key': supabase_key
    }

    guardian = Process(
        target=guardian_main,
        args=(worker_id, os.getpid(), log_queue, config),
        name=f'guardian-{worker_id}',
        daemon=True
    )
    guardian.start()

    headless_logger.essential(f"✅ Heartbeat guardian started: PID {guardian.pid} monitoring worker PID {os.getpid()}")

    return guardian, log_queue


def get_gpu_memory_usage():
    """
    Get GPU memory usage in MB.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / BYTES_PER_MB
            allocated = torch.cuda.memory_allocated(0) / BYTES_PER_MB
            return int(total), int(allocated)
    except (RuntimeError, ValueError, OSError) as e:
        headless_logger.debug(f"Failed to get GPU memory usage: {e}")

    return None, None
