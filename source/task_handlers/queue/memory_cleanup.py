"""
Post-task memory cleanup.

Frees PyTorch CUDA caches and runs Python garbage collection after each
generation task completes, without unloading models from VRAM.
"""

from __future__ import annotations

from typing import Any

from source.core.constants import BYTES_PER_GB as BYTES_PER_GIB
from source.runtime.wgp_bridge import clear_uni3c_cache_if_unused


def cleanup_memory_after_task(queue: Any, task_id: str):
    """
    Clean up memory after task completion WITHOUT unloading models.

    This clears PyTorch caches and Python garbage to prevent memory fragmentation
    that can slow down subsequent generations. Models remain loaded in VRAM.

    Args:
        queue: The HeadlessTaskQueue instance
        task_id: ID of the completed task (for logging)
    """
    import torch
    import gc

    try:
        # Log memory BEFORE cleanup
        if torch.cuda.is_available():
            vram_allocated_before = torch.cuda.memory_allocated() / BYTES_PER_GIB
            vram_reserved_before = torch.cuda.memory_reserved() / BYTES_PER_GIB
            queue.logger.info(
                f"[MEMORY_CLEANUP] Task {task_id}: "
                f"BEFORE - VRAM allocated: {vram_allocated_before:.2f}GB, "
                f"reserved: {vram_reserved_before:.2f}GB"
            )

        # Clear uni3c cache if it wasn't used this task (preserves cache for consecutive uni3c tasks)
        clear_uni3c_cache_if_unused()

        # Clear PyTorch's CUDA cache (frees unused reserved memory, keeps models)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            queue.logger.info(f"[MEMORY_CLEANUP] Task {task_id}: Cleared CUDA cache")

        # Run Python garbage collection to free CPU memory
        collected = gc.collect()
        queue.logger.info(f"[MEMORY_CLEANUP] Task {task_id}: Garbage collected {collected} objects")

        # Log memory AFTER cleanup
        if torch.cuda.is_available():
            vram_allocated_after = torch.cuda.memory_allocated() / BYTES_PER_GIB
            vram_reserved_after = torch.cuda.memory_reserved() / BYTES_PER_GIB
            vram_freed = vram_reserved_before - vram_reserved_after

            queue.logger.info(
                f"[MEMORY_CLEANUP] Task {task_id}: "
                f"AFTER - VRAM allocated: {vram_allocated_after:.2f}GB, "
                f"reserved: {vram_reserved_after:.2f}GB"
            )

            if vram_freed > 0.01:  # Only log if freed >10MB
                queue.logger.info(
                    f"[MEMORY_CLEANUP] Task {task_id}: Freed {vram_freed:.2f}GB of reserved VRAM"
                )
            else:
                queue.logger.info(
                    f"[MEMORY_CLEANUP] Task {task_id}: No significant VRAM freed (models still loaded)"
                )

    except (RuntimeError, OSError) as e:
        queue.logger.warning(f"[MEMORY_CLEANUP] Task {task_id}: Failed to cleanup memory: {e}")
