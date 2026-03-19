"""Post-generation output extraction, error reporting, and memory monitoring.

Handles extracting the output path from WGP state, logging captured
stdout/stderr/Python logs on failure, and reporting memory usage.
"""

from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from source.core.log import generation_logger
from source.models.wgp.error_extraction import extract_wgp_error, LOG_TAIL_MAX_CHARS


@dataclass(frozen=True)
class GenerationOutcome:
    ok: bool
    output_path: Optional[str]
    error_code: Optional[str]
    error_detail: Optional[str]


class GenerationFailure(RuntimeError):
    def __init__(self, error_code: str, error_detail: Optional[str] = None):
        self.error_code = error_code
        self.error_detail = error_detail
        message = error_detail or error_code
        super().__init__(message)


def build_generation_outcome(
    *,
    state: dict,
    captured_stdout: Any,
    captured_stderr: Any,
    captured_logs: Optional[Deque[Dict[str, str]]] = None,
) -> GenerationOutcome:
    try:
        file_list = state["gen"]["file_list"]
    except (KeyError, TypeError):
        return GenerationOutcome(
            ok=False,
            output_path=None,
            error_code="invalid_state",
            error_detail="Could not retrieve output path from state",
        )

    if file_list:
        return GenerationOutcome(
            ok=True,
            output_path=file_list[-1],
            error_code=None,
            error_detail=None,
        )

    stdout_content = captured_stdout.getvalue() if captured_stdout is not None else ""
    stderr_content = captured_stderr.getvalue() if captured_stderr is not None else ""
    actual_error = extract_wgp_error(stdout_content, stderr_content)
    if actual_error:
        return GenerationOutcome(
            ok=False,
            output_path=None,
            error_code="parsed_wgp_error",
            error_detail=actual_error,
        )

    return GenerationOutcome(
        ok=False,
        output_path=None,
        error_code="no_output",
        error_detail="No output generated",
    )


def extract_output_path(
    state: dict,
    model_type_desc: str,
    captured_stdout: Any,
    captured_stderr: Any,
    captured_logs: Deque[Dict[str, str]],
) -> Optional[str]:
    """Extract the generated output path from WGP state.

    Raises RuntimeError if no output was generated (after logging diagnostics).

    Returns:
        Path to the generated output file, or None if state is malformed.
    """
    try:
        outcome = build_generation_outcome(
            state=state,
            captured_stdout=captured_stdout,
            captured_stderr=captured_stderr,
            captured_logs=captured_logs,
        )
        if outcome.ok:
            output_path = outcome.output_path
            generation_logger.success(f"{model_type_desc} generation completed")
            generation_logger.essential(f"Output saved to: {output_path}")
            return output_path

        if outcome.error_code == "invalid_state":
            generation_logger.warning(outcome.error_detail)
            return None

        # No output produced -- log diagnostics
        generation_logger.warning(
            f"{model_type_desc} generation completed but no output path found in file_list"
        )
        log_captured_output(captured_stdout, captured_stderr, captured_logs)

        if outcome.error_code == "parsed_wgp_error":
            generation_logger.error(
                f"[WGP_ERROR] Extracted error from WGP output: {outcome.error_detail}"
            )
            raise GenerationFailure(outcome.error_code, f"WGP generation failed: {outcome.error_detail}")

        raise GenerationFailure(outcome.error_code or "no_output", outcome.error_detail)

    except GenerationFailure:
        raise
    except (KeyError, TypeError, IndexError) as e:
        # Only catch unexpected errors accessing state
        generation_logger.warning(f"Could not retrieve output path from state: {e}")
        return None


def log_captured_output(
    captured_stdout: Any,
    captured_stderr: Any,
    captured_logs: Deque[Dict[str, str]],
) -> None:
    """Log captured Python logs and stdout/stderr for debugging."""
    # Log captured Python logs from diffusers/transformers/torch
    if captured_logs:
        all_logs = list(captured_logs)
        error_logs = [log for log in all_logs if log['level'] in ('ERROR', 'CRITICAL', 'WARNING')]
        if error_logs:
            log_summary = '\n'.join(
                [f"  [{log['level']}] {log['name']}: {log['message'][:200]}" for log in error_logs[-30:]]
            )
            generation_logger.error(
                f"[WGP_PYLOG] Python errors/warnings ({len(error_logs)} total):\n{log_summary}"
            )

        recent_logs = all_logs[-50:]
        log_summary = '\n'.join(
            [f"  [{log['level']}] {log['name']}: {log['message'][:150]}" for log in recent_logs]
        )
        generation_logger.error(
            f"[WGP_PYLOG] Recent Python logs ({len(all_logs)} total):\n{log_summary}"
        )

    if captured_stderr is not None:
        stderr_content = captured_stderr.getvalue()
        if stderr_content:
            stderr_tail = (
                stderr_content[-LOG_TAIL_MAX_CHARS:]
                if len(stderr_content) > LOG_TAIL_MAX_CHARS
                else stderr_content
            )
            generation_logger.error(
                f"[WGP_STDERR] Generation produced no output. Captured stderr:\n{stderr_tail}"
            )

    if captured_stdout is not None:
        stdout_content = captured_stdout.getvalue()
        if stdout_content:
            stdout_lower = stdout_content.lower()
            if any(
                err in stdout_lower
                for err in ('error', 'exception', 'traceback', 'failed', 'cuda', 'oom', 'out of memory')
            ):
                stdout_tail = (
                    stdout_content[-LOG_TAIL_MAX_CHARS:]
                    if len(stdout_content) > LOG_TAIL_MAX_CHARS
                    else stdout_content
                )
                generation_logger.error(
                    f"[WGP_STDOUT] Error patterns found in stdout:\n{stdout_tail}"
                )


def log_memory_stats() -> None:
    """Log post-generation RAM and VRAM usage."""
    try:
        import torch
        import psutil

        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent

        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_percent = (vram_reserved / vram_total) * 100
            generation_logger.essential(
                f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%) | "
                f"VRAM: {vram_reserved:.1f}GB / {vram_total:.1f}GB ({vram_percent:.0f}%) "
                f"[Allocated: {vram_allocated:.1f}GB]"
            )
        else:
            generation_logger.essential(
                f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.0f}%)"
            )
    except (RuntimeError, OSError, ImportError) as e:
        generation_logger.debug(f"Could not retrieve memory stats: {e}")


__all__ = [
    "GenerationFailure",
    "GenerationOutcome",
    "build_generation_outcome",
    "extract_output_path",
    "log_captured_output",
    "log_memory_stats",
]
