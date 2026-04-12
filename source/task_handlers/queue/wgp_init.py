"""
WGP/CUDA initialization mixin for HeadlessTaskQueue.

Handles CUDA warmup, diagnostics, and lazy WanOrchestrator initialization.
Separated from the main queue coordinator to isolate the complex
startup/initialization logic.
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass

from source.core.log import is_debug_enabled
from source.runtime.process_globals import get_runtime_context, temporary_process_globals


@dataclass(frozen=True)
class OrchestratorBootstrapState:
    state: str
    retry_after_seconds: float = 0.0
    message: str | None = None


def get_orchestrator_bootstrap_state(host) -> OrchestratorBootstrapState:
    if getattr(host, "orchestrator", None) is not None:
        return OrchestratorBootstrapState(state="ready")

    now = time.monotonic()
    next_retry = float(getattr(host, "_orchestrator_next_retry_monotonic", 0.0) or 0.0)
    failures = int(getattr(host, "_orchestrator_consecutive_failures", 0) or 0)
    fatal_threshold = int(getattr(host, "_orchestrator_fatal_failure_threshold", 5) or 5)
    last_error = getattr(host, "_orchestrator_last_error", None)

    if failures >= fatal_threshold and last_error:
        return OrchestratorBootstrapState(
            state="failed_fatal",
            retry_after_seconds=max(0.0, next_retry - now),
            message=str(last_error),
        )

    if next_retry > now:
        return OrchestratorBootstrapState(
            state="cooldown",
            retry_after_seconds=max(0.0, next_retry - now),
            message=str(last_error) if last_error else None,
        )

    if (getattr(host, "_orchestrator_init_attempted", False) or failures > 0) and last_error:
        return OrchestratorBootstrapState(state="failed_transient", message=str(last_error))

    return OrchestratorBootstrapState(state="not_started")


def _attempt_orchestrator_bootstrap(host) -> None:
    try:
        host._ensure_orchestrator()
        host._orchestrator_consecutive_failures = 0
        host._orchestrator_last_error = None
        host._orchestrator_next_retry_monotonic = 0.0
    except Exception as exc:
        failures = int(getattr(host, "_orchestrator_consecutive_failures", 0) or 0) + 1
        backoff = float(getattr(host, "_orchestrator_retry_backoff_seconds", 15.0) or 15.0)
        host._orchestrator_consecutive_failures = failures
        host._orchestrator_last_error = str(exc)
        host._orchestrator_next_retry_monotonic = time.monotonic() + backoff
        raise


def ensure_orchestrator_state(host) -> OrchestratorBootstrapState:
    current = get_orchestrator_bootstrap_state(host)
    if current.state in {"ready", "cooldown", "failed_fatal"}:
        return current

    try:
        _attempt_orchestrator_bootstrap(host)
    except Exception:
        pass
    return get_orchestrator_bootstrap_state(host)


def ensure_orchestrator_bootstrap_state(host) -> OrchestratorBootstrapState:
    return ensure_orchestrator_state(host)


def ensure_orchestrator(host):
    state = ensure_orchestrator_state(host)
    if state.state == "ready":
        return host.orchestrator
    if state.state == "cooldown":
        raise RuntimeError("cooldown active")
    raise RuntimeError(state.message or "orchestrator bootstrap failed")


class WgpInitMixin:
    """Mixin providing CUDA warmup and WanOrchestrator lazy initialization."""

    def _ensure_orchestrator(self):
        """
        Lazily initialize orchestrator on first use to avoid CUDA init during queue setup.

        The orchestrator imports wgp, which triggers deep module imports (models/wan/modules/t5.py)
        that call torch.cuda.current_device() at class definition time. We defer this until
        the first task is actually processed, when CUDA is guaranteed to be ready.
        """
        if self.orchestrator is not None:
            return  # Already initialized

        if self._orchestrator_init_attempted:
            raise RuntimeError("Orchestrator initialization failed previously")

        self._orchestrator_init_attempted = True

        try:
            if is_debug_enabled():
                self.logger.debug_anomaly("LAZY_INIT", "Initializing WanOrchestrator (first use)...")
                self.logger.debug_anomaly("LAZY_INIT", "Warming up CUDA before importing wgp...")

            # Warm up CUDA before importing wgp (upstream T5EncoderModel has torch.cuda.current_device()
            # as a default arg, which is evaluated at module import time)
            import torch

            if is_debug_enabled():
                self.logger.debug_anomaly("LAZY_INIT", f"CUDA availability before import: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                if is_debug_enabled():
                    _log_cuda_available_diagnostics(self.logger, torch)
            else:
                if is_debug_enabled():
                    _log_cuda_unavailable_diagnostics(self.logger, torch)

            self._import_and_create_orchestrator()

            if is_debug_enabled():
                self.logger.debug_anomaly("LAZY_INIT", "WanOrchestrator initialized successfully")

            # Now that orchestrator exists, complete wgp integration
            self._init_wgp_integration()

        except Exception as e:
            # Always log orchestrator init failures - this is critical for debugging!
            self.logger.error(f"[LAZY_INIT] Failed to initialize WanOrchestrator: {e}")
            if is_debug_enabled():
                self.logger.error(f"[LAZY_INIT] Traceback:\n{traceback.format_exc()}")
            raise

    def _import_and_create_orchestrator(self):
        """
        Import WanOrchestrator and create instance, handling sys.argv and cwd protection.

        wgp.py parses sys.argv and uses relative paths, so we must protect both
        before import.
        """
        if is_debug_enabled():
            self.logger.debug_anomaly("LAZY_INIT", "Importing WanOrchestrator (this imports wgp and model modules)...")

        runtime_context = get_runtime_context(
            "queue.orchestrator_import",
            wan_root=self.wan_dir,
            default_argv=["headless_model_management.py"],
            require_cwd=True,
        )
        runtime_context.prepare()

        if is_debug_enabled():
            self.logger.debug_anomaly("LAZY_INIT", f"Runtime context prepared for Wan2GP directory: {self.wan_dir}")

        with temporary_process_globals(
            cwd=self.wan_dir,
            argv=["headless_model_management.py"],
            prepend_sys_path=self.wan_dir,
        ):
            actual_cwd = os.getcwd()
            if actual_cwd != self.wan_dir:
                raise RuntimeError(
                    f"Directory change failed! Expected {self.wan_dir}, got {actual_cwd}"
                )

            if not os.path.isdir("defaults"):
                raise RuntimeError(
                    f"defaults/ directory not found in {actual_cwd}. "
                    f"Cannot proceed without model definitions!"
                )

            if is_debug_enabled():
                self.logger.debug_anomaly("LAZY_INIT", "Runtime context active, importing WanOrchestrator...")

            from source.models.wgp.orchestrator import WanOrchestrator

            if is_debug_enabled():
                try:
                    from mmgp import offload
                    offload.default_verboseLevel = 2
                    self.logger.debug_anomaly("LAZY_INIT", "Set offload.default_verboseLevel=2 for debug logging")
                except ImportError:
                    pass

            self.orchestrator = WanOrchestrator(self.wan_dir, main_output_dir=self.main_output_dir)

    def _init_wgp_integration(self):
        """
        Initialize integration with wgp.py's existing systems.

        This reuses wgp.py's state management, queue handling, and model persistence
        rather than reimplementing it.

        Called after orchestrator is lazily initialized.
        """
        if self.orchestrator is None:
            self.logger.debug_anomaly("WGP_INIT", "Skipping wgp integration - orchestrator not initialized yet")
            return

        # Core integration: reuse orchestrator's state management
        self.wgp_state = self.orchestrator.state

        self.logger.debug("WGP integration initialized")


def _log_cuda_available_diagnostics(logger, torch):
    """Log detailed CUDA diagnostics when CUDA is available."""
    try:
        device_count = torch.cuda.device_count()
        current_dev = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_dev) if device_count > current_dev else "unknown"
        test_tensor = torch.tensor([1.0], device='cuda')
        logger.debug(
            f"[LAZY_INIT] CUDA warmup validated: devices={device_count}, current_device={current_dev}, "
            f"device_name={device_name}, torch_cuda={torch.version.cuda}, tensor_device={test_tensor.device}"
        )

    except Exception as e:
        logger.error(f"[CUDA_DEBUG] Error during CUDA diagnostics: {e}\n{traceback.format_exc()}")
        raise


def _log_cuda_unavailable_diagnostics(logger, torch):
    """Log diagnostics when CUDA is not available to help debug the issue."""
    logger.warning("[CUDA_DEBUG] torch.cuda.is_available() returned False")
    logger.warning("[CUDA_DEBUG] Checking why CUDA is not available...")

    driver_version = "unknown"
    device_count = "unknown"
    device_names = []

    # Try to import pynvml for driver info
    try:
        import pynvml
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            device_names.append(name.decode() if isinstance(name, bytes) else str(name))
    except Exception as e:
        logger.warning(f"[CUDA_DEBUG] Could not get NVML info: {e}")

    logger.debug(
        f"[LAZY_INIT] CUDA unavailable summary: torch_cuda={torch.version.cuda}, "
        f"cudnn={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}, "
        f"driver={driver_version}, nvml_devices={device_count}, device_names={device_names}"
    )
