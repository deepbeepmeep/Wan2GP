from __future__ import annotations

import base64
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Verified queue host surface from:
# - source/task_handlers/queue/task_processor.py:60-216 and 307-570
# - source/task_handlers/queue/task_queue.py:101-467
# Preview queue host members touched directly on the exercised path:
# queue_lock, current_task, current_model, logger, stats, wan_dir, wgp, wgp_state,
# debug_mode, task_history, main_output_dir, running, shutdown_event, orchestrator,
# _ensure_orchestrator(), _convert_to_wgp_task(), _switch_model(),
# _execute_generation(), _model_supports_vace(), _is_single_image_task(),
# _convert_single_frame_video_to_png(), _cleanup_memory_after_task(),
# submit_task(), get_task_status(), start().

MINIMAL_MP4_BYTES = base64.b64decode(
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMUbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAA"
    "AAAD6AAAACgAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAj90cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAAB"
    "AAAAAAAAACgAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAA"
    "ABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAAoAAAAAAABAAAAAAG3bWRpYQAAACBtZGhk"
    "AAAAAAAAAAAAAAAAAAAyAAAAAgBVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRl"
    "b0hhbmRsZXIAAAABYm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAA"
    "AQAAAAx1cmwgAAAAAQAAASJzdGJsAAAAvnN0c2QAAAAAAAAAAQAAAK5hdmMxAAAAAAAAAAEAAAAAAAAA"
    "AAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABFUxhdmM2MS4xOS4xMDEgbGlieDI2NAAAAAAAAAAAAAAA"
    "GP//AAAANGF2Y0MBZAAK/+EAF2dkAAqs2V7ARAAAAwAEAAADAMg8SJZYAQAGaOvjyyLA/fj4AAAAABBw"
    "YXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAAi3QAAAAAAAAABhzdHRzAAAAAAAAAAEAAAABAAACAAAAABxz"
    "dHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAUc3RzegAAAAAAAALKAAAAAQAAABRzdGNvAAAAAAAAAAEA"
    "AANEAAAAYXVkdGEAAABZbWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAs"
    "aWxzdAAAACSpdG9vAAAAHGRhdGEAAAABAAAAAExhdmY2MS43LjEwMAAAAAhmcmVlAAAC0m1kYXQAAAKu"
    "BgX//6rcRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY0IHIzMTA4IDMxZTE5ZjkgLSBILjI2NC9N"
    "UEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDIzIC0gaHR0cDovL3d3dy52aWRlb2xhbi5v"
    "cmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9"
    "MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEg"
    "bWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIx"
    "LDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhy"
    "ZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlf"
    "Y29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0x"
    "IGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUw"
    "IGtleWludF9taW49MjUgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCBy"
    "Yz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00"
    "IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAUZYiEACv//tjn8yyn9Nq78KzEmbE="
)


def _build_wgp_stub() -> types.ModuleType:
    module = types.ModuleType("wgp")
    module.attention_mode = None
    module.compile = None
    module.force_profile_no = None
    module.default_profile = None
    module.vae_config = None
    module.boost = None
    module.transformer_quantization = None
    module.transformer_dtype_policy = None
    module.text_encoder_quantization = None
    module.server_config = {
        "vae_precision": None,
        "mixed_precision": None,
        "preload_model_policy": [],
        "preload_in_VRAM": None,
        "transformer_types": [],
    }
    module.models_def = {}
    module.wan_model = None
    module.transformer_type = None
    module.reload_needed = False
    module.get_model_def = lambda model_name: module.models_def.get(model_name)
    module.get_default_settings = lambda model_name: {}
    module.get_lora_dir = lambda model_name: "/tmp/preview"
    module.get_model_name = lambda model_name: model_name
    module.get_model_min_frames_and_step = lambda model_name: (1, 4, 16)
    module.get_model_fps = lambda model_name: 16
    module.get_model_recursive_prop = lambda model_name, prop_name, return_list=False: [] if return_list else None
    module.parse_loras_multipliers = lambda *args, **kwargs: ([], [])
    module.preparse_loras_multipliers = lambda *args, **kwargs: ([], [])
    module.setup_loras = lambda *args, **kwargs: None
    return module


def _build_torch_stub() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda *_args, **_kwargs: 0.0,
        memory_reserved=lambda *_args, **_kwargs: 0.0,
        empty_cache=lambda: None,
        get_device_name=lambda *_args, **_kwargs: "Preview GPU (spoofed)",
        get_device_properties=lambda *_args, **_kwargs: SimpleNamespace(
            name="Preview GPU (spoofed)",
            total_memory=0,
            major=0,
            minor=0,
            multi_processor_count=0,
        ),
        device_count=lambda: 0,
        current_device=lambda: 0,
        get_device_capability=lambda *_args, **_kwargs: (8, 0),
    )
    module.version = SimpleNamespace(cuda=None)
    module.backends = SimpleNamespace(
        cudnn=SimpleNamespace(
            is_available=lambda: False,
            version=lambda: None,
        )
    )
    module.tensor = lambda *args, **kwargs: SimpleNamespace(device=kwargs.get("device", "cpu"))
    return module


def _build_huggingface_stub() -> types.ModuleType:
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = lambda *args, **kwargs: "/tmp/fake.safetensors"
    return module


def inject_module_stubs() -> dict[str, types.ModuleType]:
    stubs = {
        "wgp": _build_wgp_stub(),
        "torch": _build_torch_stub(),
        "huggingface_hub": _build_huggingface_stub(),
    }
    sys.modules.update(stubs)
    return stubs


def ensure_samples_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "samples" / "test.mp4"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists() or sample_path.read_bytes() != MINIMAL_MP4_BYTES:
        sample_path.write_bytes(MINIMAL_MP4_BYTES)
    return str(sample_path.resolve())


class SpoofQueue:
    def __init__(self, wan_dir: str, db_runtime=None):
        inject_module_stubs()
        ensure_samples_dir()
        os.environ.setdefault("HEADLESS_WAN2GP_SMOKE", "1")
        os.environ.setdefault("HEADLESS_WAN2GP_FORCE_CPU", "1")

        self.queue_lock = threading.RLock()
        self.current_task = None
        self.current_model = ""
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "model_switches": 0,
            "total_generation_time": 0.0,
        }

        from source.core.log.queue_runtime import queue_logger

        self.logger = queue_logger
        self.wan_dir = str(Path(wan_dir).resolve())
        self.main_output_dir = str((Path(self.wan_dir).parent / "outputs").resolve())
        self.wgp = sys.modules["wgp"]
        self.wgp_state = {"gen": {"file_list": []}}
        self.debug_mode = True
        self.task_history: dict[str, Any] = {}
        self.running = False
        self.shutdown_event = threading.Event()
        self.start_time = time.time()
        self.orchestrator = None
        self._orchestrator_init_attempted = False
        self.db_runtime = db_runtime
        self._ensure_orchestrator()

    def _ensure_orchestrator(self):
        if self.orchestrator is not None:
            return self.orchestrator
        if self._orchestrator_init_attempted:
            return self.orchestrator

        from source.models.wgp.orchestrator import WanOrchestrator
        from source.runtime.process_globals import temporary_process_globals

        self._orchestrator_init_attempted = True
        with temporary_process_globals(
            cwd=self.wan_dir,
            argv=["preview-worker.py"],
            prepend_sys_path=self.wan_dir,
        ):
            self.orchestrator = WanOrchestrator(self.wan_dir, main_output_dir=self.main_output_dir)
        self.wgp_state = self.orchestrator.state
        return self.orchestrator

    def _convert_to_wgp_task(self, task):
        from source.task_handlers.queue.download_ops import convert_to_wgp_task_impl

        return convert_to_wgp_task_impl(self, task)

    def _switch_model(self, model_key: str, worker_name: str):
        from source.models.wgp.model_ops import load_model_impl

        self._ensure_orchestrator()
        switched = load_model_impl(self.orchestrator, model_key)
        if switched:
            self.stats["model_switches"] += 1
        self.current_model = model_key
        return switched

    def _execute_generation(self, task, worker_name: str):
        from source.task_handlers.queue.task_processor import execute_generation_impl

        return execute_generation_impl(self, task, worker_name)

    def _model_supports_vace(self, model_key: str) -> bool:
        orchestrator = self._ensure_orchestrator()
        try:
            if hasattr(orchestrator, "is_model_vace"):
                return bool(orchestrator.is_model_vace(model_key))
        except Exception:
            pass
        return "vace" in str(model_key).lower()

    def _is_single_image_task(self, task) -> bool:
        return task.parameters.get("video_length", 0) == 1

    def _convert_single_frame_video_to_png(self, task, path: str, worker_name: str) -> str:
        _ = task, worker_name
        return path

    def _cleanup_memory_after_task(self, task_id: str):
        _ = task_id
        return None

    def submit_task(self, task) -> str:
        from source.task_handlers.queue.task_processor import process_task_impl

        self.stats["tasks_submitted"] += 1
        self.task_history[task.id] = task
        process_task_impl(self, task, "preview-worker")
        self.task_history[task.id] = task
        return task.id

    def get_task_status(self, task_id: str):
        return self.task_history.get(task_id)

    def start(self):
        return None
