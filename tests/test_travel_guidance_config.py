import pytest
import ast
import sys
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from source.core.params.structure_guidance import StructureGuidanceConfig
from source.core.params.travel_guidance import TravelGuidanceConfig


VIDEO_ENTRY = {
    "path": "/tmp/guidance.mp4",
    "start_frame": 0,
    "end_frame": 16,
    "treatment": "adjust",
}


def test_parse_each_travel_guidance_kind():
    vace = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}},
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )
    ltx = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    uni3c = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "uni3c", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    none_cfg = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "none"}},
        "ltx2_22B",
    )

    assert vace.kind == "vace"
    assert ltx.kind == "ltx_control"
    assert uni3c.kind == "uni3c"
    assert none_cfg.kind == "none"


def test_parse_ltx_control_cameraman():
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": "cameraman", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )

    assert config.kind == "ltx_control"
    assert config.mode == "cameraman"


@pytest.mark.parametrize(
    ("payload", "model_name"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B"),
    ],
)
def test_model_compatibility_errors(payload, model_name):
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(payload, model_name)


@pytest.mark.parametrize("mode", ["flow", "canny", "depth", "raw"])
def test_to_structure_guidance_config_round_trip_for_vace(mode):
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "vace", "mode": mode, "videos": [VIDEO_ENTRY]}},
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )

    structure_config = config.to_structure_guidance_config()

    assert isinstance(structure_config, StructureGuidanceConfig)
    assert structure_config.target == "vace"
    expected_preprocessing = "none" if mode == "raw" else mode
    assert structure_config.preprocessing == expected_preprocessing


def test_to_structure_guidance_config_round_trip_for_uni3c():
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "uni3c", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )

    structure_config = config.to_structure_guidance_config()

    assert structure_config.target == "uni3c"
    assert structure_config.preprocessing == "none"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("pose", True), ("depth", True), ("canny", True), ("cameraman", True), ("video", False)],
)
def test_needs_ic_lora(mode, expected):
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": mode, "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    assert config.needs_ic_lora() is expected


@pytest.mark.parametrize(
    ("payload", "model_name", "expected"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2", 1.0),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", 0.5),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "video", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", 1.0),
    ],
)
def test_strength_defaults(payload, model_name, expected):
    config = TravelGuidanceConfig.from_payload(payload, model_name)
    assert config.strength == expected


def test_exclusivity_rejects_structure_guidance():
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(
            {
                "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
                "structure_guidance": {"target": "vace"},
            },
            "wan_2_2_vace_lightning_baseline_2_2_2",
        )

    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(
            {
                "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
                "structure_type": "flow",
            },
            "wan_2_2_vace_lightning_baseline_2_2_2",
        )


def test_exclusivity_allows_falsy_legacy_fields():
    """use_uni3c: false and empty legacy fields should NOT conflict."""
    config = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
            "use_uni3c": False,
            "structure_type": "",
            "structure_videos": [],
            "structure_video_path": None,
        },
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )
    assert config.kind == "vace"


@pytest.mark.parametrize(
    ("payload", "model_name"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": []}}, "wan_2_2_vace_lightning_baseline_2_2_2"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": []}}, "ltx2_22B_distilled"),
        ({"travel_guidance": {"kind": "uni3c", "videos": []}}, "ltx2_22B_distilled"),
    ],
)
def test_empty_videos_on_non_none_kind_errors(payload, model_name):
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(payload, model_name)


@pytest.mark.parametrize(
    ("payload", "model_name", "expected"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2", "flow"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", "pose"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "cameraman", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", "raw"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "video", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", "raw"),
    ],
)
def test_get_preprocessor_type(payload, model_name, expected):
    config = TravelGuidanceConfig.from_payload(payload, model_name)
    assert config.get_preprocessor_type() == expected


ROOT = Path(__file__).resolve().parents[1]
TASK_REGISTRY_PATH = ROOT / "source" / "task_handlers" / "tasks" / "task_registry.py"
WGP_PATH = ROOT / "Wan2GP" / "wgp.py"


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        self.messages.append(message)


def _load_function_from_source(path: Path, function_name: str, extra_globals: dict) -> object:
    source = path.read_text()
    module = ast.parse(source)
    function_source = None

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_source = ast.get_source_segment(source, node)
            break

    if function_source is None:
        raise AssertionError(f"Could not find function {function_name} in {path}")

    namespace = dict(extra_globals)
    exec("from __future__ import annotations\n" + function_source, namespace)
    return namespace[function_name]


@contextmanager
def _patched_modules(replacements: dict[str, object]):
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def _extract_block(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text()
    after_start = source.split(start_marker, 1)[1]
    block = start_marker + after_start.split(end_marker, 1)[0]
    return textwrap.dedent(block)


def test_svi_latent_path_resolution_downloads_expected_storage_url(tmp_path):
    logger = _LoggerStub()
    download_calls: list[tuple[str, Path, str]] = []

    def fake_download_file(url: str, dest_folder: Path, filename: str) -> bool:
        download_calls.append((url, Path(dest_folder), filename))
        (Path(dest_folder) / filename).write_bytes(b"latent")
        return True

    resolve_latent_path = _load_function_from_source(
        TASK_REGISTRY_PATH,
        "_resolve_precomputed_svi_latent_path",
        {
            "Path": Path,
            "download_file": fake_download_file,
            "task_logger": logger,
        },
    )

    config_module = types.ModuleType("source.core.db.config")
    config_module.SUPABASE_URL = "https://example.supabase.co"
    db_module = types.ModuleType("source.core.db")
    db_module.config = config_module
    core_module = types.ModuleType("source.core")
    core_module.db = db_module
    source_module = types.ModuleType("source")
    source_module.core = core_module

    with _patched_modules(
        {
            "source": source_module,
            "source.core": core_module,
            "source.core.db": db_module,
            "source.core.db.config": config_module,
        }
    ):
        result = resolve_latent_path(
            "pred-task-123",
            "https://cdn.example.com/storage/v1/object/public/image_uploads/user-42/tasks/pred-task-123/output.mp4",
            tmp_path,
            "task-1",
        )

    expected_local = tmp_path / "predecessor_latent_tail.pt"
    assert result == str(expected_local)
    assert expected_local.exists()
    assert download_calls == [
        (
            "https://example.supabase.co/storage/v1/object/public/image_uploads/user-42/tasks/pred-task-123/latent_tail.pt",
            tmp_path,
            "predecessor_latent_tail.pt",
        )
    ]


def test_svi_latent_path_resolution_skips_non_storage_predecessors(tmp_path):
    logger = _LoggerStub()
    resolve_latent_path = _load_function_from_source(
        TASK_REGISTRY_PATH,
        "_resolve_precomputed_svi_latent_path",
        {
            "Path": Path,
            "download_file": lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download_file should not run")),
            "task_logger": logger,
        },
    )

    assert resolve_latent_path("pred-task", '{"storage_path":"x"}', tmp_path, "task-1") is None
    assert resolve_latent_path("pred-task", "https://example.com/not-storage.mp4", tmp_path, "task-1") is None
    assert resolve_latent_path(None, "https://example.com/storage/v1/object/public/image_uploads/u/tasks/t/out.mp4", tmp_path, "task-1") is None


def test_svi_latent_tail_upload_uses_deterministic_sibling_file(tmp_path):
    logger = _LoggerStub()
    output_video = tmp_path / "segment_02.mp4"
    output_video.write_bytes(b"video")
    latent_tail = tmp_path / "latent_tail_segment_02.pt"
    latent_tail.write_bytes(b"tensor")

    upload_calls: list[tuple[Path, str, str]] = []

    def fake_upload_intermediate_file_to_storage(*, local_file_path: Path, task_id: str, filename: str):
        upload_calls.append((local_file_path, task_id, filename))
        return "https://storage.example/latent_tail.pt"

    upload_latent = _load_function_from_source(
        TASK_REGISTRY_PATH,
        "_upload_svi_latent_tail_if_available",
        {
            "Path": Path,
            "task_logger": logger,
        },
    )

    output_paths_module = types.ModuleType("source.utils.output_paths")
    output_paths_module.upload_intermediate_file_to_storage = fake_upload_intermediate_file_to_storage
    utils_module = types.ModuleType("source.utils")
    utils_module.output_paths = output_paths_module
    source_module = types.ModuleType("source")
    source_module.utils = utils_module

    with _patched_modules(
        {
            "source": source_module,
            "source.utils": utils_module,
            "source.utils.output_paths": output_paths_module,
        }
    ):
        result = upload_latent(str(output_video), "task-99")

    assert result == "https://storage.example/latent_tail.pt"
    assert upload_calls == [(latent_tail, "task-99", "latent_tail.pt")]


def test_apply_svi_specific_params_injects_precomputed_latent_path_into_custom_settings():
    logger = _LoggerStub()

    def fake_get_param(key, *sources, default=None, prefer_truthy=False):
        return None

    apply_svi_specific_params = _load_function_from_source(
        TASK_REGISTRY_PATH,
        "_apply_svi_specific_params",
        {
            "Path": Path,
            "_get_param": fake_get_param,
            "task_logger": logger,
        },
    )

    merge_module = types.ModuleType("source.task_handlers.travel.svi_config")
    merge_module.merge_svi_into_generation_params = lambda *args, **kwargs: None
    travel_module = types.ModuleType("source.task_handlers.travel")
    travel_module.svi_config = merge_module
    task_handlers_module = types.ModuleType("source.task_handlers")
    task_handlers_module.travel = travel_module
    source_module = types.ModuleType("source")
    source_module.task_handlers = task_handlers_module

    generation_params = {
        "video_length": 17,
        "custom_settings": {"pace": 0.5},
    }
    ctx = SimpleNamespace(segment_params={}, orchestrator_details={})
    gen = SimpleNamespace(total_frames_for_segment=17)
    image_refs = SimpleNamespace(
        active_svi_continuation=True,
        start_ref_path=None,
        prefix_video_for_source=None,
        precomputed_overlapped_latents_path="/tmp/predecessor_latent_tail.pt",
    )

    with _patched_modules(
        {
            "source": source_module,
            "source.task_handlers": task_handlers_module,
            "source.task_handlers.travel": travel_module,
            "source.task_handlers.travel.svi_config": merge_module,
        }
    ):
        apply_svi_specific_params(generation_params, ctx, gen, image_refs, "task-apply")

    assert generation_params["svi2pro"] is True
    assert generation_params["custom_settings"]["pace"] == 0.5
    assert (
        generation_params["custom_settings"]["precomputed_overlapped_latents_path"]
        == "/tmp/predecessor_latent_tail.pt"
    )


def test_wgp_precomputed_latent_load_block_accepts_valid_tensor(tmp_path):
    class FakeTensor:
        def __init__(self, shape=(1, 16, 2, 8, 8)) -> None:
            self.shape = shape
            self.device = "cpu"

        def dim(self) -> int:
            return len(self.shape)

        def to(self, device=None):
            self.device = device
            return self

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def load(path, map_location=None, weights_only=None):
            return FakeTensor()

        @staticmethod
        def is_tensor(value) -> bool:
            return isinstance(value, FakeTensor)

        @staticmethod
        def device(name: str) -> str:
            return name

    block = _extract_block(
        WGP_PATH,
        "        precomputed_overlapped_latents_path = None",
        "        context_scale = None",
    )

    wrapper_source = (
        "from __future__ import annotations\n"
        "def run(custom_settings, os, torch):\n"
        "    overlapped_latents = None\n"
        + textwrap.indent(block, "    ")
        + "\n    return overlapped_latents\n"
    )
    namespace: dict[str, object] = {}
    exec(wrapper_source, namespace)

    latent_path = tmp_path / "latent.pt"
    latent_path.write_bytes(b"ok")
    result = namespace["run"](
        {"precomputed_overlapped_latents_path": str(latent_path)},
        __import__("os"),
        FakeTorch,
    )

    assert isinstance(result, FakeTensor)
    assert result.device == "cpu"


def test_wgp_latent_tail_save_block_writes_deterministic_filename(tmp_path):
    saved_calls: list[tuple[object, str]] = []

    class FakeTensor:
        def detach(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

    class FakeTorch:
        @staticmethod
        def save(value, path):
            saved_calls.append((value, path))

    block = _extract_block(
        WGP_PATH,
        "                if (\n                    overlapped_latents is not None",
        "                end_time = time.time()",
    )
    wrapper_source = (
        "from __future__ import annotations\n"
        "def run(overlapped_latents, model_def, video_path, audio_only, is_image, gen, torch, os):\n"
        + textwrap.indent(block, "    ")
        + "\n    return gen\n"
    )
    namespace: dict[str, object] = {}
    exec(wrapper_source, namespace)

    gen = {}
    video_path = str(tmp_path / "segment_alpha.mp4")
    namespace["run"](
        FakeTensor(),
        {"svi2pro": True},
        video_path,
        False,
        False,
        gen,
        FakeTorch,
        __import__("os"),
    )

    expected_latent_path = str(tmp_path / "latent_tail_segment_alpha.pt")
    assert gen["latent_tail_path"] == expected_latent_path
    assert saved_calls and saved_calls[0][1] == expected_latent_path
