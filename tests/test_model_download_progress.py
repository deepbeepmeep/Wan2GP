import ast
import contextlib
import io
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

utils_package = types.ModuleType("shared.utils")
utils_package.__path__ = [str(Path(__file__).parents[1] / "shared" / "utils")]
process_locks = types.ModuleType("shared.utils.process_locks")
process_locks.set_main_generation_running = lambda value: None
virtual_media = types.ModuleType("shared.utils.virtual_media")
virtual_media.get_virtual_media_vsource = lambda value: value
virtual_media.parse_virtual_media_path = lambda value: value
virtual_media.replace_virtual_media_source = lambda value, source: value
sys.modules.setdefault("shared.utils", utils_package)
sys.modules.setdefault("shared.utils.process_locks", process_locks)
sys.modules.setdefault("shared.utils.virtual_media", virtual_media)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    class _tqdm:
        def __init__(self, *args, total=None, initial=0, **kwargs):
            self.total = total
            self.n = initial
            self.disable = kwargs.get("disable", False)

        def update(self, n=1):
            if self.disable:
                return None
            self.n += n

        def close(self):
            return None

    tqdm_module = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = _tqdm
    tqdm_module.auto = tqdm_auto
    sys.modules["tqdm"] = tqdm_module
    sys.modules["tqdm.auto"] = tqdm_auto

try:
    import huggingface_hub
except ImportError:
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda **kwargs: None
    huggingface_hub.snapshot_download = lambda **kwargs: None
    sys.modules["huggingface_hub"] = huggingface_hub

from shared.api import WanGPSession
from shared.utils.download import (
    _ACTIVE_HF_PROGRESS_CLASS,
    _DownloadProgressTracker,
    create_hf_progress_class,
    create_progress_hook,
    download_file,
    process_files_def,
)


class NativeProgress:
    def __init__(self, total=None, initial=0, **kwargs):
        self.total = total
        self.n = initial

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def update(self, amount=1):
        self.n += amount


def legacy_hub_modules():
    fake_hub = types.ModuleType("huggingface_hub")
    file_download = types.ModuleType("huggingface_hub.file_download")

    def get_progress_bar_context(*, desc, log_level, total=None, initial=0, unit="B", unit_scale=True, name=None, _tqdm_bar=None):
        if _tqdm_bar is not None:
            return contextlib.nullcontext(_tqdm_bar)
        return NativeProgress(total=total, initial=initial)

    file_download._get_progress_bar_context = get_progress_bar_context
    fake_hub.file_download = file_download
    fake_hub.snapshot_download = lambda **kwargs: None
    return fake_hub, file_download


def load_functions(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {}
    exec(compile(module, path, "exec"), namespace)
    return namespace


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class DownloadProgressTests(unittest.TestCase):
    def test_model_download_status_is_compact_and_two_lines(self):
        update = SimpleNamespace(unit="bytes", speed_bps=192.1 * 1024**2, eta_seconds=25)
        compact_name = load_functions("shared/model_dropdowns.py", {"compact_name"})["compact_name"]
        formatter = load_functions("wgp.py", {"format_download_bytes", "format_download_eta", "format_model_download_status"})["format_model_download_status"]
        model_name = compact_name("Music", "Music ACE-Step v1.5 XL Turbo 4B")

        self.assertEqual(formatter(model_name, update), "Downloading ACE-Step v1.5 XL Turbo 4B\n192.1 MB/s ETA 25s")

    def test_tracker_speed_eta_throttle_and_completion(self):
        clock = FakeClock()
        updates = []
        tracker = _DownloadProgressTracker(callback=updates.append, source="http", filename="model.bin", emit_interval=0.5, clock=clock)

        first = tracker.update(10, 100)
        tracker.update(20, 100)
        clock.advance(1)
        measured = tracker.update(30, 100)
        completed = tracker.complete(100, 100)

        self.assertEqual((first.current, first.total, first.unit), (10, 100, "bytes"))
        self.assertEqual(measured.speed_bps, 20)
        self.assertEqual(measured.eta_seconds, 3.5)
        self.assertEqual([update.phase for update in updates], ["downloading", "downloading", "complete"])
        self.assertEqual(completed.current, 100)

    def test_callback_failure_is_isolated(self):
        calls = []

        def fail(update):
            calls.append(update)
            raise RuntimeError("broken UI")

        tracker = _DownloadProgressTracker(callback=fail, source="http", filename="model.bin", emit_interval=0)
        with contextlib.redirect_stderr(io.StringIO()):
            tracker.update(1, None)
            tracker.update(2, None)
        self.assertEqual(len(calls), 1)

    def test_url_hook_clamps_and_keeps_console_output(self):
        updates = []
        output = io.StringIO()
        hook = create_progress_hook("folder/model.bin", updates.append)
        with contextlib.redirect_stdout(output):
            hook(4, 30, 100)
            hook.complete()
        self.assertEqual(updates[-1].current, 100)
        self.assertEqual(updates[-1].phase, "complete")
        self.assertIn("model.bin", output.getvalue())
        self.assertIn("100.0%", output.getvalue())

    def test_direct_download_emits_complete_only_after_success(self):
        updates = []

        def retrieve(url, filename, reporthook):
            reporthook(0, 4, 4)
            Path(filename).write_bytes(b"test")
            reporthook(1, 4, 4)

        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlretrieve", side_effect=retrieve):
            with contextlib.redirect_stdout(io.StringIO()):
                download_file("https://example.com/model.bin", str(Path(directory) / "model.bin"), updates.append)
        self.assertEqual(updates[-1].phase, "complete")

    def test_huggingface_progress_handles_resume_and_early_close(self):
        updates = []
        Progress = create_hf_progress_class(updates.append, source="huggingface", filename="weights.bin", repo_id="owner/repo")
        bar = Progress(total=100, initial=20, unit="B", disable=True)
        result = bar.update(30)
        bar.close()
        bar.close()

        self.assertIsNone(result)
        self.assertEqual(updates[0].current, 20)
        self.assertEqual(updates[-1].current, 50)
        self.assertEqual(updates[-1].phase, "downloading")

    def test_huggingface_wiring_is_opt_in_and_snapshots_use_files(self):
        fake_hub = types.ModuleType("huggingface_hub")
        file_calls = []
        snapshot_calls = []
        def hf_hub_download(**kwargs):
            file_calls.append(kwargs)
            if "tqdm_class" in kwargs:
                bar = kwargs["tqdm_class"](total=4, disable=True, name="huggingface_hub.http_get")
                bar.update(4)
                bar.close()
        fake_hub.hf_hub_download = hf_hub_download
        def snapshot_download(**kwargs):
            snapshot_calls.append(kwargs)
            bar = kwargs["tqdm_class"](total=3, disable=True)
            bar.update(3)
            bar.close()
        fake_hub.snapshot_download = snapshot_download
        updates = []

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), \
             patch("shared.utils.files_locator.get_smart_download_root", return_value="models"), \
             patch("shared.utils.files_locator.locate_file", return_value=None), \
             patch("shared.utils.files_locator.locate_folder", return_value=None):
            process_files_def("owner/repo", ["folder"], [["weights.bin"]], progress_callback=updates.append)
            process_files_def("owner/repo", ["folder"], [["weights.bin"]])
            process_files_def("owner/repo", ["folder"], [[]], progress_callback=updates.append)

        self.assertIn("tqdm_class", file_calls[0])
        self.assertNotIn("tqdm_class", file_calls[1])
        self.assertEqual(updates[-1].unit, "files")
        self.assertEqual(updates[-1].phase, "complete")

    def test_huggingface_exception_does_not_emit_complete(self):
        fake_hub = types.ModuleType("huggingface_hub")
        updates = []

        def fail(**kwargs):
            bar = kwargs["tqdm_class"](total=1, disable=True)
            bar.update(1)
            bar.close()
            raise RuntimeError("download failed")

        fake_hub.hf_hub_download = fail
        fake_hub.snapshot_download = lambda **kwargs: None
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), \
             patch("shared.utils.files_locator.get_smart_download_root", return_value="models"), \
             patch("shared.utils.files_locator.locate_file", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                process_files_def("owner/repo", [""], [["weights.bin"]], progress_callback=updates.append)

        self.assertNotIn("complete", [update.phase for update in updates])

    def test_legacy_huggingface_http_emits_bytes_without_tqdm_class(self):
        fake_hub, file_download = legacy_hub_modules()
        calls = []
        updates = []

        def legacy_download(repo_id, filename, local_dir):
            calls.append((repo_id, filename, local_dir))
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=100,
                initial=0,
                name="huggingface_hub.http_get",
            ) as progress:
                progress.update(40)
                progress.update(60)

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file(
                "https://huggingface.co/DeepBeepMeep/TTS/resolve/main/ace_step_v1_5_transformer_quanto_bf16_int8.safetensors",
                str(Path(directory) / "model.safetensors"),
                updates.append,
            )

        self.assertEqual(calls[0][:2], ("DeepBeepMeep/TTS", "ace_step_v1_5_transformer_quanto_bf16_int8.safetensors"))
        self.assertEqual((updates[-1].phase, updates[-1].current, updates[-1].total, updates[-1].unit), ("complete", 100, 100, "bytes"))

    def test_legacy_huggingface_xet_uses_structured_progress(self):
        fake_hub, file_download = legacy_hub_modules()
        updates = []

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=64,
                name="huggingface_hub.xet_get",
            ) as progress:
                progress.update(64)

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertEqual((updates[-1].phase, updates[-1].current, updates[-1].total), ("complete", 64, 64))

    def test_legacy_huggingface_resume_starts_at_existing_bytes(self):
        fake_hub, file_download = legacy_hub_modules()
        updates = []

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=100,
                initial=40,
                name="huggingface_hub.http_get",
            ) as progress:
                progress.update(60)

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertEqual(updates[0].current, 40)
        self.assertIsNone(updates[0].speed_bps)
        self.assertEqual(updates[-1].current, 100)

    def test_legacy_huggingface_retry_reuses_one_bar(self):
        fake_hub, file_download = legacy_hub_modules()
        updates = []
        bars = []

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=100,
                name="huggingface_hub.http_get",
            ) as progress:
                bars.append(progress)
                progress.update(25)
                with file_download._get_progress_bar_context(
                    desc=filename,
                    log_level=20,
                    total=100,
                    initial=25,
                    name="huggingface_hub.http_get",
                    _tqdm_bar=progress,
                ) as retry_progress:
                    bars.append(retry_progress)
                    retry_progress.update(75)

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertIs(bars[0], bars[1])
        self.assertEqual(updates[-1].current, 100)
        self.assertEqual([update.phase for update in updates].count("complete"), 1)

    def test_legacy_huggingface_concurrent_callbacks_are_isolated(self):
        fake_hub, file_download = legacy_hub_modules()
        barrier = threading.Barrier(2)

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=10,
                name="huggingface_hub.http_get",
            ) as progress:
                barrier.wait()
                progress.update(10)

        fake_hub.hf_hub_download = legacy_download
        first_updates, second_updates = [], []
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(download_file, "https://huggingface.co/owner/one/resolve/main/one.bin", str(Path(directory) / "one.bin"), first_updates.append)
                second = pool.submit(download_file, "https://huggingface.co/owner/two/resolve/main/two.bin", str(Path(directory) / "two.bin"), second_updates.append)
                first.result()
                second.result()

        self.assertEqual({update.filename for update in first_updates}, {"one.bin"})
        self.assertEqual({update.filename for update in second_updates}, {"two.bin"})

    def test_legacy_huggingface_exception_resets_context_without_completion(self):
        fake_hub, file_download = legacy_hub_modules()
        updates = []

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=10,
                name="huggingface_hub.http_get",
            ) as progress:
                progress.update(5)
                raise RuntimeError("download failed")

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertIsNone(_ACTIVE_HF_PROGRESS_CLASS.get())
        self.assertNotIn("complete", [update.phase for update in updates])

    def test_legacy_huggingface_does_not_intercept_unrelated_progress(self):
        fake_hub, file_download = legacy_hub_modules()
        updates = []
        native_bars = []

        def legacy_download(repo_id, filename, local_dir):
            with file_download._get_progress_bar_context(
                desc=filename,
                log_level=20,
                total=10,
                name="huggingface_hub.snapshot_download",
            ) as progress:
                native_bars.append(progress)
                progress.update(10)

        fake_hub.hf_hub_download = legacy_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertIsInstance(native_bars[0], NativeProgress)
        self.assertEqual(updates, [])

    def test_legacy_huggingface_missing_private_hook_falls_back_to_indeterminate(self):
        fake_hub = types.ModuleType("huggingface_hub")
        file_download = types.ModuleType("huggingface_hub.file_download")
        calls = []

        def legacy_download(repo_id, filename, local_dir):
            calls.append(filename)

        fake_hub.file_download = file_download
        fake_hub.hf_hub_download = legacy_download
        updates = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": fake_hub, "huggingface_hub.file_download": file_download}):
            download_file("https://huggingface.co/owner/repo/resolve/main/model.bin", str(Path(directory) / "model.bin"), updates.append)

        self.assertEqual(calls, ["model.bin"])
        self.assertEqual([update.phase for update in updates], ["downloading", "complete"])
        self.assertIsNone(updates[0].total)

    def test_cached_huggingface_file_emits_no_download_progress(self):
        fake_hub = types.ModuleType("huggingface_hub")
        calls = []
        fake_hub.hf_hub_download = lambda **kwargs: calls.append(kwargs)
        fake_hub.snapshot_download = lambda **kwargs: None
        updates = []
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), \
             patch("shared.utils.files_locator.get_smart_download_root", return_value="models"), \
             patch("shared.utils.files_locator.locate_file", return_value="models/weights.bin"):
            process_files_def("owner/repo", [""], [["weights.bin"]], progress_callback=updates.append)

        self.assertEqual(calls, [])
        self.assertEqual(updates, [])


class APIProgressTests(unittest.TestCase):
    def setUp(self):
        self.session = object.__new__(WanGPSession)

    def test_extended_download_payload(self):
        details = {"kind": "model_download", "filename": "weights.bin", "downloaded_bytes": 50, "total_bytes": 100}
        update = self.session._build_progress_update(
            [(50, 100), "Downloading model Test · weights.bin · 50 B / 100 B · 50.0%", 100, "bytes", details],
            include_state_fallback=False,
        )
        details["filename"] = "changed"

        self.assertEqual(update.phase, "downloading_model")
        self.assertEqual((update.current_step, update.total_steps, update.unit), (50, 100, "bytes"))
        self.assertEqual(update.progress, 10)
        self.assertEqual(update.details["filename"], "weights.bin")

    def test_legacy_payloads_and_phase_ordering(self):
        old = self.session._build_progress_update([0, "Prompt 1/1 | Denoising"], include_state_fallback=False)
        four_item = self.session._build_progress_update([(2, 4), "Denoising", 4, "steps"], include_state_fallback=False)

        self.assertIsNone(old.details)
        self.assertIsNone(four_item.details)
        self.assertEqual((four_item.current_step, four_item.total_steps), (2, 4))
        self.assertEqual(WanGPSession._normalize_phase("Downloading model Test"), "downloading_model")
        self.assertEqual(WanGPSession._normalize_phase("Checking model files for Test"), "checking_model_files")
        self.assertEqual(WanGPSession._normalize_phase("Loading model Test into memory"), "loading_model")


if __name__ == "__main__":
    unittest.main()
