"""Offline installer contract tests; no GPU, weights or network required."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shared.preview.loader import PreviewDecoderError, download_decoder
from shared.preview.registry import PreviewDecoderSpec


class PreviewDownloadInstallTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / "decoder.safetensors"
        self.contents = b"verified decoder fixture"
        self.spec = PreviewDecoderSpec(
            "fixture", self.target.name, hashlib.sha256(self.contents).hexdigest(),
            len(self.contents), 128, 4, (True,) * 3, (True,) * 3,
            frozenset({"fixture"}), "ltx2", "https://example.invalid/decoder.safetensors",
        )
        self.local_path = patch.object(PreviewDecoderSpec, "local_path", return_value=str(self.target))
        self.local_path.start()
        self.addCleanup(self.local_path.stop)
        self.reports = []
        self.calls = []

    def invoke(self, downloader, callback=None, gen=None):
        module = types.ModuleType("shared.utils.download")
        module.download_file = downloader
        with patch.dict(sys.modules, {"shared.utils.download": module}):
            return download_decoder(self.spec, callback, gen=gen)

    def native(self, url, filename, gen=None, show_filename=True):
        # Deliberately no progress_callback keyword or **kwargs: this is the
        # upstream-native call contract, rather than the separate progress PR.
        self.calls.append((url, filename))
        callback = gen.get("download_progress_callback")
        if callback:
            callback({"filename": self.spec.filename, "completed": 1,
                      "total": len(self.contents), "speed": 2.0,
                      "file_index": 1, "file_count": 1})
        Path(filename).write_bytes(self.contents)
        if callback:
            callback({"filename": self.spec.filename, "completed": len(self.contents),
                      "total": len(self.contents), "speed": 2.0})
            callback(None)

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".preview-install-*")), [])

    def test_native_signature_progress_and_atomic_publication(self):
        result = self.invoke(self.native, self.reports.append)
        self.assertEqual(result, str(self.target))
        self.assertEqual(self.target.read_bytes(), self.contents)
        self.assertNotEqual(self.calls[0][1], str(self.target))
        self.assertEqual(self.reports[0]["current"], 1)
        self.assertEqual(self.reports[0]["completed"], 1)
        self.assertEqual(self.reports[0]["speed_bps"], 2.0)
        self.assertEqual(self.reports[0]["file_count"], 1)
        self.assertEqual(len(self.reports), 2)
        self.assert_no_staging()

    def test_valid_existing_weights_skip_transfer(self):
        self.target.write_bytes(self.contents)
        downloader = Mock(side_effect=AssertionError("unexpected transfer"))
        self.assertEqual(self.invoke(downloader), str(self.target))
        downloader.assert_not_called()

    def test_corrupt_existing_weights_survive_failed_repair(self):
        self.target.write_bytes(b"keep until verified")
        def fail(url, filename, gen=None):
            Path(filename).write_bytes(b"partial")
            raise OSError("interrupted")
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.invoke(fail)
        self.assertEqual(self.target.read_bytes(), b"keep until verified")
        self.assert_no_staging()

    def test_invalid_size_is_not_published(self):
        def wrong(url, filename, gen=None):
            Path(filename).write_bytes(b"short")
        with self.assertRaisesRegex(PreviewDecoderError, "size mismatch"):
            self.invoke(wrong)
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_invalid_hash_is_not_published(self):
        self.target.write_bytes(b"old")
        def wrong(url, filename, gen=None):
            Path(filename).write_bytes(b"x" * len(self.contents))
        with self.assertRaisesRegex(PreviewDecoderError, "SHA-256 mismatch"):
            self.invoke(wrong)
        self.assertEqual(self.target.read_bytes(), b"old")
        self.assert_no_staging()

    def test_missing_output_is_not_published(self):
        def missing(url, filename, gen=None):
            pass
        with self.assertRaisesRegex(PreviewDecoderError, "missing"):
            self.invoke(missing)
        self.assertFalse(self.target.exists())
        self.assert_no_staging()

    def test_preview_callback_failure_does_not_fail_transfer(self):
        callback = Mock(side_effect=RuntimeError("UI failed"))
        gen = {}
        with self.assertLogs("shared.preview.loader", "WARNING"):
            self.invoke(self.native, callback, gen)
        self.assertEqual(callback.call_count, 1)
        self.assertNotIn("download_progress_callback", gen)
        self.assertEqual(self.target.read_bytes(), self.contents)

    def test_existing_native_callback_is_preserved_and_restored(self):
        previous = Mock()
        gen = {"download_progress_callback": previous}
        self.invoke(self.native, self.reports.append, gen)
        self.assertIs(gen["download_progress_callback"], previous)
        self.assertEqual(previous.call_count, 3)
        self.assertIsNone(previous.call_args.args[0])
        self.assertNotIn(None, self.reports)

    def test_existing_callback_without_preview_callback_stays_unchanged(self):
        previous = Mock()
        gen = {"download_progress_callback": previous}
        self.invoke(self.native, gen=gen)
        self.assertIs(gen["download_progress_callback"], previous)
        self.assertEqual(previous.call_count, 3)

    def test_pre_cancelled_download_does_no_work(self):
        downloader = Mock(side_effect=AssertionError("unexpected transfer"))
        with self.assertRaisesRegex(PreviewDecoderError, "cancelled"):
            self.invoke(downloader, gen={"abort": True})
        downloader.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_cancelled_completed_transfer_is_not_published(self):
        self.target.write_bytes(b"old")
        def cancelled(url, filename, gen=None):
            Path(filename).write_bytes(self.contents)
            gen["abort"] = True
        with self.assertRaisesRegex(PreviewDecoderError, "cancelled"):
            self.invoke(cancelled)
        self.assertEqual(self.target.read_bytes(), b"old")
        self.assert_no_staging()

    def test_native_cancellation_exception_and_callback_are_preserved(self):
        class NativeCancelled(Exception):
            pass
        previous = Mock()
        gen = {"download_progress_callback": previous}
        def cancelled(url, filename, gen=None):
            raise NativeCancelled("native abort")
        with self.assertRaises(NativeCancelled):
            self.invoke(cancelled, self.reports.append, gen)
        self.assertIs(gen["download_progress_callback"], previous)
        self.assert_no_staging()

    def test_cancellation_callback_during_validation_preserves_previous_file(self):
        self.target.write_bytes(b"old")
        cancelled = False

        def digest(path):
            nonlocal cancelled
            cancelled = True
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with patch("shared.preview.loader._sha256", side_effect=digest):
            with self.assertRaisesRegex(PreviewDecoderError, "cancelled"):
                self.invoke(self.native, gen={"abort_callback": lambda: cancelled})
        self.assertEqual(self.target.read_bytes(), b"old")
        self.assert_no_staging()

    def test_legacy_downloader_needs_no_separate_progress_pr(self):
        def legacy(url, filename):
            Path(filename).write_bytes(self.contents)
        self.invoke(legacy, self.reports.append)
        self.assertIsNone(self.reports[0]["total"])
        self.assertEqual(self.reports[-1]["current"], len(self.contents))
        self.assertEqual(self.target.read_bytes(), self.contents)
        self.assert_no_staging()

    def test_default_location_uses_files_locator(self):
        locator = types.ModuleType("shared.utils.files_locator")
        locator.get_smart_download_location = Mock(return_value=str(self.target))
        utils = types.ModuleType("shared.utils")
        utils.files_locator = locator
        with patch.object(PreviewDecoderSpec, "local_path", return_value=None), \
             patch.dict(sys.modules, {"shared.utils": utils, "shared.utils.files_locator": locator}):
            self.invoke(self.native)
        locator.get_smart_download_location.assert_called_once_with(self.spec.filename, self.spec.target_dir)
        self.assertEqual(self.target.read_bytes(), self.contents)

    def test_successful_repair_replaces_old_file_only_after_validation(self):
        self.target.write_bytes(b"old")
        def replacement(url, filename, gen=None):
            self.assertEqual(self.target.read_bytes(), b"old")
            Path(filename).write_bytes(self.contents)
            self.assertEqual(self.target.read_bytes(), b"old")
        self.invoke(replacement)
        self.assertEqual(self.target.read_bytes(), self.contents)
        self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
