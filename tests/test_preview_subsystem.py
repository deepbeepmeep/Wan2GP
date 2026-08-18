from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sys
import threading
import time
import tempfile
import unittest
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from PIL import Image

from shared.preview.adapters.ltx2 import preview_sample_count, uniform_frame_indices
from shared.preview.coordinator import PreviewCoordinator
from shared.preview.encoding import encode_preview
from shared.preview.loader import PreviewDecoderError, load_decoder, validate_weight
from shared.preview.registry import PreviewDecoderSpec, TAEH3, TAELTX23, decoder_capability, get_decoder_for_model
from shared.preview.rendering import preview_media_to_html
from shared.preview.scheduler import CaptureScheduler
from shared.preview.types import PreviewContext, PreviewMedia, PreviewOptions
from shared.preview.worker import PreviewJob, PreviewWorker
from shared.api import SessionJob, WanGPSession
from shared.mcp_server import _json_safe
from shared.api_cli import _handle_command
import shared.preview.encoding as preview_encoding
import shared.preview.coordinator as preview_coordinator
import shared.preview.worker as preview_worker

try:
    import torch as _torch
except Exception:
    _torch = None
try:
    import safetensors as _safetensors
except Exception:
    _safetensors = None


@dataclass(frozen=True)
class _Event:
    kind: str
    data: Any = None


def _load_session_stream():
    source = Path("shared/api.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "SessionStream")
    namespace = {"Any": Any, "Iterator": Iterator, "SessionEvent": _Event, "deque": deque, "threading": threading, "time": time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "shared/api.py", "exec"), namespace)
    return namespace["SessionStream"]


class PreviewSubsystemTests(unittest.TestCase):
    LTX_MODEL_DEF = {
        "architecture": "ltx2_22B",
        "capabilities": {"live_preview": {"modes": ["off", "rgb", "tae"], "decoders": ["taeltx2_3"]}},
    }

    def test_options_validate_and_clamp(self):
        options = PreviewOptions.from_value({"mode": "tae", "max_edge": 9999, "preview_fps": 16, "webp_quality": 0, "target_updates": 1})
        self.assertEqual(options.mode, "tae")
        self.assertEqual(options.max_edge, 1024)
        self.assertEqual(options.preview_fps, 16)
        self.assertEqual(options.webp_quality, 1)
        self.assertEqual(options.target_updates, 2)
        with self.assertRaises(ValueError): PreviewOptions.from_value({"preview_fps": 3})

    def test_preview_sample_count_uses_decoded_duration(self):
        self.assertEqual(preview_sample_count(241, 16, 24), 160)
        self.assertEqual(preview_sample_count(17, 16, 24), 11)

    def test_uniform_sampling_keeps_temporal_endpoints(self):
        indices = uniform_frame_indices(121, 8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 120)

    def test_registry_requires_architecture_and_declared_capability(self):
        self.assertIs(get_decoder_for_model("unregistered_model_type", self.LTX_MODEL_DEF), TAELTX23)
        unsupported_architecture = {
            "architecture": "ltx2_19B",
            "capabilities": {"live_preview": {"modes": ["tae"], "decoders": ["taeltx2_3"]}},
        }
        missing_decoder_capability = {
            "architecture": "ltx2_22B",
            "capabilities": {"live_preview": {"modes": ["tae"]}},
        }
        self.assertIsNone(get_decoder_for_model("unregistered_model_type", unsupported_architecture))
        self.assertIsNone(get_decoder_for_model("unregistered_model_type", missing_decoder_capability))
        capability = decoder_capability("unregistered_model_type", missing_decoder_capability)
        self.assertEqual(capability["modes"], ["off", "rgb"])

    def test_default_ltx_profiles_advertise_tae_capability(self):
        profiles = (
            "ltx2_22B.json",
            "ltx2_22B_distilled.json",
            "ltx2_22B_1_1.json",
            "ltx2_22B_distilled_1_1.json",
        )
        for filename in profiles:
            with self.subTest(filename=filename):
                model_def = json.loads(Path("defaults", filename).read_text(encoding="utf-8"))["model"]
                self.assertIsNotNone(get_decoder_for_model(filename.removesuffix(".json"), model_def))

    def test_default_h3_profiles_advertise_tae_capability(self):
        self.assertEqual(TAEH3.filename, "taeh3.safetensors")
        self.assertEqual(TAEH3.size_bytes, 9_791_388)
        self.assertEqual(TAEH3.sha256, "f0f60fa072089997f817402098c2fd90777cb2660dd79cf5df42fc1e3e08e527")
        self.assertIn("Kijai/MiniMax-H3-TAE", TAEH3.source_url)
        for filename in (
            "minimax_h3_fl2va.json",
            "minimax_h3_fl2va_pruned.json",
            "minimax_h3_ref2va.json",
            "minimax_h3_ref2va_pruned.json",
        ):
            with self.subTest(filename=filename):
                model_def = json.loads(Path("defaults", filename).read_text(encoding="utf-8"))["model"]
                self.assertIs(get_decoder_for_model(model_def["architecture"], model_def), TAEH3)

    def test_missing_weight_is_not_advertised(self):
        with patch.object(PreviewDecoderSpec, "local_path", return_value=None):
            capability = decoder_capability("ltx2_22B", self.LTX_MODEL_DEF)
        self.assertEqual(capability["modes"], ["off", "rgb"])
        self.assertFalse(capability["tiny_vae_available"])

    def test_weight_validation_rejects_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.safetensors"
            self.assertFalse(validate_weight(missing, TAELTX23)[0])
            corrupt = Path(directory) / "corrupt.safetensors"
            corrupt.write_bytes(b"corrupt")
            self.assertFalse(validate_weight(corrupt, TAELTX23)[0])
            with self.assertRaises(PreviewDecoderError):
                load_decoder(corrupt, TAELTX23)

    def test_adaptive_scheduler_uses_one_based_callbacks_and_captures_final_once(self):
        for target_updates in (7, 16):
            with self.subTest(target_updates=target_updates):
                scheduler = CaptureScheduler(100, target_updates=target_updates)
                captured = [step for step in range(101) if scheduler.should_capture(step)]
                self.assertNotIn(0, captured)
                self.assertLessEqual(captured[0], 2)
                self.assertEqual(captured.count(100), 1)
                self.assertEqual(captured[-1], 100)
                self.assertEqual(len(captured), target_updates)

    def test_webp_is_animated(self):
        frames = [Image.new("RGB", (32, 24), color=(index * 30, 0, 0)) for index in range(4)]
        context = PreviewContext("g", "c", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 2, 8, fps=8)
        with patch.object(preview_encoding, "_encode_mp4", return_value=None):
            media = encode_preview(frames, context, PreviewOptions(max_edge=32), decode_ms=1.0)
        self.assertEqual(media.mime_type, "image/webp")
        with Image.open(io.BytesIO(media.data)) as image:
            self.assertEqual(getattr(image, "n_frames", 1), 4)
            self.assertEqual(image.info.get("loop"), 0)

    def test_webp_fallback_keeps_all_samples(self):
        def anmf_durations(data):
            durations, offset = [], 12
            while offset + 8 <= len(data):
                length = int.from_bytes(data[offset + 4 : offset + 8], "little")
                if data[offset : offset + 4] == b"ANMF":
                    durations.append(int.from_bytes(data[offset + 20 : offset + 23], "little"))
                offset += 8 + length + (length & 1)
            return durations
        context = PreviewContext("g", "c", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 2, 8, fps=24, duration_seconds=121 / 24)
        with patch.object(preview_encoding, "_encode_mp4", return_value=None):
            media = encode_preview([Image.new("RGB", (4, 4), color=(index * 16, 0, 0)) for index in range(16)], context, PreviewOptions(max_edge=128), decode_ms=1.0)
        self.assertEqual(media.frame_count, 16)
        self.assertEqual(media.duration_ms, 1000)
        self.assertEqual(media.media_kind, "animated_image")
        self.assertEqual(sum(anmf_durations(media.data)), 1000)

    def test_mp4_preview_media_and_renderer(self):
        context = PreviewContext("g", "c", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 2, 8)
        with patch.object(preview_encoding, "_encode_mp4", return_value=b"mp4"):
            media = encode_preview([Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4), color=(1, 0, 0))], context, PreviewOptions(), decode_ms=1.0)
        self.assertEqual((media.media_kind, media.mime_type), ("video", "video/mp4"))
        self.assertEqual((media.fps, media.duration_ms, media.frame_count), (16, 125, 2))
        rendered = preview_media_to_html(media)
        self.assertIn("<video", rendered)
        self.assertIn("autoplay loop muted playsinline", rendered)
        self.assertRegex(rendered, r'data-preview-generation="g" ontimeupdate="[^"]*window\.__wangpPreviewPlayback[^"]*" onloadedmetadata="[^"]*window\.__wangpPreviewPlayback[^"]*"')
        self.assertNotIn("showImageModal", rendered)

    def test_encoder_reports_static_when_frames_have_no_temporal_change(self):
        frames = [Image.new("RGB", (32, 24), color=(0, 0, 0)) for _ in range(4)]
        context = PreviewContext("g", "c", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 2, 8)
        with patch.object(preview_encoding, "_encode_mp4", return_value=None):
            media = encode_preview(frames, context, PreviewOptions(max_edge=32), decode_ms=1.0)
        self.assertEqual(media.media_kind, "image")
        self.assertEqual(media.frame_count, 1)
        self.assertIn("temporal change", media.warning or "")
        with Image.open(io.BytesIO(media.data)) as image:
            self.assertEqual(getattr(image, "n_frames", 1), 1)

    def test_encoder_falls_back_to_first_frame(self):
        frames = [Image.new("RGB", (32, 24), color=(index * 30, 0, 0)) for index in range(3)]
        context = PreviewContext("g", "c", 1, "ltx2_22B", "ltx2_22B", "taeltx2_3", 2, 8)
        original = preview_encoding._encode_webp

        def fail_animation(frames, *args, **kwargs):
            if len(frames) > 1:
                raise RuntimeError("test animation failure")
            return original(frames, *args, **kwargs)

        with patch.object(preview_encoding, "_encode_mp4", return_value=None), patch.object(preview_encoding, "_encode_webp", fail_animation):
            media = encode_preview(frames, context, PreviewOptions(max_edge=32), decode_ms=1.0)
        self.assertEqual(media.media_kind, "image")
        self.assertEqual(media.frame_count, 1)
        self.assertIn("first frame", media.warning or "")

    def test_media_serialization_keeps_binary_out_of_in_process_path(self):
        media = PreviewMedia("g", "c", 1, "image", "image/webp", b"bytes", 2, 2, 1, None, None, 1, 2, "decoder", 1.0, 1.0)
        self.assertEqual(media.to_dict(encode_data=False)["data"], b"bytes")
        self.assertEqual(media.to_dict()["data"], "Ynl0ZXM=")

    def test_mcp_json_serializes_preview_media(self):
        media = PreviewMedia("g", "c", 1, "image", "image/webp", b"bytes", 2, 2, 1, None, None, 1, 2, "decoder", 1.0, 1.0)
        self.assertEqual(_json_safe(media)["data"], "Ynl0ZXM=")

    def test_stream_coalesces_previews_without_overtaking_events(self):
        stream = _load_session_stream()()
        stream.put("preview", "old")
        stream.put("progress", 1)
        stream.put("status", "working")
        stream.put("preview", "new")
        stream.put("error", "keep")
        self.assertEqual(stream.get().data, 1)
        self.assertEqual(stream.get().data, "working")
        self.assertEqual(stream.get().data, "new")
        self.assertEqual(stream.get().data, "keep")

    def test_api_cli_preserves_structured_preview_media(self):
        first_frame = Image.new("RGB", (4, 4), color=(255, 0, 0))
        media = PreviewMedia("g", "c", 1, "image", "image/webp", b"bytes", 4, 4, 1, 24.0, 42, 3, 8, "taeltx2_3", 1.2, 2.3, first_frame=first_frame)
        session = WanGPSession(console_output=False)
        job = SessionJob(session)

        _handle_command(session, job, None, [], "preview_media", media)
        event = job.events.get()
        update = event.data

        self.assertEqual(event.kind, "preview")
        self.assertIs(update.media, media)
        self.assertIs(update.image, first_frame)
        self.assertEqual(update.current_step, 3)

    def test_worker_bounds_pending_jobs_and_is_cancel_safe(self):
        frames = (Image.new("RGB", (2, 2)),)
        options = PreviewOptions(max_edge=2)
        first_started = threading.Event()
        release_first = threading.Event()
        result_ready = threading.Event()
        results = []

        def fake_encode(frames, context, options, *, decode_ms, dropped_count=0):
            if context.sequence == 1:
                first_started.set()
                release_first.wait(1)
            return context

        def context(sequence):
            return PreviewContext("generation", "context", sequence, "model", "arch", "decoder", sequence, 4)

        with patch.object(preview_worker, "encode_preview", fake_encode):
            worker = PreviewWorker(lambda result: (results.append(result), result_ready.set()))
            worker.try_submit(PreviewJob(frames, context(1), options, 0))
            self.assertTrue(first_started.wait(1))
            worker.try_submit(PreviewJob(frames, context(2), options, 0))
            worker.try_submit(PreviewJob(frames, context(3), options, 0))
            release_first.set()
            self.assertTrue(result_ready.wait(1))
            worker.close(wait=True)
        self.assertEqual([item.sequence for item in results], [1, 3])
        self.assertEqual(worker.dropped_count, 1)

        first_started.clear()
        release_first.clear()
        result_ready.clear()
        results.clear()
        with patch.object(preview_worker, "encode_preview", fake_encode):
            worker = PreviewWorker(results.append)
            worker.try_submit(PreviewJob(frames, context(1), options, 0))
            self.assertTrue(first_started.wait(1))
            worker.invalidate("generation")
            release_first.set()
            worker.close(wait=True)
        self.assertEqual(results, [])

    def test_coordinator_rejects_stale_and_cancelled_media(self):
        coordinator = PreviewCoordinator.__new__(PreviewCoordinator)
        coordinator.cancelled = False
        coordinator.disabled = False
        coordinator.generation_id = "generation"
        coordinator.context_id = "context"
        coordinator.sequence = 2
        published = []
        coordinator._publish_callback = published.append

        def media(sequence):
            return PreviewMedia("generation", "context", sequence, "image", "image/webp", b"", 2, 2, 1, None, None, 1, 2, "decoder", 1.0, 1.0)

        coordinator._publish(media(1))
        coordinator._publish(media(2))
        coordinator.cancelled = True
        coordinator._publish(media(2))
        self.assertEqual([item.sequence for item in published], [2])

    def test_coordinator_unloads_decoder_cache_on_close(self):
        coordinator = PreviewCoordinator("ltx2_22B", "ltx2_22B", TAELTX23, PreviewOptions(), lambda media: None)
        coordinator._decoder = object()
        with patch.object(preview_coordinator, "unload_decoders") as unload:
            coordinator.close()
        unload.assert_called_once_with()
        self.assertIsNone(coordinator._decoder)

    def test_coordinator_falls_back_to_cpu_after_initial_gpu_oom(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda()

        class Latent:
            is_cuda = True

        loaded_devices = []
        decoded_devices = []

        def fake_load(_path, _spec, *, device):
            loaded_devices.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return {"device": device}

        def fake_decode(decoder, _latent, **kwargs):
            decoded_devices.append((decoder["device"], kwargs["parallel"]))
            return [Image.new("RGB", (2, 2))], 1.0, 1

        coordinator = PreviewCoordinator("ltx2_22B", "ltx2_22B", TAELTX23, PreviewOptions(), lambda media: None)
        with (
            patch.dict(sys.modules, {"torch": FakeTorch}),
            patch.object(preview_coordinator, "load_decoder", side_effect=fake_load),
            patch.object(preview_coordinator, "decode_ltx2_latent", side_effect=fake_decode),
            patch.object(preview_coordinator, "unload_decoders"),
            patch.object(coordinator._worker, "try_submit", return_value=True),
        ):
            self.assertTrue(coordinator.capture(Latent(), step=1, total_steps=2))
        coordinator.close()
        self.assertEqual(loaded_devices, ["cuda", "cpu"])
        self.assertEqual(decoded_devices, [("cpu", False)])
        self.assertEqual(coordinator._decode_device, "cpu")

    @unittest.skipUnless(_torch is not None, "torch runtime unavailable")
    def test_ltx_adapter_converts_and_does_not_mutate_latent(self):
        from shared.preview.adapters.ltx2 import decode_ltx2_latent

        class FakeDecoder:
            def __init__(self):
                self.parameter = _torch.nn.Parameter(_torch.zeros(1))

            def parameters(self):
                return iter((self.parameter,))

            def decode_video(self, value, parallel=True, show_progress_bar=False):
                self.received = value
                frames = 8 * (value.shape[1] - 1) + 1
                return _torch.zeros((1, frames, 3, 8, 12), device=value.device, dtype=value.dtype)

        latent = _torch.randn(128, 3, 2, 3)
        original = latent.clone()
        frames, _, decoded_count = decode_ltx2_latent(FakeDecoder(), latent, spec=TAELTX23, max_edge=12, preview_fps=8, source_fps=16)
        self.assertEqual(len(frames), 8)
        self.assertEqual(decoded_count, 17)
        self.assertTrue(_torch.equal(latent, original))

    @unittest.skipUnless(_torch is not None, "torch runtime unavailable")
    def test_h3_adapter_uses_raw_latent_frames(self):
        from shared.preview.adapters.h3 import decode_h3_latent

        class FakeDecoder:
            def __init__(self):
                self.parameter = _torch.nn.Parameter(_torch.zeros(1))

            def parameters(self):
                return iter((self.parameter,))

            def decode_video(self, value, parallel=True, show_progress_bar=False):
                self.received = value
                return _torch.zeros((1, value.shape[1], 3, 8, 12), device=value.device, dtype=value.dtype)

        decoder = FakeDecoder()
        latent = _torch.randn(24, 3, 2, 3)
        original = latent.clone()
        frames, _, decoded_count = decode_h3_latent(decoder, latent, spec=TAEH3, max_edge=12, preview_fps=8, source_fps=16)
        self.assertEqual(tuple(decoder.received.shape), (1, 3, 24, 2, 3))
        self.assertTrue(_torch.equal(decoder.received[0, :, :, :, :].permute(1, 0, 2, 3), original))
        self.assertEqual(decoded_count, 3)
        self.assertEqual(len(frames), 2)
        self.assertTrue(_torch.equal(latent, original))

    @unittest.skipUnless(_torch is not None, "torch runtime unavailable")
    def test_euler_callback_uses_postprocessed_denoised_latent(self):
        source = Path("models/ltx2/ltx_pipelines/utils/helpers.py").read_text(encoding="utf-8")
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body
            + [functions["_invoke_callback"], functions["euler_denoising_loop"]],
            type_ignores=[],
        )

        class Offload:
            @staticmethod
            def set_step_no_for_lora(transformer, step_idx):
                pass

        namespace = {
            "offload": Offload,
            "post_process_latent": lambda denoised, denoise_mask, clean_latent: denoised + 10,
            "replace": replace,
            "tqdm": lambda steps: steps,
        }
        exec(compile(ast.fix_missing_locations(module), "models/ltx2/ltx_pipelines/utils/helpers.py", "exec"), namespace)

        @dataclass(frozen=True)
        class State:
            latent: Any
            clean_latent: Any = None
            denoise_mask: Any = None

        class PreviewTools:
            @staticmethod
            def clear_conditioning(state):
                return state

            @staticmethod
            def unpatchify(state):
                return state

        class Stepper:
            @staticmethod
            def step(sample, denoised, sigmas, step_idx):
                return denoised + 100

        captured = []
        final_video, _ = namespace["euler_denoising_loop"](
            _torch.tensor([1.0, 0.0]),
            State(_torch.tensor([1.0])),
            None,
            Stepper(),
            lambda video_state, audio_state, sigmas, step_idx: (_torch.tensor([3.0]), None),
            callback=lambda step, latent, is_final, *, pass_no: captured.append((step, latent, is_final, pass_no)),
            preview_tools=PreviewTools(),
            pass_no=7,
        )

        self.assertEqual([(step, is_final, pass_no) for step, _, is_final, pass_no in captured], [(0, False, 7)])
        self.assertTrue(_torch.equal(captured[0][1], _torch.tensor(13.0)))
        self.assertTrue(_torch.equal(final_video.latent, _torch.tensor([113.0])))

    @unittest.skipUnless(_torch is not None, "torch runtime unavailable")
    def test_h3_euler_callback_uses_denoised_latent(self):
        source = ast.parse(Path("models/minimax_h3/pipeline.py").read_text(encoding="utf-8"))
        denoise_pass = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "denoise_pass")
        denoise_pass.body[0] = ast.Global(names=["video", "audio"])
        namespace = {
            "torch": _torch,
            "tqdm": lambda steps, **_: steps,
            "offload": type("Offload", (), {"set_step_no_for_lora": staticmethod(lambda *_: None)}),
            "model_steps": 1,
            "sigmas_video": _torch.tensor([1.0, 0.5]),
            "sigmas_audio": _torch.tensor([1.0, 0.5]),
            "res_coefficients": None,
            "spectrum": None,
            "first_block_cache": None,
            "target_audio_condition_latents": 0,
            "target_video_condition_frames": 0,
            "source_latents": None,
            "source_noise": None,
            "source_buffer": None,
            "editable_mask": None,
            "denoising_start_step": 0,
            "mask_end_step": 0,
            "offline_spectrum": False,
            "payload": None,
            "context": None,
            "audio_scale": 1.0,
            "video": _torch.tensor([1.0]),
            "audio": _torch.tensor([0.0]),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[denoise_pass], type_ignores=[])), "models/minimax_h3/pipeline.py", "exec"), namespace)

        class Transformer:
            cache = None

            def __call__(self, *_args, **_kwargs):
                return _torch.tensor([2.0]), _torch.tensor([0.0])

        class Pipeline:
            transformer = Transformer()

            @staticmethod
            def _set_interrupt_state():
                pass

            @staticmethod
            def _check_abort():
                pass

        captured = []
        namespace["self"] = Pipeline()
        namespace["callback"] = lambda step, latent, is_final: captured.append((step, latent, is_final))
        namespace["denoise_pass"]("H3")

        self.assertEqual([(step, is_final) for step, _, is_final in captured], [(0, False)])
        self.assertTrue(_torch.equal(captured[0][1], _torch.tensor(3.0)))
        self.assertTrue(_torch.equal(namespace["video"], _torch.tensor([2.0])))

    @unittest.skipUnless(_torch is not None and _safetensors is not None and os.getenv("WANGP_PREVIEW_FIXTURE_TEST"), "opt-in torch fixture test")
    def test_strict_loader_accepts_a_matching_safetensors_fixture(self):
        from safetensors.torch import save_file
        from shared.preview.vendor.taehv import TAEHV

        model = TAEHV(
            checkpoint_path=None,
            patch_size=TAELTX23.patch_size,
            latent_channels=TAELTX23.latent_channels,
            encoder_time_downscale=TAELTX23.encoder_time_downscale,
            decoder_time_upscale=TAELTX23.decoder_time_upscale,
            decoder_space_upscale=(True, True, True),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = replace(TAELTX23, size_bytes=path.stat().st_size, sha256=digest)
            loaded = load_decoder(path, spec)
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
