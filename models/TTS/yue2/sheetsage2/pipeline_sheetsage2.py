"""Whole-song inference with cached overlap prefixes and right-hand audio context."""
from pathlib import Path
import json
import re
import time

import torch

from .io_sheetsage2 import atomic_write_text
from .audio_sheetsage2 import SAMPLE_RATE, load_audio, slice_audio
from .generation_sheetsage2 import (
    FULL_TASK_PROMPTS, build_overlap_prefix_tokens, constrained_prompt_generate,
    decode_generated_tokens, event_time_map, stitched_window_events, write_window_tokens,
)
from .exports_sheetsage2 import export_result


def _tensor_files(output_dir):
    """Inventory only files owned by the optional tensor exporter."""
    files = set()
    tensor_dir = output_dir / "tensors"
    if tensor_dir.is_symlink():
        return files
    for directory in tensor_dir.glob("window-*"):
        if re.fullmatch(r"window-\d{4,}", directory.name) and directory.is_dir() and not directory.is_symlink():
            for path in directory.iterdir():
                if path.is_file() and re.fullmatch(r"(?:index\.json|audio\.safetensors|(?:tokens|decoder)-\d{4,}\.safetensors)", path.name):
                    files.add(path)
    return files


def sliding_window_plan(duration, window_seconds=300.0, overlap_seconds=200.0, lookahead_seconds=100.0):
    if not duration > 0 or not window_seconds > 0:
        raise ValueError("Duration and window length must be positive")
    if not 0 <= lookahead_seconds <= overlap_seconds < window_seconds:
        raise ValueError("Require 0 <= lookahead <= overlap < window length")
    hop = window_seconds - overlap_seconds
    start, accepted = 0.0, 0.0
    result = []
    while True:
        last = start + window_seconds >= duration - 1e-6
        accept_end = duration if last else start + window_seconds - lookahead_seconds
        result.append(dict(start=start, end=min(duration, start + window_seconds),
                           accept_start=accepted, accept_end=accept_end, prefix_end=accepted,
                           generation_stop=None if last else window_seconds - lookahead_seconds))
        if last:
            return result
        accepted = accept_end
        start = min(start + hop, duration - window_seconds)


class Transcriber:
    def __init__(self, model, dtype="bf16"):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = {"bf16": torch.bfloat16, "fp32": None}[dtype]

    @torch.inference_mode()
    def analyze(self, audio_path, output_dir=None, *, prompts=FULL_TASK_PROMPTS,
                sampling_rate=None, max_seconds=None, preset="default",
                overlap_seconds=None, lookahead_seconds=None, progress=None,
                export_logits=False, export_scores=False, export_embeddings=False,
                output_hidden_states=False, melody_only=False):
        def report(stage, **fields):
            if progress:
                progress(dict(stage=stage, **fields))

        if not isinstance(melody_only, bool):
            raise ValueError("melody_only must be True or False")
        if preset not in ("default", "paper"):
            raise ValueError("preset must be default or paper")
        defaults = (200.0, 100.0) if preset == "default" else (100.0, 0.0)
        overlap_seconds = defaults[0] if overlap_seconds is None else overlap_seconds
        lookahead_seconds = defaults[1] if lookahead_seconds is None else lookahead_seconds
        if preset == "paper" and (overlap_seconds, lookahead_seconds) != defaults:
            raise ValueError("paper preset fixes overlap=100 and lookahead=0")
        started = time.monotonic()
        output_dir = Path(output_dir) if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        previous_tensors = _tensor_files(output_dir) if output_dir is not None else set()
        current_tensors = set()
        tensor_results = []
        prompts = self.model.tokenizer.normalize_prompts(prompts)
        if "timestamp" not in prompts:
            raise ValueError("timestamp is required to export timed annotations")
        report("audio")
        audio = load_audio(audio_path, sampling_rate=sampling_rate, max_seconds=max_seconds, preset=preset)
        duration = len(audio) / SAMPLE_RATE
        window_length = float(self.model.hparams.input_audio_length)
        plan = sliding_window_plan(duration, window_length, overlap_seconds, lookahead_seconds)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        stitched, records, warnings = [], [], []
        for index, window in enumerate(plan):
            report("encoding", window=index + 1, windows=len(plan), start=window["start"])
            segment = slice_audio(audio, window["start"], window_length)[None].to(self.device)
            prefix, base = None, 0
            if index:
                _, prefix, base = build_overlap_prefix_tokens(
                    stitched, self.model.tokenizer, prompts, window["start"], window["prefix_end"])
                if prefix is not None:
                    if preset == "paper" and len(prefix) >= self.model.max_output_seq_len - 16:
                        warnings.append(f"Window {index + 1}: overlap prefix exceeded the context")
                        prefix = None
                    elif preset != "paper" and len(prefix) >= self.model.max_output_seq_len - 128:
                        raise ValueError("Overlap prefix fills the context; reduce overlap_seconds")
            tick = time.monotonic()
            memory = None
            tensor_record = None
            if export_embeddings or output_hidden_states or export_logits or export_scores:
                from .tensors_sheetsage2 import WindowTensorWriter
                tensor_record = WindowTensorWriter(output_dir, index, window, export_logits, export_scores)
                if export_embeddings or output_hidden_states:
                    memory = tensor_record.audio_features(self.model, segment, self.dtype, output_hidden_states)
            tokens = constrained_prompt_generate(
                self.model, segment, prompts, self.model.max_output_seq_len,
                prefix_tokens=prefix, autocast_dtype=self.dtype,
                stop_time_seconds=(None if preset == "paper" else
                                   window["generation_stop"] if window["generation_stop"] is not None else
                                   min(duration - window["start"], window_length)),
                memory=memory,
                step_callback=tensor_record.capture if tensor_record is not None and (export_logits or export_scores) else None,
                progress_callback=lambda n: report("decoding", window=index + 1, windows=len(plan), tokens=n),
            )
            if tensor_record is not None:
                if export_embeddings:
                    tensor_record.decoder_features(self.model, segment, tokens, self.dtype, memory)
                tensor_data = tensor_record.finish()
                if tensor_data is not None:
                    tensor_results.append(tensor_data)
                if tensor_record.directory is not None:
                    current_tensors.update(tensor_record.directory / name for name in tensor_record.files)
                    current_tensors.add(tensor_record.directory / "index.json")
            if len(tokens) > self.model.max_output_seq_len:
                warnings.append(f"Window {index + 1} reached the token limit; inspect its token coverage")
            decoded, warning = decode_generated_tokens(self.model.tokenizer, tokens, Path(audio_path).stem if isinstance(audio_path, (str, Path)) else "audio", index)
            if warning:
                warnings.append(warning["error"])
            lookup = event_time_map(decoded, window_length)
            accepted = stitched_window_events(
                decoded, lookup, window["start"], window["accept_start"], window["accept_end"],
                duration, index, global_subbeat_base=base or 0,
            )
            stitched.extend(accepted)
            if preset == "paper":
                stitched.sort(key=lambda e: (float(e.get("time", 0)), int(e.get("window_index", 0)),
                                             int(e.get("source_subbeat", e["subbeat"]))))
            record = dict(window, window_index=index, prefix_tokens=0 if prefix is None else len(prefix),
                          tokens=tokens, events=len(decoded["events"]), accepted_events=len(accepted),
                          elapsed_seconds=time.monotonic() - tick)
            records.append(record)
            report("window_complete", window=index + 1, windows=len(plan), tokens=len(tokens))
        if preset != "paper":
            stitched.sort(key=lambda e: (e["time"], e["global_subbeat"]))
        decoded = dict(schema_version=self.model.tokenizer.schema_version,
                       prompts=list(prompts), events=stitched, has_eos=True)
        report("notation")
        exported = export_result(decoded, self.model.tokenizer, output_dir, duration,
                                 paper=preset == "paper", melody_only=melody_only)
        payload = exported.pop("payload")
        if output_dir is not None:
            write_window_tokens(output_dir / "tokens.txt", records, self.model.tokenizer)
            atomic_write_text(output_dir / "tokens.json", json.dumps([
                dict(r, tokens=r["tokens"].tolist()) for r in records
            ]))
        result = dict(
            audio=Path(audio_path).name if isinstance(audio_path, (str, Path)) else "audio",
            duration_seconds=duration, prompts=list(prompts), preset=preset, melody_only=melody_only,
            dtype="bf16" if self.dtype and self.device.type == "cuda" else "fp32",
            window_seconds=window_length, overlap_seconds=overlap_seconds, lookahead_seconds=lookahead_seconds,
            windows=[dict(r, tokens=len(r["tokens"])) for r in records],
            elapsed_seconds=time.monotonic() - started, warnings=warnings,
            peak_gpu_mib=torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0,
            **exported,
        )
        if output_dir is not None:
            atomic_write_text(output_dir / "result.json", json.dumps(result, indent=2))
        for path in previous_tensors - current_tensors:
            path.unlink(missing_ok=True)
        for directory in {path.parent for path in previous_tensors - current_tensors}:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        report("complete", **exported)
        return dict(result, num_events=result["events"], **payload,
                    tokens=[r["tokens"] for r in records], tensors=tensor_results,
                    _metadata=result)


@torch.inference_mode()
def transcribe(model, audio, output_dir=None, *, render_audio=False, render_score=False,
               render_parts=("mix",), dtype="bf16", melody_only=False, **kwargs):
    """Transcribe a path, encoded audio bytes, binary stream, or waveform.

    Arrays use channels-first layout and require sampling_rate. With
    output_dir=None, all audio/results stay in memory: ABC is text, MIDI is
    bytes, events are a list, and optional features are CPU tensors grouped by
    window. A directory saves the standard output files and writes optional
    tensors to disk instead of keeping them in memory. Optional
    rendering also returns bytes/text when no output directory is given.
    melody_only=True omits chords from ABC and MIDI playback, retaining both
    melody voices and the original decoded events/LAB annotations. If the
    requested melody-only ABC cannot be built, raises RuntimeError with the
    completed transcription available as error.result.
    """
    if not isinstance(melody_only, bool):
        raise ValueError("melody_only must be True or False")
    output_dir = Path(output_dir) if output_dir is not None else None
    if render_audio or render_score:
        from .rendering_sheetsage2 import render_outputs, render_memory, validate_render_options
        formats, render_parts = validate_render_options(audio=render_audio, score=render_score, parts=render_parts)
    result = Transcriber(model, dtype=dtype).analyze(audio, output_dir, melody_only=melody_only, **kwargs)
    metadata = result.pop("_metadata", None)
    if metadata is None:
        metadata = dict(result)
    if render_audio or render_score:
        try:
            assets = Path(__file__).with_name("render_assets")
            if not assets.is_dir():
                source = getattr(model, "_source_snapshot", Path(model.config._name_or_path))
                if (source / "render_assets").is_dir():
                    assets = source / "render_assets"
                else:
                    from huggingface_hub import snapshot_download
                    assets = Path(snapshot_download(model.config._name_or_path,
                        revision=model.config._commit_hash, allow_patterns=["render_assets/**"],
                        **getattr(model, "_hub_resource_options", {}))) / "render_assets"
            # A missing score does not prevent a requested MIDI piano preview.
            rendered = None
            if render_audio or not result.get("abc_error"):
                options = dict(audio=render_audio, score=() if result.get("abc_error") else formats,
                               parts=render_parts, assets_dir=assets)
                if output_dir is None:
                    rendered = render_memory(midi=result["midi"], abc=result["abc"],
                                             duration=result["duration_seconds"], **options)
                else:
                    rendered = render_outputs(input_dir=output_dir, output_dir=output_dir, **options)
            result["rendered"] = rendered
            if output_dir is not None:
                metadata["rendered"] = rendered
            if formats and result.get("abc_error"):
                raise ValueError(f"ABC unavailable: {result['abc_error']}")
        except Exception as exc:
            result["render_error"] = str(exc)
            metadata["render_error"] = str(exc)
            if output_dir is not None:
                atomic_write_text(output_dir / "result.json", json.dumps(metadata, indent=2))
            location = f"saved to {output_dir.resolve()}" if output_dir is not None else "completed in memory"
            error = RuntimeError(f"Transcription {location}; rendering failed: {exc}")
            error.result = result
            raise error from exc
        if output_dir is not None:
            atomic_write_text(output_dir / "result.json", json.dumps(metadata, indent=2))
    if melody_only and not result.get("abc"):
        reason = result.get("abc_error") or "No ABC score was produced"
        error = RuntimeError(f"Melody-only ABC unavailable: {reason}")
        error.result = result
        raise error
    return result
