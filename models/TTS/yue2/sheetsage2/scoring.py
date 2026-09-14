"""On-demand music transcription, with MMGP residency and WanGP progress."""
import gc
import json
import time
from pathlib import Path

import torch
from mmgp import offload
from tqdm import tqdm

from shared.utils.offload_registry import register_offloadobj, unregister_offloadobj


@torch.inference_mode()
def score_audio(filename, checkpoint, melody_only, callback, abort_fn):
    from .configuration_sheetsage2 import SheetSage2Config
    from .modeling_sheetsage2 import SheetSage2Model
    from .pipeline_sheetsage2 import Transcriber

    progress = None
    phase = ""
    window = ""
    last_update = 0.0
    handles = []
    manager = None
    model = None
    transcriber = None

    def report(title, completed, total, force=False):
        nonlocal progress, phase, last_update
        if abort_fn():
            raise InterruptedError("SheetSage2 scoring interrupted")
        if title != phase:
            if progress is not None:
                progress.close()
            phase = title
            unit = "layer" if "Encoding" in title else "token" if "Transcribing" in title else "step"
            progress = tqdm(total=total, desc=title, unit=unit)
            force = True
        progress.update(completed - progress.n)
        now = time.monotonic()
        if callback is not None and (force or now - last_update >= 1 / 3):
            last_update = now
            callback(step_idx=max(0, completed - 1), override_num_inference_steps=total, progress_title=title, denoising_extra=title, progress_unit=progress.unit + "s")
            if abort_fn():
                raise InterruptedError("SheetSage2 scoring interrupted")

    def layer_hook(index):
        def hook(module, inputs, output):
            total = len(model.encoder.layers)
            report("Encoding Source Song" + window, index + 1, total, index + 1 == total)
        return hook

    def decoder_hook(module, inputs, output):
        if abort_fn():
            raise InterruptedError("SheetSage2 scoring interrupted")

    def on_progress(event):
        nonlocal window
        stage = event["stage"]
        if stage == "encoding":
            window = f" {event['window']}/{event['windows']}" if event["windows"] > 1 else ""
            report("Encoding Source Song" + window, 0, len(model.encoder.layers), True)
        elif stage == "decoding":
            report("Transcribing Music Score" + window, event["tokens"], model.max_output_seq_len)
        elif stage == "notation":
            report("Writing ABC Score", 0, 1, True)
        elif stage == "complete":
            report("Writing ABC Score", 1, 1, True)

    try:
        report("Loading SheetSage2 / MERT2", 0, 1, True)
        loading = offload.LoadingCallback(abort_fn, lambda phase, completed, total, model_id: report(f"Loading - {phase} SheetSage2 / MERT2", completed, total, True))
        config = SheetSage2Config(**json.loads(Path(__file__).with_name("config.json").read_text()))
        with torch.device("meta"):
            model = SheetSage2Model(config)
        with offload.loading_context(loading, {checkpoint: "text_encoder"}):
            offload.load_model_data(model, checkpoint, writable_tensors=False, default_dtype=None)
        model.eval().requires_grad_(False)
        model._offload_hooks = ["encode", "decode"]
        manager = offload.profile({"text_encoder": model}, profile_no=3, budgets={"text_encoder": 0}, pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=None, verboseLevel=1, loading_callback=loading)
        register_offloadobj("yue2_scoring", manager)
        handles = [layer.register_forward_hook(layer_hook(index)) for index, layer in enumerate(model.encoder.layers)]
        handles += [layer.register_forward_hook(decoder_hook) for layer in model.decoder.layers]
        transcriber = Transcriber(model)
        transcriber.device = torch.device("cuda")
        result = transcriber.analyze(filename, melody_only=melody_only, progress=on_progress)
        for warning in result["warnings"]:
            print(f"[SheetSage2] {warning}", flush=True)
        if result.get("abc_error") or not result["abc"]:
            raise ValueError(f"SheetSage2 could not produce a usable score: {result.get('abc_error')}")
        return result["abc"], result["midi"]
    except offload.LoadingCancelled:
        raise InterruptedError("SheetSage2 loading interrupted") from None
    finally:
        for handle in handles:
            handle.remove()
        if progress is not None:
            progress.close()
        if manager is not None:
            unregister_offloadobj("yue2_scoring", manager)
            manager.release()
        transcriber = model = manager = None
        gc.collect()
        torch.cuda.empty_cache()
