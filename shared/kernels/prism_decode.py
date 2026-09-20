"""Select bit-exact PTQ1 decode fusions before CUDA graph capture."""
import statistics

import torch
from torch._subclasses.fake_tensor import FakeTensor

_choices = {}


def select_decode(input, raw, signs, bias, rows, grouped_shape, native):
    key = (input.device, input.dtype, input.shape[-1], rows, bias is not None, tuple(grouped_shape))
    if key in _choices:
        return _choices[key]
    if isinstance(input, FakeTensor) or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        # Do not freeze an untuned choice when capture is the first caller.
        return 0
    with torch.inference_mode(), torch.cuda.device(input.device):
        def reference():
            rotated = native.prism_hadamard(input, signs, False, grouped_shape)
            return native.linear(raw, "PTQ1_0", (rows, input.shape[-1]), rotated, bias, input.dtype)

        expected = reference()
        candidates = {0: reference}
        for tile in (1, 2, 4):
            def candidate(tile=tile):
                return native.prism_decode(input, raw, signs, bias, rows, grouped_shape, tile)
            # Same four-warp FP32 reduction and intermediate dtype rounding.
            torch.testing.assert_close(candidate(), expected, rtol=0, atol=0)
            candidates[tile] = candidate
        graphs = {}
        outputs = {}
        try:
            for tile, fn in candidates.items():
                for _ in range(3):
                    fn()
                graph = torch.cuda.CUDAGraph()
                graphs[tile] = graph
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        outputs[tile] = fn()
            timings = {tile: [] for tile in candidates}
            for repeat in range(5):
                for tile in list(candidates)[repeat % 4:] + list(candidates)[:repeat % 4]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8):
                        graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    timings[tile].append(start.elapsed_time(end) / 128)
            medians = {tile: statistics.median(values) for tile, values in timings.items()}
            best = min(medians, key=medians.get)
            # Retain the established kernel when differences are within noise.
            choice = best if medians[best] < medians[0] * .97 else 0
        finally:
            outputs.clear()
            graphs.clear()
            graph = None
        _choices[key] = choice
        print(f"[GGUF][Prism] Decode {rows}x{input.shape[-1]}: "
              f"{'Fused, Row Tile ' + str(choice) if choice else 'Existing Kernel'} "
              f"({medians[0]*1000:.2f} -> {medians[choice]*1000:.2f} us).")
        return choice
