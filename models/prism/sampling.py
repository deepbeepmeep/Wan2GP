"""LightX2V audio integration adapted from FreeVideo 40525196 (Apache-2.0).

See NOTICE. Keep native BF16 conditioning on CPU instead of FreeVideo's FP8
cache; MMGP manages all weights, and shared attention selects the kernel.
"""
import torch

from shared.utils.phase_progress import check_abort


def audio_substeps(transformer, audio, contexts, start, end, training, cache, fps):
    times = torch.stack([start + (end - start) * (i / 4) for i in range(5)])
    sigmas = training[(training[:, None] - times).abs().argmin(0)]
    # The reference passes an explicit zero timestep at the final endpoint;
    # its scheduler snaps that to the last training sigma, rather than None.
    for index in range(4):
        check_abort()
        outputs = transformer(None, audio, contexts, None, (times[index:index + 1] * 1000).to(audio.device), fps,
                              False, cache, audio_only=True)
        conditional, unconditional = (prediction.float() for prediction in outputs)
        prediction = unconditional + 5 * (conditional - unconditional)
        audio.add_(prediction, alpha=float(sigmas[index + 1] - sigmas[index]))
        del outputs, conditional, unconditional, prediction
        check_abort()
