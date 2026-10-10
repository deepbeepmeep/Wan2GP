"""Checkpoint-native token constraints and windowed repetition penalties."""

import torch

from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE


class YuE2LogitsProcessor:
    def __init__(self, sampling, phase, direct=False):
        self.sampling, self.phase, self.direct = sampling, phase, direct

    def __call__(self, logits, history):
        settings = self.sampling
        scores = logits.to(torch.bfloat16).clone() if self.direct else logits.float().clone()
        end = ABC_END if self.phase == "abc" else MUSIC_END
        end_score = scores[:, end].clone()
        if self.phase == "abc":
            scores[:, EOD:] = -torch.inf
        else:
            scores[:, :CODEC_OFFSET] = -torch.inf
            scores[:, CODEC_OFFSET + CODEC_SIZE:] = -torch.inf
        scores[:, end] = -torch.inf if len(history) < settings.min_tokens else end_score
        if history:
            recent = torch.tensor(history[-settings.penalty_window:], device=scores.device)[None]
            frequencies = torch.zeros_like(scores)
            frequencies.scatter_add_(1, recent, torch.ones_like(recent, dtype=scores.dtype))
            alpha = settings.repetition_penalty ** frequencies
            scores = torch.where(scores < 0, scores * alpha, scores / alpha)
        if settings.temperature == 0:
            best = scores.argmax(-1, keepdim=True)
            return torch.full_like(scores, -torch.inf).scatter_(1, best, 0).float()
        scores.div_(settings.temperature)
        if settings.top_k:
            threshold = scores.topk(settings.top_k).values[:, -1:]
            scores.masked_fill_(scores < threshold, -torch.inf)
        if settings.top_p < 1:
            values, indices = scores.sort(descending=True)
            probabilities = values.softmax(-1)
            removed = probabilities.cumsum(-1) - probabilities > settings.top_p
            removed[:, :3 if self.direct else 1] = False
            values.masked_fill_(removed, -torch.inf)
            scores = values.scatter(-1, indices, values)
        return scores.float()
