"""Training-free spectral feature forecasting for MiniMax H3 denoising."""

import math
from dataclasses import dataclass

import torch


DEGREE = 4
RIDGE_LAMBDA = 0.1
BLEND_WEIGHT = 0.5
MAX_HISTORY = 8
WINDOW_SIZE = 2.0
FLEX_WINDOW = 0.75
TAIL_ACTUAL_STEPS = 1
MAX_CONSECUTIVE_FORECASTS = 1
CHUNK_BYTES = 32 * 1024 * 1024
MIN_FIT_POINTS = DEGREE + 1


@dataclass(slots=True)
class _HistoryEntry:
    coordinate: float
    feature_flat: torch.Tensor


class MiniMaxH3Spectrum:
    """Forecast H3's compact post-transformer target feature on selected steps."""

    def __init__(self, cache, sigmas):
        self.cache = cache
        evaluated = sigmas.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        sigma_min, sigma_max = float(evaluated.min()), float(evaluated.max())
        if sigma_max > sigma_min:
            evaluated = 2.0 * (evaluated - sigma_min) / (sigma_max - sigma_min) - 1.0
        else:
            evaluated.zero_()
        self.coordinates = tuple(float(value) for value in evaluated)
        self.total_steps = len(self.coordinates)
        self.warmup_steps = max(MIN_FIT_POINTS, int(cache.start_step))
        self._history = []
        self._feature_shape = None
        self._feature_dtype = None
        self._next_step = 0
        self._current_coordinate = 0.0
        self._actual = None
        self._observed = False
        self._adaptive_recompute = False
        self._current_window = WINDOW_SIZE
        self._consecutive_forecasts = 0

    @property
    def forecasting(self):
        if self._actual is None:
            raise RuntimeError("Spectrum step has not started")
        return not self._actual

    @property
    def history_length(self):
        return len(self._history)

    def begin_step(self, step):
        if self._actual is not None or step != self._next_step:
            raise RuntimeError("Spectrum step order is inconsistent with H3 denoising")
        self._current_coordinate = self.coordinates[step]
        tail_start = max(0, self.total_steps - TAIL_ACTUAL_STEPS)
        self._adaptive_recompute = False
        if step < self.warmup_steps or step >= tail_start or len(self._history) < MIN_FIT_POINTS:
            self._actual = True
        else:
            interval = max(1, math.floor(self._current_window))
            self._actual = (self._consecutive_forecasts + 1) % interval == 0
            self._adaptive_recompute = self._actual
            if not self._actual and self._consecutive_forecasts >= MAX_CONSECUTIVE_FORECASTS:
                self._actual = True
                self._adaptive_recompute = False
        self._observed = False

    def observe(self, feature, abort_callback):
        if not self._actual or self._observed:
            raise RuntimeError("Spectrum received an unexpected actual H3 feature")
        feature_shape = tuple(feature.shape)
        if self._feature_shape is None:
            self._feature_shape, self._feature_dtype = feature_shape, feature.dtype
        elif feature_shape != self._feature_shape or feature.dtype != self._feature_dtype:
            raise RuntimeError("MiniMax H3 target feature topology changed during Spectrum forecasting")
        abort_callback()
        archived = feature.detach().to(device="cpu", dtype=feature.dtype, copy=True, non_blocking=False).contiguous().reshape(-1)
        abort_callback()
        self._history.append(_HistoryEntry(self._current_coordinate, archived))
        if len(self._history) > MAX_HISTORY:
            self._history.pop(0)
        self._observed = True

    @staticmethod
    def _chebyshev_design(coordinates):
        x = coordinates.reshape(-1, 1).to(dtype=torch.float32)
        columns = [torch.ones_like(x), x]
        for _ in range(2, DEGREE + 1):
            columns.append(2.0 * x * columns[-1] - columns[-2])
        return torch.cat(columns, dim=1)

    def _weights(self):
        coordinates = torch.tensor([entry.coordinate for entry in self._history], dtype=torch.float32)
        design = self._chebyshev_design(coordinates)
        gram = design.T @ design + RIDGE_LAMBDA * torch.eye(DEGREE + 1, dtype=torch.float32)
        phi = self._chebyshev_design(torch.tensor([self._current_coordinate], dtype=torch.float32))
        spectral = (phi @ torch.linalg.solve(gram, design.T)).reshape(-1)
        linear = torch.zeros(len(self._history), dtype=torch.float32)
        previous, latest = self._history[-2].coordinate, self._history[-1].coordinate
        ratio = (self._current_coordinate - latest) / (latest - previous)
        linear[-2], linear[-1] = -ratio, 1.0 + ratio
        return BLEND_WEIGHT * spectral + (1.0 - BLEND_WEIGHT) * linear

    def predict(self, device, dtype, abort_callback):
        if self._actual or len(self._history) < MIN_FIT_POINTS:
            raise RuntimeError("Spectrum forecast requested without sufficient actual H3 history")
        weights = tuple(float(value) for value in self._weights())
        result = torch.empty(self._feature_shape, device=device, dtype=dtype)
        result_flat = result.reshape(-1)
        chunk_elements = max(1024, CHUNK_BYTES // torch.empty((), dtype=torch.float32).element_size())
        for offset in range(0, result_flat.numel(), chunk_elements):
            abort_callback()
            length = min(chunk_elements, result_flat.numel() - offset)
            accumulator = torch.zeros(length, device=device, dtype=torch.float32)
            for weight, entry in zip(weights, self._history, strict=True):
                abort_callback()
                source = entry.feature_flat.narrow(0, offset, length).to(device=device, dtype=torch.float32, non_blocking=False)
                accumulator.add_(source, alpha=weight)
                del source
            result_flat.narrow(0, offset, length).copy_(accumulator.to(dtype))
            del accumulator
        return result

    def finish_step(self):
        if self._actual is None or (self._actual and not self._observed):
            raise RuntimeError("Spectrum H3 step finished without its required feature")
        if self._actual:
            self._consecutive_forecasts = 0
            if self._adaptive_recompute:
                self._current_window = min(round(self._current_window + FLEX_WINDOW, 6), float(MAX_HISTORY))
        else:
            self._consecutive_forecasts += 1
            self.cache.skipped_steps += 1
        self._next_step += 1
        self._actual = None
        self._observed = False
        self._adaptive_recompute = False

    def reset(self):
        self._history.clear()
        self._feature_shape = None
        self._feature_dtype = None
        self._actual = None
        self._observed = False


__all__ = ["MiniMaxH3Spectrum"]
