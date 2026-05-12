import numpy as np


DEFAULT_TRANSIENT_MAX_SECONDS = 0.1
DEFAULT_TRANSIENT_CONTEXT_SILENCE_SECONDS = 0.2
DEFAULT_TRANSIENT_WINDOW_SECONDS = 0.01
DEFAULT_TRANSIENT_SILENCE_THRESHOLD = 0.015


def _mono(audio_np: np.ndarray) -> np.ndarray:
    return audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np


def _window_settings(sample_rate: int, max_transient_seconds: float, context_silence_seconds: float, window_seconds: float) -> tuple[int, int, int]:
    window = max(1, int(window_seconds * sample_rate))
    max_noise_windows = max(1, int(np.ceil(max_transient_seconds * sample_rate / window)))
    min_silence_windows = max(1, int(np.ceil(context_silence_seconds * sample_rate / window)))
    return window, max_noise_windows, min_silence_windows


def _rms_windows(mono: np.ndarray, window: int, frame_count: int, offset: int = 0) -> np.ndarray:
    return np.array([np.sqrt(np.mean(mono[offset + i * window : offset + (i + 1) * window].astype(np.float64) ** 2)) for i in range(frame_count)])


def _debug_print(debug: bool, label: str, message: str) -> None:
    if debug:
        print(f"[{label}] {message}")


def trim_leading_transient_noise(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    max_transient_seconds: float = DEFAULT_TRANSIENT_MAX_SECONDS,
    context_silence_seconds: float = DEFAULT_TRANSIENT_CONTEXT_SILENCE_SECONDS,
    window_seconds: float = DEFAULT_TRANSIENT_WINDOW_SECONDS,
    threshold: float = DEFAULT_TRANSIENT_SILENCE_THRESHOLD,
    debug: bool = False,
    label: str = "Audio Cleaning",
) -> np.ndarray:
    mono = _mono(audio_np)
    window, max_noise_windows, min_silence_windows = _window_settings(sample_rate, max_transient_seconds, context_silence_seconds, window_seconds)
    frame_count = min(len(mono) // window, max_noise_windows + min_silence_windows)
    if frame_count < max_noise_windows + min_silence_windows:
        return audio_np
    active = _rms_windows(mono, window, frame_count) > threshold
    if not active[0]:
        return audio_np
    for silence_start in range(1, max_noise_windows + 1):
        if not active[silence_start : silence_start + min_silence_windows].any():
            trim_end = silence_start * window
            _debug_print(debug, label, f"Trimmed leading transient noise ({trim_end / sample_rate:.2f}s)")
            return audio_np[trim_end:]
    return audio_np


def trim_trailing_transient_noise(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    max_transient_seconds: float = DEFAULT_TRANSIENT_MAX_SECONDS,
    context_silence_seconds: float = DEFAULT_TRANSIENT_CONTEXT_SILENCE_SECONDS,
    window_seconds: float = DEFAULT_TRANSIENT_WINDOW_SECONDS,
    threshold: float = DEFAULT_TRANSIENT_SILENCE_THRESHOLD,
    debug: bool = False,
    label: str = "Audio Cleaning",
) -> np.ndarray:
    mono = _mono(audio_np)
    window, max_noise_windows, min_silence_windows = _window_settings(sample_rate, max_transient_seconds, context_silence_seconds, window_seconds)
    frame_count = min(len(mono) // window, max_noise_windows + min_silence_windows)
    if frame_count < max_noise_windows + min_silence_windows:
        return audio_np
    offset = len(mono) - frame_count * window
    active = _rms_windows(mono, window, frame_count, offset=offset) > threshold
    if not active[-1]:
        return audio_np
    for noise_start in range(frame_count - 1, frame_count - max_noise_windows - 1, -1):
        if not active[noise_start - min_silence_windows : noise_start].any():
            trim_start = offset + noise_start * window
            if trim_start <= 0 or trim_start >= len(audio_np):
                return audio_np
            _debug_print(debug, label, f"Trimmed trailing transient noise ({(len(audio_np) - trim_start) / sample_rate:.2f}s)")
            return audio_np[:trim_start]
    return audio_np


def mute_isolated_transient_noise(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    max_transient_seconds: float = DEFAULT_TRANSIENT_MAX_SECONDS,
    context_silence_seconds: float = DEFAULT_TRANSIENT_CONTEXT_SILENCE_SECONDS,
    window_seconds: float = DEFAULT_TRANSIENT_WINDOW_SECONDS,
    threshold: float = DEFAULT_TRANSIENT_SILENCE_THRESHOLD,
    debug: bool = False,
    label: str = "Audio Cleaning",
) -> np.ndarray:
    mono = _mono(audio_np)
    window, max_noise_windows, min_silence_windows = _window_settings(sample_rate, max_transient_seconds, context_silence_seconds, window_seconds)
    frame_count = len(mono) // window
    if frame_count < max_noise_windows + 2 * min_silence_windows:
        return audio_np
    active = _rms_windows(mono, window, frame_count) > threshold
    muted = audio_np.copy()
    active_start = None
    muted_count = 0
    for idx, is_active in enumerate(active):
        if is_active and active_start is None:
            active_start = idx
        elif not is_active and active_start is not None:
            if idx - active_start <= max_noise_windows:
                prev_start = max(0, active_start - min_silence_windows)
                next_end = min(frame_count, idx + min_silence_windows)
                if not active[prev_start:active_start].any() and not active[idx:next_end].any() and active_start > 0 and idx < frame_count:
                    muted[active_start * window : idx * window] = 0
                    muted_count += 1
            active_start = None
    _debug_print(debug and muted_count > 0, label, f"Muted {muted_count} isolated transient noise segment(s)")
    return muted


def trim_leading_noise_before_speech(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    speech_threshold: float = 0.03,
    max_leading_seconds: float = 1.0,
    keep_silence_seconds: float = 0.1,
    window_seconds: float = DEFAULT_TRANSIENT_WINDOW_SECONDS,
    debug: bool = False,
    label: str = "Audio Cleaning",
) -> np.ndarray:
    mono = _mono(audio_np)
    window = max(1, int(window_seconds * sample_rate))
    frame_count = min(len(mono) // window, max(1, int(max_leading_seconds * sample_rate / window)))
    if frame_count == 0:
        return audio_np
    strong_windows = np.where(_rms_windows(mono, window, frame_count) > speech_threshold)[0]
    if len(strong_windows) == 0:
        return audio_np
    first_speech = int(strong_windows[0]) * window
    keep_samples = int(keep_silence_seconds * sample_rate)
    trim_end = max(0, first_speech - keep_samples)
    if trim_end <= 0:
        return audio_np
    _debug_print(debug, label, f"Trimmed leading low-level noise before speech ({trim_end / sample_rate:.2f}s)")
    return audio_np[trim_end:]


def ensure_trailing_silence(audio_np: np.ndarray, sample_rate: int, min_silence_seconds: float, *, threshold: float = DEFAULT_TRANSIENT_SILENCE_THRESHOLD) -> np.ndarray:
    if min_silence_seconds <= 0:
        return audio_np
    mono = _mono(audio_np)
    window = max(1, int(DEFAULT_TRANSIENT_WINDOW_SECONDS * sample_rate))
    frame_count = len(mono) // window
    if frame_count == 0:
        return audio_np
    active = _rms_windows(mono, window, frame_count) > threshold
    active_windows = np.where(active)[0]
    if len(active_windows) == 0:
        return audio_np
    last_active_end = min(len(audio_np), (int(active_windows[-1]) + 1) * window)
    existing_silence = len(audio_np) - last_active_end
    target_silence = int(min_silence_seconds * sample_rate)
    missing_silence = target_silence - existing_silence
    if missing_silence <= 0:
        return audio_np
    pad_shape = (missing_silence,) if audio_np.ndim == 1 else (missing_silence, audio_np.shape[1])
    return np.concatenate([audio_np, np.zeros(pad_shape, dtype=audio_np.dtype)], axis=0)


def trim_after_silence_boundary(
    audio_np: np.ndarray,
    sample_rate: int,
    earliest_seconds: float,
    *,
    search_seconds: float = 2.0,
    min_silence_seconds: float = 0.18,
    keep_silence_seconds: float = 0.12,
    window_seconds: float = DEFAULT_TRANSIENT_WINDOW_SECONDS,
    threshold: float = DEFAULT_TRANSIENT_SILENCE_THRESHOLD,
    debug: bool = False,
    label: str = "Audio Cleaning",
) -> np.ndarray:
    if earliest_seconds <= 0 or search_seconds <= 0 or len(audio_np) == 0:
        return audio_np
    mono = _mono(audio_np)
    window = max(1, int(window_seconds * sample_rate))
    earliest_sample = max(0, int(earliest_seconds * sample_rate))
    if earliest_sample >= len(mono):
        return audio_np
    offset = (earliest_sample // window) * window
    search_end = min(len(mono), earliest_sample + int(search_seconds * sample_rate))
    frame_count = max(0, (search_end - offset) // window)
    min_silence_windows = max(1, int(np.ceil(min_silence_seconds * sample_rate / window)))
    if frame_count < min_silence_windows:
        return audio_np
    active = _rms_windows(mono, window, frame_count, offset=offset) > threshold
    for idx in range(0, len(active) - min_silence_windows + 1):
        if not active[idx : idx + min_silence_windows].any():
            cut_sample = min(len(audio_np), offset + idx * window + int(keep_silence_seconds * sample_rate))
            if cut_sample <= 0 or cut_sample >= len(audio_np):
                return audio_np
            _debug_print(debug, label, f"Trimmed chunk tail at silence boundary ({cut_sample / sample_rate:.2f}s)")
            return audio_np[:cut_sample]
    return audio_np
