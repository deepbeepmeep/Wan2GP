from pathlib import Path
import os, tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
import gc

from audio_separator.separator import Separator

def _resolve_separator_output(out_file, output_dir: Path) -> Path:
    out = Path(out_file)
    return out if out.is_absolute() else output_dir / out


def _separator_stem_key(path_or_stem) -> str:
    return "_".join(part for part in Path(path_or_stem).stem.lower().split("_") if part)


def _prepare_separator_input(src_path: str, min_seconds: float):
    duration = librosa.get_duration(path=src_path)
    use_path = src_path
    temp_path = None

    if duration < min_seconds:
        # Load (resample) and pad in memory
        y, sr = librosa.load(src_path, sr=None, mono=False)
        if y.ndim == 1:  # ensure shape (channels, samples)
            y = y[np.newaxis, :]
        target_len = int(min_seconds * sr)
        pad = max(0, target_len - y.shape[1])
        if pad:
            y = np.pad(y, ((0, 0), (0, pad)), mode="constant")

        # Write a temp WAV for the separator
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, y.T, sr)  # soundfile expects (frames, channels)
        use_path = temp_path

    return use_path, temp_path


def extract_vocal_and_background_stems(src_path: str, vocals_dst_path: str, background_dst_path: str, min_seconds: float = 8) -> tuple[str, str]:
    default_device = torch.get_default_device()
    torch.set_default_device('cpu')

    vocals_dst = Path(vocals_dst_path)
    background_dst = Path(background_dst_path)
    vocals_dst.parent.mkdir(parents=True, exist_ok=True)
    background_dst.parent.mkdir(parents=True, exist_ok=True)

    sep = None
    temp_path = None
    try:
        use_path, temp_path = _prepare_separator_input(src_path, min_seconds)
        sep = Separator(
            output_dir=str(vocals_dst.parent),
            output_format=(vocals_dst.suffix.lstrip(".") or "wav"),
            model_file_dir="ckpts/roformer/" #model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        )
        sep.load_model()
        out_files = sep.separate(use_path, {"Vocals": vocals_dst.stem, "Instrumental": background_dst.stem})
        out_paths = [_resolve_separator_output(out_file, vocals_dst.parent) for out_file in out_files]
        by_stem = {_separator_stem_key(out_path): str(out_path) for out_path in out_paths}
        return by_stem[_separator_stem_key(vocals_dst)], by_stem[_separator_stem_key(background_dst)]
    finally:
        if sep is not None:
            del sep
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        torch.cuda.empty_cache()
        gc.collect()
        torch.set_default_device(default_device)


def get_vocals(src_path: str, dst_path: str, min_seconds: float = 8) -> str:
    """
    If the source audio is shorter than `min_seconds`, pad with trailing silence
    in a temporary file, then run separation and save only the vocals to dst_path.
    Returns the full path to the vocals file.
    """

    default_device = torch.get_default_device()
    torch.set_default_device('cpu')

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    sep = None
    temp_path = None
    try:
        use_path, temp_path = _prepare_separator_input(src_path, min_seconds)
        # Run separation: emit only the vocals, with your exact filename
        sep = Separator(
            output_dir=str(dst.parent),
            output_format=(dst.suffix.lstrip(".") or "wav"),
            output_single_stem="Vocals",
            model_file_dir="ckpts/roformer/" #model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        )
        sep.load_model()
        out_files = sep.separate(use_path, {"Vocals": dst.stem})

        return str(_resolve_separator_output(out_files[0], dst.parent))
    finally:
        if sep is not None:
            del sep
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        torch.cuda.empty_cache()
        gc.collect()
        torch.set_default_device(default_device)

# Example:
# final = extract_vocals("in/clip.mp3", "out/vocals.wav")
# print(final)

