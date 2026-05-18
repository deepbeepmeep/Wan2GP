from __future__ import annotations

import gc
import os
import tempfile
from typing import Any, Callable


_persistent_converter = None
_persistent_offloadobj = None
_persistent_profile = None
KEEP_ORIGINAL_AUDIO_OUTSIDE_TWO_SPEAKERS = True


def _release_runtime_objects(converter=None, offloadobj=None) -> None:
    import torch

    if offloadobj is not None:
        offloadobj.unload_all()
        offloadobj.release()
    del converter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def release_models() -> None:
    global _persistent_converter, _persistent_offloadobj, _persistent_profile

    _release_runtime_objects(_persistent_converter, _persistent_offloadobj)
    _persistent_converter = None
    _persistent_offloadobj = None
    _persistent_profile = None


def _get_runtime(persistent_models: bool, profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None):
    import torch
    from mmgp import offload
    from postprocessing import seedvc

    global _persistent_converter, _persistent_offloadobj, _persistent_profile

    profile_key = profile_no
    if _persistent_offloadobj is not None and _persistent_profile != profile_key:
        release_models()

    keep_alive = persistent_models
    if _persistent_offloadobj is None:
        converter = seedvc.get_model(dtype=torch.float16)
        pipe = seedvc.get_pipe(profile_no=profile_no, model=converter)
        offload_kwargs = {"coTenantsMap": seedvc.get_cotenants_map(pipe)}
        if init_pipe is not None:
            profile_no = init_pipe(pipe, offload_kwargs, profile_no)
        offloadobj = offload.profile(pipe, profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=torch.float16, verboseLevel=verbose_level, **offload_kwargs)
        if persistent_models:
            _persistent_converter = converter
            _persistent_offloadobj = offloadobj
            _persistent_profile = profile_key
    else:
        converter = _persistent_converter
        offloadobj = _persistent_offloadobj
        keep_alive = True

    return converter, offloadobj, keep_alive


def convert_audio_file(source_audio_path: str, voice_sample_path: str, output_path: str, *, persistent_models: bool = False, profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, diffusion_steps: int | None = None, cfg_rate: float | None = None) -> str:
    import torch
    import torchaudio
    from postprocessing import seedvc
    from shared.utils.audio_video import write_wav_file

    converter, offloadobj, keep_alive = _get_runtime(persistent_models, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe)
    try:
        source_audio, source_rate = torchaudio.load(os.fspath(source_audio_path))
        reference_audio, reference_rate = torchaudio.load(os.fspath(voice_sample_path))
        with torch.inference_mode():
            converted = converter.convert_tensor(
                source_audio,
                source_rate,
                reference_audio,
                reference_rate,
                output_rate=source_rate,
                diffusion_steps=seedvc.SEEDVC_DEFAULT_STEPS if diffusion_steps is None else diffusion_steps,
                cfg_rate=seedvc.SEEDVC_DEFAULT_CFG_RATE if cfg_rate is None else cfg_rate,
            )
        write_wav_file(output_path, converted, source_rate)
    finally:
        if offloadobj is not None:
            offloadobj.unload_all()
        if not keep_alive:
            _release_runtime_objects(converter, offloadobj)
    return output_path


def _make_temp_wav(output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=output_dir)
    os.close(fd)
    return path


def _fit_audio_mask_to_audio(mask, mask_sample_rate, target_sample_rate, target_length):
    import numpy as np
    from shared.utils.audio_video import resample_audio_array

    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if int(mask_sample_rate) != int(target_sample_rate):
        mask = resample_audio_array(mask, int(mask_sample_rate), int(target_sample_rate))
    mask = np.clip(mask, 0.0, 1.0)
    if mask.shape[0] < target_length:
        mask = np.pad(mask, (0, target_length - mask.shape[0]))
    return mask[:target_length]


def _merge_audio_files_to_wav(audio_paths, output_path, masks=None, mask_sample_rate=None):
    import numpy as np
    import soundfile as sf
    from shared.utils.audio_video import resample_audio_array, write_wav_file

    mixed_audio = None
    target_rate = 0
    target_channels = 1
    for track_no, audio_path in enumerate(audio_paths):
        audio_data, sample_rate = sf.read(os.fspath(audio_path), dtype="float32", always_2d=True)
        if mixed_audio is None:
            target_rate = int(sample_rate)
            target_channels = audio_data.shape[1]
            mixed_audio = np.zeros((audio_data.shape[0], target_channels), dtype=np.float32)
        elif int(sample_rate) != target_rate:
            audio_data = resample_audio_array(audio_data, int(sample_rate), target_rate)
            if audio_data.ndim == 1:
                audio_data = audio_data[:, None]
        if audio_data.shape[1] != target_channels:
            audio_data = np.repeat(audio_data[:, :1], target_channels, axis=1) if audio_data.shape[1] == 1 else audio_data[:, :target_channels]
        if masks is not None:
            audio_data = audio_data * _fit_audio_mask_to_audio(masks[track_no], mask_sample_rate, target_rate, audio_data.shape[0])[:, None]
        if audio_data.shape[0] > mixed_audio.shape[0]:
            mixed_audio = np.pad(mixed_audio, ((0, audio_data.shape[0] - mixed_audio.shape[0]), (0, 0)))
        mixed_audio[:audio_data.shape[0]] += audio_data
    return write_wav_file(output_path, np.clip(mixed_audio, -1.0, 1.0), target_rate)


class SeedVCBridge:
    MODE_OFF = 0
    MODE_V1 = 1
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    CURRENT_VERSION_LABEL = "SeedVC v1.0"

    _VERSIONS = {
        MODE_V1: "v1.0",
    }

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def mode_choices(cls) -> list[tuple[str, int]]:
        return [("Off", cls.MODE_OFF), ("v1.0", cls.MODE_V1)]

    @classmethod
    def persistence_choices(cls) -> list[tuple[str, int]]:
        return [("Unload after use", cls.PERSIST_UNLOAD), ("Persistent in RAM", cls.PERSIST_RAM)]

    def normalize_config(self, config: dict[str, Any] | None = None) -> tuple[int, int]:
        config = self.server_config if config is None else config
        mode = config.get("seedvc_mode", self.MODE_OFF)
        persistence = config.get("seedvc_persistence", self.PERSIST_UNLOAD)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = self.MODE_OFF
        try:
            persistence = int(persistence)
        except (TypeError, ValueError):
            persistence = self.PERSIST_UNLOAD
        if mode not in self._VERSIONS and mode != self.MODE_OFF:
            mode = self.MODE_OFF
        if persistence not in (self.PERSIST_UNLOAD, self.PERSIST_RAM):
            persistence = self.PERSIST_UNLOAD
        config["seedvc_mode"] = mode
        config["seedvc_persistence"] = persistence
        return mode, persistence

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None, int]:
        mode, persistence = self.normalize_config(config)
        return mode != self.MODE_OFF, self._VERSIONS.get(mode), persistence

    def enabled(self) -> bool:
        return self.settings()[0]

    def query_download_def(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        if enabled_only and not self.enabled():
            return []
        from postprocessing import seedvc
        return seedvc.query_download_def()

    def _assets_available(self) -> bool:
        from postprocessing import seedvc

        required_files = [
            os.path.join(seedvc.SEEDVC_ROOT, seedvc.SEEDVC_CHECKPOINT_FILENAME),
            os.path.join(seedvc.SEEDVC_ROOT, seedvc.SEEDVC_CONFIG_FILENAME),
            os.path.join(seedvc.SEEDVC_ROOT, seedvc.SEEDVC_CAMPPLUS_FILENAME),
            *[os.path.join(seedvc.SEEDVC_BIGVGAN_DIR, filename) for filename in seedvc.SEEDVC_BIGVGAN_FILES],
            *[os.path.join(seedvc.SEEDVC_WHISPER_DIR, filename) for filename in seedvc.SEEDVC_WHISPER_FILES],
        ]
        return all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in required_files)

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None) -> bool:
        download_defs = self.query_download_def()
        if not download_defs or self._assets_available():
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        for download_def in download_defs:
            process_files(**download_def)
        return True

    def _replace_two_speaker_audio_file(self, source_audio_path: str, voice_sample_path: str, output_path: str, *, voice_sample2_path: str, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, prefix: str = "seedvc") -> str:
        import numpy as np
        import soundfile as sf
        from preprocessing.speakers_separator import extract_dual_audio
        from shared.utils.audio_video import cleanup_temp_audio_files, normalize_audio_pair_volumes_to_temp_files

        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        split_track1 = _make_temp_wav(output_dir, f"{prefix}_speaker1_")
        split_track2 = _make_temp_wav(output_dir, f"{prefix}_speaker2_")
        converted_track1 = _make_temp_wav(output_dir, f"{prefix}_speaker1_seedvc_")
        converted_track2 = _make_temp_wav(output_dir, f"{prefix}_speaker2_seedvc_")
        temp_tracks = [split_track1, split_track2, converted_track1, converted_track2]
        try:
            _, speaker_masks, mask_sample_rate = extract_dual_audio(source_audio_path, split_track1, split_track2, verbose=verbose_level >= 2, return_masks=True, speech_masks_only=True)
            mask_values = list(speaker_masks.values())
            info1, info2 = sf.info(split_track1), sf.info(split_track2)
            active_mask1 = _fit_audio_mask_to_audio(mask_values[0], mask_sample_rate, info1.samplerate, info1.frames)
            active_mask2 = _fit_audio_mask_to_audio(mask_values[1], mask_sample_rate, info2.samplerate, info2.frames)
            normalized_track1, normalized_track2, _ = normalize_audio_pair_volumes_to_temp_files(split_track1, split_track2, output_dir=output_dir, prefix=f"{prefix}_norm_", active_mask1=active_mask1, active_mask2=active_mask2)
            temp_tracks += [normalized_track1, normalized_track2]
            self.replace_audio_file(normalized_track1, voice_sample_path, converted_track1, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe)
            self.replace_audio_file(normalized_track2, voice_sample2_path, converted_track2, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe)
            converted_info1, converted_info2 = sf.info(converted_track1), sf.info(converted_track2)
            converted_mask1 = _fit_audio_mask_to_audio(mask_values[0], mask_sample_rate, converted_info1.samplerate, converted_info1.frames)
            converted_mask2 = _fit_audio_mask_to_audio(mask_values[1], mask_sample_rate, converted_info2.samplerate, converted_info2.frames)
            normalized_converted_track1, normalized_converted_track2, _ = normalize_audio_pair_volumes_to_temp_files(converted_track1, converted_track2, output_dir=output_dir, prefix=f"{prefix}_converted_norm_", active_mask1=converted_mask1, active_mask2=converted_mask2)
            temp_tracks += [normalized_converted_track1, normalized_converted_track2]
            merge_tracks, merge_masks = [normalized_converted_track1, normalized_converted_track2], mask_values
            if KEEP_ORIGINAL_AUDIO_OUTSIDE_TWO_SPEAKERS:
                merge_tracks.append(source_audio_path)
                merge_masks.append(np.clip(1.0 - np.maximum(mask_values[0], mask_values[1]), 0.0, 1.0))
            return _merge_audio_files_to_wav(merge_tracks, output_path, masks=merge_masks, mask_sample_rate=mask_sample_rate)
        finally:
            cleanup_temp_audio_files(temp_tracks)

    def replace_audio_file(self, source_audio_path: str, voice_sample_path: str, output_path: str, *, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, voice_sample2_path: str | None = None, speaker_count: int = 1, prefix: str = "seedvc") -> str:
        enabled, _, persistence = self.settings()
        if not enabled:
            raise RuntimeError("SeedVC voice replacement is disabled in Configuration > Extensions.")
        self.download(process_files)
        if int(speaker_count) == 2:
            if voice_sample2_path is None:
                raise RuntimeError("Two-speaker SeedVC voice replacement requires a second voice sample.")
            return self._replace_two_speaker_audio_file(source_audio_path, voice_sample_path, output_path, voice_sample2_path=voice_sample2_path, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, prefix=prefix)
        return convert_audio_file(source_audio_path, voice_sample_path, output_path, persistent_models=persistence == self.PERSIST_RAM, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe)

    def replace_audio_tracks(self, audio_tracks: list[str], voice_sample_path: str | None, output_dir: str, prefix: str, *, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, voice_sample2_path: str | None = None, speaker_count: int = 1) -> tuple[list[str], list[str]]:
        if voice_sample_path is None or len(audio_tracks) == 0:
            return audio_tracks, []
        converted_tracks = []
        for track_no, audio_track in enumerate(audio_tracks):
            output_path = _make_temp_wav(output_dir, f"{prefix}_seedvc_track{track_no}_")
            converted_tracks.append(self.replace_audio_file(audio_track, voice_sample_path, output_path, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, voice_sample2_path=voice_sample2_path, speaker_count=speaker_count, prefix=f"{prefix}_track{track_no}"))
        return converted_tracks, converted_tracks

    def release_vram(self) -> None:
        release_models()
