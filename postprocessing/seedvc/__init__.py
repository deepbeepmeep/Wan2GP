import os
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchaudio

from shared.utils import files_locator as fl
from shared.utils.download import process_download_defs


SEEDVC_CHECKPOINT_FILENAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
SEEDVC_CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
SEEDVC_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
SEEDVC_DEFAULT_STEPS = 25
SEEDVC_DEFAULT_CFG_RATE = 0.5
SEEDVC_SAMPLE_RATE = 22050
SEEDVC_MAX_REFERENCE_SECONDS = 25.0
SEEDVC_REPO_ID = "DeepBeepMeep/LTX-2"
SEEDVC_ROOT = "seed-vc"
SEEDVC_CHECKPOINT_DIR = SEEDVC_ROOT
SEEDVC_BIGVGAN_DIR = "bigvgan_v2_22khz_80band_256x"
SEEDVC_WHISPER_DIR = "whisper-small"
SEEDVC_BIGVGAN_FILES = ["config.json", "bigvgan_generator.pt"]
SEEDVC_WHISPER_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def query_download_def(root: str = SEEDVC_ROOT) -> list[dict]:
    return [
        {
            "repoId": SEEDVC_REPO_ID,
            "sourceFolderList": [root],
            "fileList": [[SEEDVC_CHECKPOINT_FILENAME, SEEDVC_CONFIG_FILENAME, SEEDVC_CAMPPLUS_FILENAME]],
        },
        {
            "repoId": SEEDVC_REPO_ID,
            "sourceFolderList": [SEEDVC_BIGVGAN_DIR],
            "fileList": [SEEDVC_BIGVGAN_FILES],
        },
        {
            "repoId": SEEDVC_REPO_ID,
            "sourceFolderList": [SEEDVC_WHISPER_DIR],
            "fileList": [SEEDVC_WHISPER_FILES],
        },
    ]


def download_assets(root: str = SEEDVC_ROOT) -> list[dict]:
    download_def = query_download_def(root)
    process_download_defs(download_def)
    return download_def


def _asset_paths(root: str = SEEDVC_ROOT) -> dict[str, str]:
    return {
        "checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_CHECKPOINT_FILENAME)),
        "config_path": fl.locate_file(os.path.join(root, SEEDVC_CONFIG_FILENAME)),
        "campplus_path": fl.locate_file(os.path.join(root, SEEDVC_CAMPPLUS_FILENAME)),
        "bigvgan_folder": fl.locate_folder(SEEDVC_BIGVGAN_DIR),
        "whisper_folder": fl.locate_folder(SEEDVC_WHISPER_DIR),
    }


def _closure_modules(fn) -> list[torch.nn.Module]:
    modules = []
    for cell in fn.__closure__ or []:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, torch.nn.Module):
            modules.append(value)
    return modules


def _make_mono(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    return waveform.mean(dim=0, keepdim=True)


def _save_mono_resampled(path: str, waveform: torch.Tensor, source_rate: int, target_rate: int = SEEDVC_SAMPLE_RATE, max_seconds: float | None = None) -> None:
    waveform = _make_mono(waveform)
    if int(source_rate) != int(target_rate):
        waveform = torchaudio.functional.resample(waveform, int(source_rate), int(target_rate))
    if max_seconds is not None:
        waveform = waveform[:, : int(round(float(max_seconds) * int(target_rate)))]
    torchaudio.save(path, waveform.clamp_(-1.0, 1.0), int(target_rate))


def _register_unmanaged_seedvc_tensors(modules) -> None:
    for module in modules:
        for submodule in module.modules():
            for attr in ("freqs_cis", "causal_mask", "mask_cache", "input_pos"):
                value = getattr(submodule, attr, None)
                if isinstance(value, torch.Tensor) and attr not in submodule._buffers:
                    delattr(submodule, attr)
                    submodule.register_buffer(attr, value, persistent=False)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        return tensor.device
    return torch.device("cpu")


def _runtime_device(pipe: dict[str, torch.nn.Module]) -> torch.device:
    for module in pipe.values():
        for submodule in module.modules():
            if hasattr(submodule, "_mm_manager"):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for module in pipe.values():
        return _module_device(module)
    return torch.device("cpu")


def _load_seedvc_app():
    try:
        from . import app_vc
    except ImportError as exc:
        raise ImportError("SeedVC support requires the bundled `postprocessing/seedvc` package files.") from exc
    return app_vc


class SeedVCVoiceConverter:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        campplus_path: str,
        bigvgan_folder: str,
        whisper_folder: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.checkpoint_path = os.fspath(checkpoint_path)
        self.config_path = os.fspath(config_path)
        self.campplus_path = os.fspath(campplus_path)
        self.bigvgan_folder = os.fspath(bigvgan_folder)
        self.whisper_folder = os.fspath(whisper_folder)
        self.dtype = dtype
        self._app_vc = None
        self._patched_config_path = None
        self._load()

    def _build_local_config(self) -> str:
        import yaml

        with open(self.config_path, "r", encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        config["model_params"]["vocoder"]["name"] = self.bigvgan_folder
        config["model_params"]["speech_tokenizer"]["name"] = self.whisper_folder
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", encoding="utf-8", delete=False)
        with tmp:
            yaml.safe_dump(config, tmp, sort_keys=False)
        self._patched_config_path = tmp.name
        return tmp.name

    def _load(self) -> None:
        app_vc = _load_seedvc_app()
        app_vc.device = torch.device("cpu")
        app_vc.load_custom_model_from_hf = self._load_custom_model_from_local_assets
        os.environ.setdefault("HF_HUB_CACHE", str(Path(self.campplus_path).parent / "hf_cache"))
        args = Namespace(checkpoint=self.checkpoint_path, config=self._build_local_config(), fp16=self.dtype == torch.float16, gpu=0)
        (
            app_vc.model,
            app_vc.semantic_fn,
            app_vc.vocoder_fn,
            app_vc.campplus_model,
            app_vc.to_mel,
            app_vc.mel_fn_args,
        ) = app_vc.load_models(args)
        app_vc.max_context_window = app_vc.sr // app_vc.hop_length * 30
        app_vc.overlap_wave_len = app_vc.overlap_frame_len * app_vc.hop_length
        self._app_vc = app_vc

        self.seedvc_model = torch.nn.ModuleDict({str(name): module for name, module in app_vc.model.items() if isinstance(module, torch.nn.Module)})
        self.semantic_modules = torch.nn.ModuleList(_closure_modules(app_vc.semantic_fn))
        self.campplus_model = app_vc.campplus_model
        self.vocoder_fn = app_vc.vocoder_fn
        _register_unmanaged_seedvc_tensors(self.pipe_modules().values())
        for module in self.pipe_modules().values():
            for submodule in module.modules():
                submodule._lock_dtype = None

    def pipe_modules(self) -> dict[str, torch.nn.Module]:
        pipe = {f"seedvc_{name}": module for name, module in self.seedvc_model.items()}
        if len(self.semantic_modules) == 1:
            pipe["seedvc_whisper_small"] = self.semantic_modules[0]
        else:
            pipe.update({f"seedvc_speech_tokenizer_{idx + 1}": module for idx, module in enumerate(self.semantic_modules)})
        if isinstance(self.campplus_model, torch.nn.Module):
            pipe["seedvc_campplus"] = self.campplus_model
        if isinstance(self.vocoder_fn, torch.nn.Module):
            pipe["seedvc_bigvgan"] = self.vocoder_fn
        return pipe

    def _load_custom_model_from_local_assets(self, repo_id, model_filename, config_filename=None):
        if repo_id == "funasr/campplus" and model_filename == SEEDVC_CAMPPLUS_FILENAME:
            return self.campplus_path
        raise FileNotFoundError(f"SeedVC asset is not declared for local loading: {repo_id}/{model_filename}")

    def forward(
        self,
        source_wav_path: str,
        target_wav_path: str,
        diffusion_steps: int = SEEDVC_DEFAULT_STEPS,
        cfg_rate: float = SEEDVC_DEFAULT_CFG_RATE,
    ) -> tuple[np.ndarray, int]:
        if self._app_vc is None:
            raise RuntimeError("SeedVC is not loaded.")
        self._app_vc.device = _runtime_device(self.pipe_modules())
        audio_tuple = None
        for result in self._app_vc.voice_conversion(
            source=source_wav_path,
            target=target_wav_path,
            diffusion_steps=int(diffusion_steps),
            length_adjust=1.0,
            inference_cfg_rate=float(cfg_rate),
        ):
            if isinstance(result, tuple) and len(result) == 2:
                _, audio_tuple = result
        if audio_tuple is None:
            raise RuntimeError("SeedVC produced no output.")
        sample_rate, samples = audio_tuple
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        peak = np.abs(samples).max(initial=0.0)
        if peak > 1.0:
            samples = samples / peak
        return samples, int(sample_rate)

    def convert_tensor(
        self,
        source_audio: torch.Tensor,
        source_rate: int,
        reference_audio: torch.Tensor,
        reference_rate: int,
        output_rate: int,
        diffusion_steps: int = SEEDVC_DEFAULT_STEPS,
        cfg_rate: float = SEEDVC_DEFAULT_CFG_RATE,
    ) -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source_22k.wav")
            target_path = os.path.join(tmpdir, "target_22k.wav")
            _save_mono_resampled(source_path, source_audio, source_rate)
            _save_mono_resampled(target_path, reference_audio, reference_rate, max_seconds=SEEDVC_MAX_REFERENCE_SECONDS)
            converted, converted_rate = self.forward(source_path, target_path, diffusion_steps=diffusion_steps, cfg_rate=cfg_rate)
        converted_tensor = torch.from_numpy(converted).float().unsqueeze(0)
        if int(converted_rate) != int(output_rate):
            converted_tensor = torchaudio.functional.resample(converted_tensor, int(converted_rate), int(output_rate))
        return converted_tensor.repeat(2, 1)


def get_model(dtype: torch.dtype = torch.float16, root: str = SEEDVC_ROOT) -> SeedVCVoiceConverter:
    return SeedVCVoiceConverter(**_asset_paths(root), dtype=dtype)


def get_pipe(profile_no=None, dtype: torch.dtype = torch.float16, root: str = SEEDVC_ROOT, model: SeedVCVoiceConverter | None = None) -> dict[str, torch.nn.Module]:
    seedvc_model = get_model(dtype=dtype, root=root) if model is None else model
    return seedvc_model.pipe_modules()


def get_cotenants_map(pipe: dict[str, torch.nn.Module]) -> dict[str, list[str]]:
    seedvc_keys = [key for key in pipe if str(key).startswith("seedvc_")]
    return {key: list(seedvc_keys) for key in seedvc_keys}
