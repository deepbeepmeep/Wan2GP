import glob
import librosa
import tqdm
import numpy as np
import torchaudio
import torch

# ignore all warning
import warnings

warnings.filterwarnings("ignore")

import concurrent.futures
import glob
import os
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class DNSMOSComputer:
    def __init__(
        self, primary_model_path, p808_model_path, device="cuda", device_id=0
    ) -> None:
        self.onnx_sess = ort.InferenceSession(
            primary_model_path, providers=["CUDAExecutionProvider"]
        )
        self.p808_onnx_sess = ort.InferenceSession(
            p808_model_path, providers=["CUDAExecutionProvider"]
        )
        self.onnx_sess.set_providers(["CUDAExecutionProvider"], [{"device_id": device_id}])
        self.p808_onnx_sess.set_providers(
            ["CUDAExecutionProvider"], [{"device_id": device_id}]
        )
        kwargs = {
            "sample_rate": 16000,
            "hop_length": 160,
            "n_fft": 320 + 1,
            "n_mels": 120,
            "mel_scale": "slaney",
        }
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**kwargs).to(f"cuda:{device_id}")

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_specgram = self.mel_transform(torch.Tensor(audio).cuda())
        mel_spec = mel_specgram.cpu()
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def compute(self, audio, sampling_rate, is_personalized_MOS=False):
        fs = SAMPLING_RATE
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=fs)
        elif sampling_rate != fs:
            # resample audio
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=fs)
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
        clip_dict = {
            "filename": "audio_clip",
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
        }
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict["SIG_raw"] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict["BAK_raw"] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)
        clip_dict["P808_MOS"] = np.mean(predicted_p808_mos)
        return clip_dict
