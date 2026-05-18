import os
import io
import gc
import math
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from .gpt.model_v2 import UnifiedVoice
from .utils.maskgct_utils import build_semantic_model, build_semantic_codec
from .utils.front import TextNormalizer, TextTokenizer

from .s2mel.modules.commons import load_checkpoint2, MyModel
from .s2mel.modules.bigvgan import bigvgan
from .s2mel.modules.campplus.DTDNN import CAMPPlus
from .s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer
import safetensors
from transformers import SeamlessM4TFeatureExtractor
from transformers.cache_utils import StaticCache
import random
import torch.nn.functional as F
from mmgp import offload
from shared.llm_engines.cudagraph_kit import AutoRegressiveCudaGraphKit

_SEMANTIC_CODEC_FILENAME = "index_tts2_semantic_codec.safetensors"
_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
_W2V_BERT_FOLDER = "w2v-bert-2.0"
_FORCE_GPT_FP16 = True
_ABORT_ERROR = "INDEXTTS2_ABORT"
_MEL_TOKENS_PER_SOUND_TOKEN = 1.72
_MIN_CG_SOUND_SECONDS = 10.0
# Measured on local IndexTTS2 runs and multiplied by x3 safety margin.
# Calibration run (test_kvcache.zip): weighted avg ratio ~=9.8884, safety x3 => 29.67.
_CG_SOUND_TOKENS_PER_TEXT_TOKEN = 29.67
_LOG_KV_RATIO = str(os.environ.get("INDEXTTS2_LOG_KV_RATIO", "0")).strip().lower() in ("1", "true", "yes")


@contextmanager
def _quiet_load_output(show_logs):
    if show_logs:
        yield
        return
    with io.StringIO() as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False, show_load_logs=False,
            lm_decoder_engine="legacy",
            force_no_flash2=False,
            accel_allow_vllm_kernels=False,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """
        self.show_load_logs = bool(show_load_logs)
        self._load_log = print if self.show_load_logs else (lambda *args, **kwargs: None)
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            self._load_log(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        if _FORCE_GPT_FP16:
            self.use_fp16 = True
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.force_no_flash2 = bool(force_no_flash2)
        self.accel_allow_vllm_kernels = bool(accel_allow_vllm_kernels)
        self.use_torch_compile = use_torch_compile

        qwen_emo_dir = self.cfg.qwen_emo_path
        if not os.path.isabs(qwen_emo_dir) and not os.path.isdir(qwen_emo_dir):
            qwen_emo_dir = os.path.join(self.model_dir, qwen_emo_dir)
        with _quiet_load_output(self.show_load_logs):
            self.qwen_emo = QwenEmotion(qwen_emo_dir, lm_decoder_engine=lm_decoder_engine)
        self._load_log(">> qwen_emo weights restored from:", qwen_emo_dir)

        w2v_bert_dir = getattr(self.cfg, "w2v_bert_path", _W2V_BERT_FOLDER)
        if not os.path.isabs(w2v_bert_dir) and not os.path.isdir(w2v_bert_dir):
            w2v_bert_dir = os.path.join(os.path.dirname(self.model_dir), w2v_bert_dir)
        if not os.path.isdir(w2v_bert_dir):
            raise FileNotFoundError(
                f"w2v-bert-2.0 folder not found at '{w2v_bert_dir}'. "
                "Download it via the IndexTTS2 handler model files."
            )

        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        gpt_config_path = os.path.join(self.model_dir, "configs", "gpt_runtime_config.json")
        os.makedirs(os.path.dirname(gpt_config_path), exist_ok=True)
        with open(gpt_config_path, "w", encoding="utf-8") as gpt_cfg_writer:
            json.dump(OmegaConf.to_container(self.cfg.gpt, resolve=True), gpt_cfg_writer)
        with _quiet_load_output(self.show_load_logs):
            self.gpt = offload.fast_load_transformers_model(
                self.gpt_path,
                writable_tensors=False,
                modelClass=UnifiedVoice,
                forcedConfigPath=gpt_config_path,
                default_dtype=torch.float16 if self.use_fp16 else torch.float32,
                configKwargs={
                    "use_accel": self.use_accel,
                    "gpt_build_fp16": self.use_fp16,
                    "gpt_build_meta": True,
                    "force_no_flash2": self.force_no_flash2,
                    "accel_allow_vllm_kernels": self.accel_allow_vllm_kernels,
                },
            )
        self.gpt = self.gpt.to("cpu")
        if self.use_fp16:
            first_param = next(self.gpt.parameters(), None)
            if first_param is not None and first_param.dtype != torch.float16:
                self.gpt = self.gpt.half()
            self.gpt.eval()
        else:
            self.gpt.eval()
        self._load_log(">> GPT weights restored from:", self.gpt_path)

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from .s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                self._load_log(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                self._load_log(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self._load_log(f"{e!r}")
                self.use_cuda_kernel = False

        with _quiet_load_output(self.show_load_logs):
            self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_dir, local_files_only=True)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat), model_dir=w2v_bert_dir, use_fp16=self.use_fp16
        )
        self.semantic_model = self.semantic_model.to("cpu")
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to("cpu")
        self.semantic_std = self.semantic_std.to("cpu")
        self._load_log(">> semantic_model weights restored from:", w2v_bert_dir)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, _SEMANTIC_CODEC_FILENAME)
        if not os.path.isfile(semantic_code_ckpt):
            raise FileNotFoundError(
                f"semantic_codec checkpoint not found at '{semantic_code_ckpt}'. "
                "Download it via the IndexTTS2 handler model files."
            )
        with _quiet_load_output(self.show_load_logs):
            safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to("cpu")
        self.semantic_codec.eval()
        self._load_log('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        with _quiet_load_output(self.show_load_logs):
            s2mel, _, _, _ = load_checkpoint2(
                s2mel,
                None,
                s2mel_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
        self.s2mel = s2mel.to("cpu")
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            self._load_log(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            self._load_log(">> torch.compile optimization enabled successfully")
        
        self.s2mel.eval()
        self._load_log(">> s2mel weights restored from:", s2mel_path)

        # load campplus_model
        campplus_ckpt_path = os.path.join(self.model_dir, _CAMPPLUS_FILENAME)
        if not os.path.isfile(campplus_ckpt_path):
            raise FileNotFoundError(
                f"campplus checkpoint not found at '{campplus_ckpt_path}'. "
                "Download it via the IndexTTS2 handler model files."
            )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        with _quiet_load_output(self.show_load_logs):
            campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to("cpu")
        self.campplus_model.eval()
        self._load_log(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        with _quiet_load_output(self.show_load_logs):
            self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to("cpu")
        with _quiet_load_output(self.show_load_logs):
            self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        self._load_log(">> bigvgan weights restored from:", bigvgan_name)

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True)
        with _quiet_load_output(self.show_load_logs):
            self.normalizer.load()
        self._load_log(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        self._load_log(">> bpe model loaded from:", self.bpe_path)

        # åŠ è½½æœ¯è¯­è¯æ±‡è¡¨ï¼ˆå¦‚æžœå­˜åœ¨ï¼‰
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            with _quiet_load_output(self.show_load_logs):
                self.normalizer.load_glossary_from_yaml(self.glossary_path)
            self._load_log(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix), map_location="cpu")
        self.emo_matrix = emo_matrix
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix), map_location="cpu")
        self.spk_matrix = spk_matrix

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # ç¼“å­˜å‚è€ƒéŸ³é¢‘ï¼š
        self.cache_spk_entries = {}
        self.cache_emo_entries = {}
        self.cache_emo_text_vectors = {}
        self.abort_checker = None

        # è¿›åº¦å¼•ç”¨æ˜¾ç¤ºï¼ˆå¯é€‰ï¼‰
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        semantic_mean = self.semantic_mean.to(feat.device)
        semantic_std = self.semantic_std.to(feat.device)
        feat = (feat - semantic_mean) / semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        self._raise_if_aborted()
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)
        self._raise_if_aborted()

    def _abort_requested_runtime(self):
        checker = getattr(self, "abort_checker", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _raise_if_aborted(self):
        if self._abort_requested_runtime():
            raise RuntimeError(_ABORT_ERROR)

    def _clear_persistent_generation_cache(self):
        self.cache_spk_entries = {}
        self.cache_emo_entries = {}

    def _release_runtime_gpu_caches(self):
        gpt_model = getattr(self, "gpt", None)
        if gpt_model is not None and hasattr(gpt_model, "release_runtime_generation_cache"):
            gpt_model.release_runtime_generation_cache()
        qwen_model = getattr(self, "qwen_emo", None)
        if qwen_model is not None and hasattr(qwen_model, "release_cuda_graph"):
            qwen_model.release_cuda_graph()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio, device="cpu").unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    def _ensure_speaker_entry(self, spk_audio_prompt, verbose=False):
        key = str(spk_audio_prompt)
        cached = self.cache_spk_entries.get(key, None)
        if cached is not None:
            return cached

        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
        audio = audio.float()
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        codec_device = next(self.semantic_codec.parameters()).device
        spk_cond_for_codec = spk_cond_emb if spk_cond_emb.device == codec_device else spk_cond_emb.to(codec_device)
        _, S_ref = self.semantic_codec.quantize(spk_cond_for_codec)
        if S_ref.device != spk_cond_emb.device:
            S_ref = S_ref.to(spk_cond_emb.device)

        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        cached = {
            "speaker_cache_key": key,
            "spk_cond_emb": spk_cond_emb.detach().cpu(),
            "style": style.detach().cpu(),
            "ref_mel": ref_mel.detach().cpu(),
            "speaker_ref_codes": S_ref.detach().cpu(),
            "speaker_ref_lengths": ref_target_lengths.detach().cpu(),
        }
        self.cache_spk_entries[key] = cached
        del audio, audio_22k, audio_16k, inputs, input_features, attention_mask
        del spk_cond_emb, spk_cond_for_codec, S_ref, ref_mel, ref_target_lengths, feat, style
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        return cached

    def _ensure_emo_entry(self, emo_audio_prompt, verbose=False):
        key = str(emo_audio_prompt)
        cached = self.cache_emo_entries.get(key, None)
        if cached is not None:
            return cached
        spk_cached = self.cache_spk_entries.get(key, None)
        if isinstance(spk_cached, dict):
            spk_cond_emb = spk_cached.get("spk_cond_emb", None)
            if spk_cond_emb is not None:
                self.cache_emo_entries[key] = spk_cond_emb
                return spk_cond_emb
        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"].to(self.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        emo_cond_emb = emo_cond_emb.detach().cpu()
        self.cache_emo_entries[key] = emo_cond_emb
        del emo_audio, emo_inputs, emo_input_features, emo_attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        return emo_cond_emb

    def _infer_emotion_vector_cached(self, emo_text, log_if_default=True):
        key = str(emo_text or "").strip()
        key = re.sub(r"[\[\]]", " ", key).strip()
        if len(key) == 0:
            return None
        cached = self.cache_emo_text_vectors.get(key, None)
        if cached is not None:
            return list(cached)
        keyword_emotions = self.qwen_emo.try_keyword_emotions(key)
        if keyword_emotions is not None:
            emo_vector = [float(keyword_emotions.get(one, 0.0)) for one in self.qwen_emo.emotion_keys]
            self.cache_emo_text_vectors[key] = list(emo_vector)
            key_preview = re.sub(r"\s+", " ", key).strip()
            if len(key_preview) > 160:
                key_preview = key_preview[:157] + "..."
            print(f">> detected emotion from keywords: '{key_preview}' -> {keyword_emotions}")
            return list(emo_vector)
        emo_dict = self.qwen_emo.inference(key, log_if_default=log_if_default)
        emo_vector = list(emo_dict.values())
        self.cache_emo_text_vectors[key] = list(emo_vector)
        key_preview = re.sub(r"\s+", " ", key).strip()
        if len(key_preview) > 160:
            key_preview = key_preview[:157] + "..."
        print(f">> detected emotion from text: '{key_preview}' -> {emo_dict}")
        return list(emo_vector)

    def precache_reference_audio(self, audio_paths, verbose=False):
        self._raise_if_aborted()
        if audio_paths is None:
            return
        unique_paths = []
        seen = set()
        for one in audio_paths:
            if one is None:
                continue
            one_path = str(one)
            if len(one_path) == 0:
                continue
            if one_path in seen or one_path in self.cache_spk_entries:
                continue
            seen.add(one_path)
            unique_paths.append(one_path)
        if len(unique_paths) == 0:
            return

        pending = []
        for one_path in unique_paths:
            self._raise_if_aborted()
            audio, sr = self._load_and_cut_audio(one_path, 15, verbose)
            audio = audio.float()
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
            pending.append(
                {
                    "speaker_cache_key": one_path,
                    "audio_16k": audio_16k,
                    "audio_22k": audio_22k,
                }
            )
            del audio, sr

        # Phase 1: semantic/w2v + semantic codec + ref mel for all speakers.
        for item in pending:
            self._raise_if_aborted()
            audio_16k = item["audio_16k"]
            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            codec_device = next(self.semantic_codec.parameters()).device
            spk_cond_for_codec = spk_cond_emb if spk_cond_emb.device == codec_device else spk_cond_emb.to(codec_device)
            _, S_ref = self.semantic_codec.quantize(spk_cond_for_codec)
            if S_ref.device != spk_cond_emb.device:
                S_ref = S_ref.to(spk_cond_emb.device)

            ref_mel = self.mel_fn(item["audio_22k"].to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

            item["spk_cond_emb"] = spk_cond_emb.detach().cpu()
            item["speaker_ref_codes"] = S_ref.detach().cpu()
            item["speaker_ref_lengths"] = ref_target_lengths.detach().cpu()
            item["ref_mel"] = ref_mel.detach().cpu()

            del inputs, input_features, attention_mask, spk_cond_emb, spk_cond_for_codec, S_ref, ref_mel, ref_target_lengths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

        # Phase 2: campplus style for all speakers.
        for item in pending:
            self._raise_if_aborted()
            audio_16k = item["audio_16k"]
            ref_mel = item["ref_mel"].to(self.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))
            self.cache_spk_entries[item["speaker_cache_key"]] = {
                "speaker_cache_key": item["speaker_cache_key"],
                "spk_cond_emb": item["spk_cond_emb"],
                "style": style.detach().cpu(),
                "ref_mel": item["ref_mel"],
                "speaker_ref_codes": item["speaker_ref_codes"],
                "speaker_ref_lengths": item["speaker_ref_lengths"],
            }
            del feat, style, ref_mel
            item["audio_16k"] = None
            item["audio_22k"] = None
            item["spk_cond_emb"] = None
            item["speaker_ref_codes"] = None
            item["speaker_ref_lengths"] = None
            item["ref_mel"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

        pending.clear()
        del pending
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def precache_emotion_audio(self, audio_paths, verbose=False):
        self._raise_if_aborted()
        if audio_paths is None:
            return
        for one in audio_paths:
            self._raise_if_aborted()
            if one is None:
                continue
            one_path = str(one)
            if len(one_path) == 0:
                continue
            self._ensure_emo_entry(one_path, verbose=verbose)

    def precache_emotion_texts(self, emotion_texts):
        self._raise_if_aborted()
        if emotion_texts is None:
            return
        for one in emotion_texts:
            self._raise_if_aborted()
            if one is None:
                continue
            one_text = str(one).strip()
            if len(one_text) == 0:
                continue
            self._infer_emotion_vector_cached(one_text, log_if_default=False)

    @torch.no_grad()
    def synthesize_from_segment_payloads(self, segment_payloads, verbose=False):
        if segment_payloads is None or len(segment_payloads) == 0:
            return []
        diffusion_steps = 25
        inference_cfg_rate = 0.7
        results = []
        prompt_condition_cache = {}
        for item_index, item in enumerate(segment_payloads):
            pause_ms = 0
            payload = item
            if isinstance(item, dict):
                if "segment_payload" in item:
                    payload = item.get("segment_payload")
                elif "payload" in item:
                    payload = item.get("payload")
                try:
                    pause_ms = int(item.get("pause_after_ms", 0) or 0)
                except Exception:
                    pause_ms = 0
            if payload is None:
                continue
            latent = payload["latent"].to(self.device)
            codes = payload["codes"].to(self.device)
            code_lens = payload["code_lens"].to(self.device)
            ref_mel = payload["ref_mel"].to(self.device)
            style = payload["style"].to(self.device)
            speaker_cache_key = payload.get("speaker_cache_key", None)
            speaker_cache_key = str(speaker_cache_key) if speaker_cache_key is not None else f"__payload_{len(prompt_condition_cache)}"
            prompt_condition_cpu = prompt_condition_cache.get(speaker_cache_key, None)
            if prompt_condition_cpu is None:
                speaker_ref_codes = payload["speaker_ref_codes"].to(self.device)
                speaker_ref_lengths = payload["speaker_ref_lengths"].to(self.device)
                prompt_condition = self.s2mel.models["length_regulator"](
                    speaker_ref_codes,
                    ylens=speaker_ref_lengths,
                    n_quantizers=3,
                    f0=None,
                )[0]
                prompt_condition_cpu = prompt_condition.detach().cpu()
                prompt_condition_cache[speaker_cache_key] = prompt_condition_cpu
                del speaker_ref_codes, speaker_ref_lengths, prompt_condition
            prompt_condition = prompt_condition_cpu.to(self.device)
            latent = self.s2mel.models["gpt_layer"](latent)
            quantizer_device = next(self.semantic_codec.quantizer.parameters()).device
            codes_for_quantizer = codes.unsqueeze(1)
            if codes_for_quantizer.device != quantizer_device:
                codes_for_quantizer = codes_for_quantizer.to(quantizer_device)
            S_infer = self.semantic_codec.quantizer.vq2emb(codes_for_quantizer)
            if S_infer.device != latent.device:
                S_infer = S_infer.to(latent.device)
            S_infer = S_infer.transpose(1, 2)
            S_infer = S_infer + latent
            target_lengths = (code_lens * 1.72).long()
            cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            vc_target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                ref_mel,
                style,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1):].detach().cpu()
            if verbose:
                print(f"vc_target shape: {vc_target.shape}")
            if isinstance(item, dict):
                results.append({"vc_target": vc_target, "pause_after_ms": max(0, pause_ms)})
                if "segment_payload" in item:
                    item["segment_payload"] = None
                if "payload" in item:
                    item["payload"] = None
            else:
                results.append(vc_target)
            segment_payloads[item_index] = None
            if isinstance(payload, dict):
                payload.clear()
            del latent, codes, code_lens, prompt_condition, ref_mel, style, S_infer, cond, cat_condition, target_lengths, payload, prompt_condition_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        prompt_condition_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        return results

    @torch.no_grad()
    def synthesize_from_vc_targets(self, vc_targets, interval_silence=0, verbose=False):
        if vc_targets is None or len(vc_targets) == 0:
            return None
        sampling_rate = 22050
        wavs = []
        pause_after_samples = []
        for item_index, item in enumerate(vc_targets):
            pause_ms = 0
            vc_target = item
            if isinstance(item, dict):
                vc_target = item.get("vc_target", item.get("target", None))
                try:
                    pause_ms = int(item.get("pause_after_ms", 0) or 0)
                except Exception:
                    pause_ms = 0
            if vc_target is None:
                continue
            vc_target = vc_target.to(self.device)
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0).squeeze(1)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            if verbose:
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
            wavs.append(wav.cpu())
            pause_after_samples.append(max(0, int(round(sampling_rate * pause_ms / 1000.0))))
            if isinstance(item, dict):
                if "vc_target" in item:
                    item["vc_target"] = None
                if "target" in item:
                    item["target"] = None
            vc_targets[item_index] = None
            del vc_target, wav
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        if not wavs:
            return None
        if any(pause_after_samples):
            wavs_with_pauses = []
            for idx, one_wav in enumerate(wavs):
                wavs_with_pauses.append(one_wav)
                if idx >= len(wavs) - 1:
                    continue
                gap = pause_after_samples[idx] if idx < len(pause_after_samples) else 0
                if gap <= 0 and interval_silence > 0:
                    gap = int(round(sampling_rate * interval_silence / 1000.0))
                if gap > 0:
                    wavs_with_pauses.append(torch.zeros(one_wav.size(0), gap, device=one_wav.device, dtype=one_wav.dtype))
            wav = torch.cat(wavs_with_pauses, dim=1).cpu()
        else:
            wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
            wav = torch.cat(wavs, dim=1).cpu()
        wav_data = wav.type(torch.int16).numpy().T
        return (sampling_rate, wav_data)
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    # åŽŸå§‹æŽ¨ç†æ¨¡å¼
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, clear_cache_after=True, **generation_kwargs):
        if stream_return:
            gen = self.infer_generator(
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
            )
            def _wrapped():
                try:
                    for one in gen:
                        yield one
                finally:
                    if clear_cache_after:
                        self._clear_persistent_generation_cache()
                        self._release_runtime_gpu_caches()
            return _wrapped()
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                ))[0]
            except IndexError:
                return None
            finally:
                if clear_cache_after:
                    self._clear_persistent_generation_cache()
                    self._release_runtime_gpu_caches()

    def infer_generator(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0, **generation_kwargs):
        self._raise_if_aborted()
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_vector = self._infer_emotion_vector_cached(emo_text)
            if verbose:
                print(f"detected emotion vectors from text: {emo_vector}")

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # å¦‚æžœå‚è€ƒéŸ³é¢‘æ”¹å˜äº†ï¼Œæ‰éœ€è¦é‡æ–°ç”Ÿæˆ, æå‡é€Ÿåº¦
        speaker_entry = self._ensure_speaker_entry(spk_audio_prompt, verbose=verbose)
        spk_cond_emb = speaker_entry["spk_cond_emb"].to(self.device)
        style = speaker_entry["style"].to(self.device)
        ref_mel = speaker_entry["ref_mel"].to(self.device)
        speaker_cache_key = speaker_entry["speaker_cache_key"]
        speaker_ref_codes_cpu = speaker_entry["speaker_ref_codes"].contiguous()
        speaker_ref_lengths_cpu = speaker_entry["speaker_ref_lengths"].contiguous()

        weight_vector = None
        emovec_mat = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if style.device != weight_vector.device:
                weight_vector = weight_vector.to(style.device)
            spk_mats = [tmp.to(style.device) for tmp in self.spk_matrix]
            emo_mats = [tmp.to(style.device) for tmp in self.emo_matrix]
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in spk_mats]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, emo_mats)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)
            del spk_mats, emo_mats, random_index, emo_matrix
        emo_cond_emb = self._ensure_emo_entry(emo_audio_prompt, verbose=verbose).to(self.device)

        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens)
        segments_count = len(segments)
        segment_text_token_lengths = [max(1, len(self.tokenizer.convert_tokens_to_ids(one_segment))) for one_segment in segments]

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print( "     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")
                  
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        duration_seconds_hint = float(generation_kwargs.pop("duration_seconds", 0.0) or 0.0)
        shared_segment_generation_tokens = generation_kwargs.pop("cg_segment_generation_tokens", None)
        defer_s2mel = bool(generation_kwargs.pop("defer_s2mel", False))
        defer_bigvgan = bool(generation_kwargs.pop("defer_bigvgan", False))
        sampling_rate = 22050
        hop_size = int(self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"])
        ref_mel_cpu = ref_mel.detach().cpu().contiguous() if defer_s2mel and not stream_return else None
        style_cpu = style.detach().cpu().contiguous() if defer_s2mel and not stream_return else None
        cg_max_total_tokens = generation_kwargs.pop("cg_max_total_tokens", None)
        segment_generation_token_budget = max(1, int(max_mel_tokens))
        if shared_segment_generation_tokens is not None:
            try:
                segment_generation_token_budget = max(1, min(int(max_mel_tokens), int(shared_segment_generation_tokens)))
            except Exception:
                segment_generation_token_budget = max(1, int(max_mel_tokens))
        if getattr(self.gpt, "accel_engine", None) is not None and (cg_max_total_tokens is None or shared_segment_generation_tokens is None):
            try:
                mel_sr = float(self.cfg.s2mel["preprocess_params"]["sr"])
                mel_hop = float(self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"])
            except Exception:
                mel_sr = 22050.0
                mel_hop = 256.0
            sound_tokens_per_second = max(1.0, (mel_sr / max(1.0, mel_hop)) / _MEL_TOKENS_PER_SOUND_TOKEN)
            min_cg_sound_tokens = int(math.ceil(_MIN_CG_SOUND_SECONDS * sound_tokens_per_second))
            if duration_seconds_hint > 0:
                duration_sound_tokens = int(math.ceil(duration_seconds_hint * sound_tokens_per_second))
            else:
                duration_sound_tokens = int(max_mel_tokens)
            max_segment_text_tokens = max(1, max(segment_text_token_lengths) if len(segment_text_token_lengths) > 0 else int(max_text_tokens_per_segment))
            longest_segment_estimated_sound_tokens = int(math.ceil(max_segment_text_tokens * _CG_SOUND_TOKENS_PER_TEXT_TOKEN))
            capped_segment_sound_tokens = max(1, min(int(duration_sound_tokens), int(longest_segment_estimated_sound_tokens)))
            cg_generation_tokens = max(min_cg_sound_tokens, capped_segment_sound_tokens)
            if shared_segment_generation_tokens is None:
                segment_generation_token_budget = max(1, min(int(max_mel_tokens), int(cg_generation_tokens)))
            if cg_max_total_tokens is None:
                prompt_budget = int(getattr(self.gpt, "cond_num", 32) + max_segment_text_tokens + 3)
                cg_max_total_tokens = prompt_budget + max(1, int(segment_generation_token_budget))

        wavs = []
        deferred_segment_payloads = []
        deferred_vc_targets = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # for stream_return
        transformer_progress = None
        try:
            for seg_idx, sent in enumerate(segments):
                self._raise_if_aborted()
                try:
                    text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                    text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                    if verbose:
                        print(text_tokens)
                        print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                        # debug tokenizer
                        text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                        print("text_token_syms is same as segment tokens", text_token_syms == sent)

                    m_start_time = time.perf_counter()
                    with torch.no_grad():
                        with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                            emovec = self.gpt.merge_emovec(
                                spk_cond_emb,
                                emo_cond_emb,
                                torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                                torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                                alpha=emo_alpha
                            )

                        if emo_vector is not None:
                            emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                            # emovec = emovec_mat

                        codes, speech_conditioning_latent = self.gpt.inference_speech(
                            spk_cond_emb,
                            text_tokens,
                            emo_cond_emb,
                            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_vec=emovec,
                            abort_checker=self._abort_requested_runtime,
                            do_sample=True,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            num_return_sequences=autoregressive_batch_size,
                            length_penalty=length_penalty,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            max_generate_length=segment_generation_token_budget,
                            cg_max_total_tokens=cg_max_total_tokens,
                            **generation_kwargs
                        )

                    gpt_gen_time += time.perf_counter() - m_start_time
                    aborted_during_generation = self._abort_requested_runtime()
                    if not has_warned and not aborted_during_generation and (codes[:, -1] != self.stop_mel_token).any():
                        warnings.warn(
                            f"WARN: generation stopped due to exceeding `max_mel_tokens` ({segment_generation_token_budget}). "
                            f"Input text tokens: {text_tokens.shape[1]}. "
                            f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                            category=RuntimeWarning
                        )
                        has_warned = True

                    code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                    #                 if verbose:
                    #                     print(codes, type(codes))
                    #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    #                     print(f"code len: {code_lens}")

                    code_lens = []
                    max_code_len = 0
                    for code in codes:
                        if self.stop_mel_token not in code:
                            code_len = len(code)
                        else:
                            len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                            code_len = len_[0].item() if len_.numel() > 0 else len(code)
                        code_lens.append(code_len)
                        max_code_len = max(max_code_len, code_len)
                    codes = codes[:, :max_code_len]
                    code_lens = torch.LongTensor(code_lens)
                    code_lens = code_lens.to(self.device)
                    if _LOG_KV_RATIO:
                        text_tok_count = int(text_tokens.shape[-1])
                        ratio_value = float(max_code_len) / float(max(1, text_tok_count))
                        print(f"[KV_RATIO_SAMPLE] text_tokens={text_tok_count} sound_tokens={int(max_code_len)} ratio={ratio_value:.6f}")
                    if verbose:
                        print(codes, type(codes))
                        print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                        print(f"code len: {code_lens}")

                    m_start_time = time.perf_counter()
                    use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = self.gpt(
                            speech_conditioning_latent,
                            text_tokens,
                            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                            codes,
                            torch.tensor([codes.shape[-1]], device=text_tokens.device),
                            emo_cond_emb,
                            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                            emo_vec=emovec,
                            use_speed=use_speed,
                        )
                        gpt_forward_time += time.perf_counter() - m_start_time

                    if defer_s2mel and not stream_return:
                        target_lengths = (code_lens * 1.72).long()
                        deferred_segment_payloads.append(
                            {
                                "speaker_cache_key": speaker_cache_key,
                                "speaker_ref_codes": speaker_ref_codes_cpu,
                                "speaker_ref_lengths": speaker_ref_lengths_cpu,
                                "latent": latent.detach().cpu().contiguous(),
                                "codes": codes.detach().cpu().contiguous(),
                                "code_lens": code_lens.detach().cpu().contiguous(),
                                "text_token_count": int(text_tokens.shape[-1]),
                                "sound_token_count": int(max_code_len),
                                "ref_mel": ref_mel_cpu,
                                "style": style_cpu,
                                "estimated_samples": int(max(1, int(target_lengths.max().item())) * hop_size),
                            }
                        )
                        del target_lengths, latent, codes, code_lens, speech_conditioning_latent, emovec, text_tokens
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    dtype = None
                    with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                        m_start_time = time.perf_counter()
                        diffusion_steps = 25
                        inference_cfg_rate = 0.7
                        latent = self.s2mel.models['gpt_layer'](latent)
                        quantizer_device = next(self.semantic_codec.quantizer.parameters()).device
                        codes_for_quantizer = codes.unsqueeze(1)
                        if codes_for_quantizer.device != quantizer_device:
                            codes_for_quantizer = codes_for_quantizer.to(quantizer_device)
                        S_infer = self.semantic_codec.quantizer.vq2emb(codes_for_quantizer)
                        if S_infer.device != latent.device:
                            S_infer = S_infer.to(latent.device)
                        S_infer = S_infer.transpose(1, 2)
                        S_infer = S_infer + latent
                        target_lengths = (code_lens * 1.72).long()
                        speaker_ref_codes = speaker_ref_codes_cpu.to(latent.device)
                        speaker_ref_lengths = speaker_ref_lengths_cpu.to(latent.device)
                        prompt_condition = self.s2mel.models["length_regulator"](
                            speaker_ref_codes,
                            ylens=speaker_ref_lengths,
                            n_quantizers=3,
                            f0=None,
                        )[0]

                        cond = self.s2mel.models['length_regulator'](S_infer,
                                                                     ylens=target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]
                        cat_condition = torch.cat([prompt_condition, cond], dim=1)
                        vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                       torch.LongTensor([cat_condition.size(1)]).to(
                                                                           cond.device),
                                                                       ref_mel, style, None, diffusion_steps,
                                                                       inference_cfg_rate=inference_cfg_rate)
                        vc_target = vc_target[:, :, ref_mel.size(-1):]
                        s2mel_time += time.perf_counter() - m_start_time

                        if defer_bigvgan and not stream_return:
                            deferred_vc_targets.append(vc_target.detach().cpu())
                            del speaker_ref_codes, speaker_ref_lengths, prompt_condition, cond, cat_condition, target_lengths
                            del S_infer, codes_for_quantizer, latent, codes, code_lens, speech_conditioning_latent, emovec, text_tokens, vc_target
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        m_start_time = time.perf_counter()
                        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                        bigvgan_time += time.perf_counter() - m_start_time
                        wav = wav.squeeze(1)

                        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                        if verbose:
                            print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                        # wavs.append(wav[:, :-512])
                        wavs.append(wav.cpu())  # to cpu before saving
                        if stream_return:
                            yield wav.cpu()
                            if silence == None:
                                silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                            yield silence
                        del speaker_ref_codes, speaker_ref_lengths, prompt_condition, cond, cat_condition, target_lengths
                        del S_infer, codes_for_quantizer, latent, codes, code_lens, speech_conditioning_latent, emovec, text_tokens, wav, vc_target
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finally:
                    if transformer_progress is not None:
                        transformer_progress.update(1)
        finally:
            if transformer_progress is not None:
                transformer_progress.close()
            if weight_vector is not None:
                del weight_vector
            if emovec_mat is not None:
                del emovec_mat
            if "emo_cond_emb" in locals():
                del emo_cond_emb
            if "spk_cond_emb" in locals():
                del spk_cond_emb
            if "style" in locals():
                del style
            if "ref_mel" in locals():
                del ref_mel
            if "speaker_ref_codes_cpu" in locals():
                del speaker_ref_codes_cpu
            if "speaker_ref_lengths_cpu" in locals():
                del speaker_ref_lengths_cpu
            if "speaker_entry" in locals():
                del speaker_entry
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        end_time = time.perf_counter()

        if defer_s2mel and not stream_return:
            yield {"deferred_segment_payloads": deferred_segment_payloads, "audio_sampling_rate": sampling_rate}
            return

        if defer_bigvgan and not stream_return:
            yield {"deferred_vc_targets": deferred_vc_targets, "audio_sampling_rate": sampling_rate}
            return

        self._set_gr_progress(0.9, "saving audio...")
        if not wavs:
            return
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # ç›´æŽ¥ä¿å­˜éŸ³é¢‘åˆ°æŒ‡å®šè·¯å¾„ä¸­
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # è¿”å›žä»¥ç¬¦åˆGradioçš„æ ¼å¼è¦æ±‚
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir, lm_decoder_engine="legacy"):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        def _pretie_qwen_lm_head(state_dict, quantization_map=None, tied_weights_map=None):
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            if tied_weights_map is not None:
                tied_weights_map["lm_head.weight"] = "model.embed_tokens.weight"
            return state_dict, quantization_map, tied_weights_map

        model_path = os.path.join(self.model_dir, "model.safetensors")
        config_path = os.path.join(self.model_dir, "config.json")
        self.model = offload.fast_load_transformers_model(
            model_path,
            defaultConfigPath=config_path,
            default_dtype=torch.bfloat16,
            preprocess_sd=_pretie_qwen_lm_head,
            writable_tensors=False,
        )
        self.prompt = "æ–‡æœ¬æƒ…æ„Ÿåˆ†ç±»"
        self.cn_key_to_en = {
            "é«˜å…´": "happy",
            "æ„¤æ€’": "angry",
            "æ‚²ä¼¤": "sad",
            "ææƒ§": "afraid",
            "åæ„Ÿ": "disgusted",
            # TODO: the "ä½Žè½" (melancholic) emotion will always be mapped to
            # "æ‚²ä¼¤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "ä½Žè½": "melancholic",
            "æƒŠè®¶": "surprised",
            "è‡ªç„¶": "calm",
        }
        self.desired_vector_order = ["é«˜å…´", "æ„¤æ€’", "æ‚²ä¼¤", "ææƒ§", "åæ„Ÿ", "ä½Žè½", "æƒŠè®¶", "è‡ªç„¶"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "æ‚²ä¼¤" (sad) detection
            # to become "ä½Žè½" (melancholic) instead, to fix limitations mentioned above.
            "ä½Žè½",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0
        # Runtime canonical settings (override legacy vendor defaults).
        self.emotion_keys = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
        self.prompt = (
            "You are an emotion intensity classifier for TTS.\n"
            "Return ONLY a valid JSON object with numeric scores in [0.0, 1.2] for keys:\n"
            "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.\n"
            "No explanations, no markdown."
        )
        self.cn_key_to_en = {
            "\u9ad8\u5174": "happy",
            "\u6124\u6012": "angry",
            "\u60b2\u4f24": "sad",
            "\u6050\u60e7": "afraid",
            "\u53cd\u611f": "disgusted",
            "\u4f4e\u843d": "melancholic",
            "\u60ca\u8bb6": "surprised",
            "\u81ea\u7136": "calm",
        }
        self.en_to_cn = {v: k for k, v in self.cn_key_to_en.items()}
        self.melancholic_words = {"\u4f4e\u843d", "melancholy", "melancholic", "depression", "depressed", "gloomy"}
        self._keyword_aliases = self._build_keyword_aliases()
        self._cg_enabled = False
        self._cg_kit = None
        self._cg_cache = None
        self._cg_max_cache_tokens = 0
        self._cg_decode_cache_position = None
        self._cg_last_prepare_reused = False
        self.set_lm_decoder_engine(lm_decoder_engine)

    def set_lm_decoder_engine(self, lm_decoder_engine):
        engine = str(lm_decoder_engine or "legacy").strip().lower()
        self._cg_enabled = engine in ("cg", "cudagraph", "vllm")
        if not self._cg_enabled:
            self.release_cuda_graph()

    def _build_keyword_aliases(self):
        alias_map = {}
        alias_groups = {
            "happy": {"happy", "joy", "joyful", "cheerful", "excited", "delighted", "smiling", "positive", "happiness", "高兴", "开心", "喜悦"},
            "angry": {"angry", "anger", "mad", "furious", "irritated", "rage", "愤怒", "生气"},
            "sad": {"sad", "sadness", "sorrow", "unhappy", "depressed-sad", "悲伤", "伤心", "难过"},
            "afraid": {"afraid", "fear", "fearful", "scared", "anxious", "panic", "恐惧", "害怕"},
            "disgusted": {"disgusted", "disgust", "repulsed", "revolted", "反感", "厌恶"},
            "melancholic": {"melancholic", "melancholy", "gloomy", "depression", "depressed", "downcast", "低落", "忧郁"},
            "surprised": {"surprised", "surprise", "astonished", "amazed", "shocked", "惊讶", "吃惊"},
            "calm": {"calm", "neutral", "natural", "relaxed", "composed", "peaceful", "自然", "平静"},
        }
        for canonical, aliases in alias_groups.items():
            for alias in aliases:
                normalized = re.sub(r"\s+", " ", str(alias).strip().lower())
                if normalized:
                    alias_map[normalized] = canonical
                    alias_map[normalized.replace(" ", "")] = canonical
        for cn_key, en_key in self.cn_key_to_en.items():
            alias_map[cn_key.strip().lower()] = en_key
        return alias_map

    def try_keyword_emotions(self, text_input):
        raw = re.sub(r"[\[\]]", " ", str(text_input or "")).strip().lower()
        if len(raw) == 0:
            return None
        chunks = [chunk.strip() for chunk in re.split(r"[,，;/|+]+", raw) if chunk.strip()]
        if len(chunks) == 0:
            return None
        explicit_weights = {}
        unweighted_keys = []
        for chunk in chunks:
            normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
            parsed_key = None
            parsed_value = 1.0
            weighted_match = re.match(r"^(.+?)\s*[:=]\s*([-+]?\d*\.?\d+)\s*$", normalized_chunk)
            if weighted_match is not None:
                alias = re.sub(r"\s+", " ", weighted_match.group(1)).strip()
                parsed_key = self._keyword_aliases.get(alias) or self._keyword_aliases.get(alias.replace(" ", ""))
                try:
                    parsed_value = float(weighted_match.group(2))
                except Exception:
                    parsed_value = 1.0
            else:
                parsed_key = self._keyword_aliases.get(normalized_chunk) or self._keyword_aliases.get(normalized_chunk.replace(" ", ""))
            if parsed_key is None:
                return None
            if weighted_match is not None:
                explicit_weights[parsed_key] = max(float(explicit_weights.get(parsed_key, 0.0)), self.clamp_score(parsed_value))
                if parsed_key in unweighted_keys:
                    unweighted_keys.remove(parsed_key)
            elif parsed_key not in explicit_weights and parsed_key not in unweighted_keys:
                unweighted_keys.append(parsed_key)
        values = {key: 0.0 for key in self.emotion_keys}
        for key, value in explicit_weights.items():
            values[key] = float(value)
        remaining = 1.0 - sum(float(v) for v in explicit_weights.values())
        if len(unweighted_keys) > 0:
            shared = (remaining / float(len(unweighted_keys))) if remaining > 0 else 0.0
            for key in unweighted_keys:
                values[key] = float(shared)
        if all(v <= 0.0 for v in values.values()):
            values["calm"] = 1.0
        return values

    def release_cuda_graph(self):
        if self._cg_kit is not None:
            self._cg_kit.release()
        self._cg_kit = None
        self._cg_cache = None
        self._cg_max_cache_tokens = 0
        self._cg_decode_cache_position = None
        self._cg_last_prepare_reused = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _prepare_cuda_graph(self, max_cache_tokens):
        self._cg_last_prepare_reused = False
        if not self._cg_enabled or max_cache_tokens <= 0:
            self.release_cuda_graph()
            return False
        first_param = next(self.model.parameters(), None)
        if first_param is None or first_param.device.type != "cuda":
            self.release_cuda_graph()
            return False
        target_tokens = int(max_cache_tokens)
        if (
            self._cg_cache is not None
            and self._cg_kit is not None
            and self._cg_max_cache_tokens >= target_tokens
        ):
            self._cg_cache.reset()
            self._cg_last_prepare_reused = True
            if self._cg_decode_cache_position is None or self._cg_decode_cache_position.device != first_param.device:
                self._cg_decode_cache_position = torch.zeros(1, device=first_param.device, dtype=torch.long)
            return True
        self.release_cuda_graph()
        self._cg_cache = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=target_tokens,
            device=first_param.device,
            dtype=first_param.dtype,
        )
        self._cg_kit = AutoRegressiveCudaGraphKit("index_tts2_qwen_emo")
        self._cg_max_cache_tokens = target_tokens
        self._cg_decode_cache_position = torch.zeros(1, device=first_param.device, dtype=torch.long)
        return True

    def _cg_decode_step(self, token_ids, cache_position):
        outputs = self.model(
            input_ids=token_ids,
            past_key_values=self._cg_cache,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
        return outputs.logits

    def _generate_with_optional_cg(self, model_inputs, max_new_tokens):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        if input_ids.device.type != "cuda":
            return self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        graph_active = self._prepare_cuda_graph(int(input_ids.shape[1]) + int(max_new_tokens) + 2)
        if self._cg_enabled:
            prepare_status = "reused" if self._cg_last_prepare_reused else "new"
            state = "active" if graph_active else "inactive"
            if state == "inactive" or prepare_status == "new":
                print(f"[IndexTTS2][qwen_emo][cg] decode graph {state} ({prepare_status})")
        if not graph_active:
            return self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self._cg_cache.reset()
        prompt_len = int(input_ids.shape[1])
        prefill_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self._cg_cache,
            use_cache=True,
            return_dict=True,
            cache_position=torch.arange(prompt_len, device=input_ids.device, dtype=torch.long),
        )
        logits = prefill_outputs.logits[:, -1, :]
        eos_token_id = self.tokenizer.eos_token_id
        generated_tokens = []
        for step in range(int(max_new_tokens)):
            next_token = torch.argmax(logits, dim=-1)
            token_value = int(next_token[0].item())
            generated_tokens.append(token_value)
            if eos_token_id is not None and token_value == int(eos_token_id):
                break
            token_position = prompt_len + step
            if token_position >= self._cg_max_cache_tokens:
                break
            token_ids = next_token.view(1, 1).to(device=input_ids.device, dtype=input_ids.dtype)
            cache_position = self._cg_decode_cache_position.fill_(token_position)
            logits_step = self._cg_kit.run("decode", self._cg_decode_step, token_ids, cache_position)
            logits = logits_step[:, -1, :]
        if len(generated_tokens) == 0:
            return input_ids.clone()
        generated_tensor = torch.tensor(generated_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
        return torch.cat([input_ids, generated_tensor], dim=1)

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "æ‚²ä¼¤" (sad) vs "ä½Žè½" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["æ‚²ä¼¤"], content["ä½Žè½"] = content.get("ä½Žè½", 0.0), content.get("æ‚²ä¼¤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)

    # Runtime overrides for robust English/Chinese emotion JSON parsing.
    def convert(self, content, log_if_default=True):
        if not isinstance(content, dict):
            content = {}
        normalized = {}
        for k, v in content.items():
            if not isinstance(k, str):
                continue
            key = k.strip().strip('"').strip("'")
            normalized[key] = v
            normalized[key.lower()] = v

        emotion_dict = {}
        for en_key in self.emotion_keys:
            value = normalized.get(en_key, normalized.get(self.en_to_cn.get(en_key, ""), 0.0))
            try:
                value = float(value)
            except Exception:
                value = 0.0
            emotion_dict[en_key] = self.clamp_score(value)

        if all(val <= 0.0 for val in emotion_dict.values()):
            if log_if_default:
                print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0
        return emotion_dict

    def inference(self, text_input, log_if_default=True):
        clean_input = re.sub(r"[\[\]]", " ", str(text_input or "")).strip()
        if len(clean_input) == 0:
            clean_input = "calm"

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": clean_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self._generate_with_optional_cg(model_inputs, max_new_tokens=256)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        raw = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

        content = None
        json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        candidate = json_match.group(0) if json_match else raw
        try:
            content = json.loads(candidate)
        except Exception:
            parsed = {}
            for m in re.finditer(r'"?([A-Za-z_\u4e00-\u9fff]+)"?\s*[:=]\s*([-+]?\d*\.?\d+)', raw):
                try:
                    parsed[m.group(1)] = float(m.group(2))
                except Exception:
                    continue
            content = parsed

        text_input_lower = clean_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            sad_cn = self.en_to_cn["sad"]
            mel_cn = self.en_to_cn["melancholic"]
            sad_en = "sad"
            mel_en = "melancholic"
            content[sad_cn], content[mel_cn] = content.get(mel_cn, content.get(mel_en, 0.0)), content.get(sad_cn, content.get(sad_en, 0.0))
            content[sad_en], content[mel_en] = content.get(mel_en, content.get(mel_cn, 0.0)), content.get(sad_en, content.get(sad_cn, 0.0))

        return self.convert(content, log_if_default=log_if_default)


if __name__ == "__main__":
    prompt_wav = "examples/voice_01.wav"
    text = 'æ¬¢è¿Žå¤§å®¶æ¥ä½“éªŒindextts2ï¼Œå¹¶ç»™äºˆæˆ‘ä»¬æ„è§ä¸Žåé¦ˆï¼Œè°¢è°¢å¤§å®¶ã€‚'
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_cuda_kernel=False,
        use_torch_compile=True
    )
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
    char_size = 5
    import string
    time_buckets = []
    for i in range(10):
        text = ''.join(random.choices(string.ascii_letters, k=char_size))
        start_time = time.time()
        tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
        time_buckets.append(time.time() - start_time)
    print(time_buckets)
