import gc
import logging
import os

import torch

from .eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from .model.flow_matching import FlowMatching
from .model.networks import MMAudio, get_my_mmaudio
from .model.sequence_config import SequenceConfig
from .model.utils.features_utils import FeaturesUtils
from shared.utils import files_locator as fl

persistent_offloadobj = None
persistent_model_id = None


def _resolve_mmaudio_path(path):
    if path is None:
        return None
    path_str = str(path)
    if os.path.isabs(path_str):
        if os.path.isfile(path_str):
            return path_str
        raise FileNotFoundError(f"MMAudio file not found: {path_str}")
    if os.path.isfile(path_str):
        return path_str
    located = fl.locate_file(path_str, error_if_none=False)
    if located is not None:
        return located
    basename = os.path.basename(path_str)
    return fl.locate_file(os.path.join("mmaudio", basename))


def _load_state_dict(model_path, device):
    model_path = str(model_path)
    if model_path.lower().endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(model_path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def get_model(persistent_models = False, verboseLevel = 1, model_name = None, model_path = None) -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    global device, persistent_offloadobj, persistent_net, persistent_features_utils, persistent_seq_cfg, persistent_model_id

    log = logging.getLogger()

    device =  'cpu' #"cuda"
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    # else:
    #     log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.bfloat16

    if model_name is None:
        model_name = "large_44k_v2"
    if model_name not in all_model_cfg:
        raise ValueError(f"Unknown MMAudio model '{model_name}'. Available: {', '.join(all_model_cfg.keys())}")
    model: ModelConfig = all_model_cfg[model_name]
    # model.download_if_needed()

    setup_eval_logging()

    seq_cfg = model.seq_cfg
    resolved_model_path = _resolve_mmaudio_path(model_path or model.model_path)
    resolved_vae_path = _resolve_mmaudio_path(model.vae_path)
    resolved_synchformer_ckpt = _resolve_mmaudio_path(model.synchformer_ckpt)
    resolved_bigvgan_path = _resolve_mmaudio_path(model.bigvgan_16k_path) if model.bigvgan_16k_path else None
    model_id = (model_name, os.path.normcase(str(resolved_model_path)))

    if persistent_offloadobj is not None and persistent_model_id != model_id:
        persistent_offloadobj.unload_all()
        persistent_offloadobj.release()
        persistent_offloadobj = None
        persistent_net = None
        persistent_features_utils = None
        persistent_seq_cfg = None
        persistent_model_id = None

    if persistent_offloadobj == None:
        from accelerate import init_empty_weights
        # with init_empty_weights():
        net: MMAudio = get_my_mmaudio(model.model_name)
        net.load_weights(_load_state_dict(resolved_model_path, device))
        net.to(device, dtype).eval()
        log.info(f'Loaded weights from {resolved_model_path}')
        feature_utils = FeaturesUtils(tod_vae_ckpt=resolved_vae_path,
                                    synchformer_ckpt=resolved_synchformer_ckpt,
                                    enable_conditions=True,
                                    mode=model.mode,
                                    bigvgan_vocoder_ckpt=resolved_bigvgan_path,
                                    need_vae_encoder=False)
        feature_utils = feature_utils.to(device, dtype).eval()
        feature_utils.device = "cuda"

        pipe = { "net" : net, "clip" : feature_utils.clip_model, "syncformer" : feature_utils.synchformer, "vocode" : feature_utils.tod.vocoder, "vae" : feature_utils.tod.vae }
        from mmgp import offload
        offloadobj = offload.profile(pipe, profile_no=4, verboseLevel=2)
        if persistent_models:
            persistent_offloadobj = offloadobj
            persistent_net = net
            persistent_features_utils = feature_utils
            persistent_seq_cfg = seq_cfg
            persistent_model_id = model_id

    else:
        offloadobj = persistent_offloadobj  
        net = persistent_net 
        feature_utils = persistent_features_utils
        seq_cfg = persistent_seq_cfg

    if not persistent_models:
        persistent_offloadobj = None
        persistent_net = None
        persistent_features_utils = None
        persistent_seq_cfg = None
        persistent_model_id = None

    return net, feature_utils, seq_cfg, offloadobj

@torch.inference_mode()
def video_to_audio(video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float, save_path , persistent_models = False, audio_file_only = False, verboseLevel = 1, model_name = None, model_path = None):

    global device

    net, feature_utils, seq_cfg, offloadobj = get_model(persistent_models, verboseLevel, model_name=model_name, model_path=model_path )

    rng = torch.Generator(device="cuda")
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength,
                      offloadobj = offloadobj
                      )
    audio = audios.float().cpu()[0]


    if audio_file_only:
        import torchaudio
        torchaudio.save(save_path, audio.unsqueeze(0) if audio.dim() == 1 else audio, seq_cfg.sampling_rate)
    else:
        make_video(video, video_info, save_path, audio, sampling_rate=seq_cfg.sampling_rate)

    offloadobj.unload_all()
    if not persistent_models:
        offloadobj.release()

    torch.cuda.empty_cache()
    gc.collect()
    return save_path
