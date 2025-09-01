import gc
import logging

import torch

from .eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from .model.flow_matching import FlowMatching
from .model.networks import MMAudio, get_my_mmaudio
from .model.sequence_config import SequenceConfig
from .model.utils.features_utils import FeaturesUtils

persistent_offloadobj = None

def get_model(persistent_models = False, verboseLevel = 1) -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    global device, persistent_offloadobj, persistent_net, persistent_features_utils, persistent_seq_cfg

    log = logging.getLogger()

    device =  'cpu' #"cuda"
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    # else:
    #     log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.bfloat16

    model: ModelConfig = all_model_cfg['large_44k_v2']
    # model.download_if_needed()

    setup_eval_logging()

    seq_cfg = model.seq_cfg
    if persistent_offloadobj == None:
        from accelerate import init_empty_weights
        # with init_empty_weights():
        net: MMAudio = get_my_mmaudio(model.model_name)
        net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
        net.to(device, dtype).eval()
        log.info(f'Loaded weights from {model.model_path}')
        feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                    synchformer_ckpt=model.synchformer_ckpt,
                                    enable_conditions=True,
                                    mode=model.mode,
                                    bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
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

    return net, feature_utils, seq_cfg, offloadobj

@torch.inference_mode()
def video_to_audio(video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float, save_path , persistent_models = False, audio_file_only = False, verboseLevel = 1):

    global device

    net, feature_utils, seq_cfg, offloadobj = get_model(persistent_models, verboseLevel )

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
