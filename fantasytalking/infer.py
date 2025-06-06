# Copyright Alibaba Inc. All Rights Reserved.

from transformers import Wav2Vec2Model, Wav2Vec2Processor

from .model import FantasyTalkingAudioConditionModel
from .utils import get_audio_features
import gc, torch

def parse_audio(audio_path, num_frames, fps = 23, device = "cuda"):
    fantasytalking = FantasyTalkingAudioConditionModel(None, 768, 2048).to(device)
    from mmgp import offload
    from accelerate import init_empty_weights
    from fantasytalking.model import AudioProjModel

    torch.set_grad_enabled(False) 

    with init_empty_weights():
        proj_model = AudioProjModel( 768, 2048)
    offload.load_model_data(proj_model, "ckpts/fantasy_proj_model.safetensors")
    proj_model.to("cpu").eval().requires_grad_(False)

    wav2vec_model_dir = "ckpts/wav2vec"
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, device_map="cpu").eval().requires_grad_(False)
    wav2vec.to(device)
    proj_model.to(device)
    audio_wav2vec_fea = get_audio_features( wav2vec, wav2vec_processor, audio_path, fps, num_frames )

    audio_proj_fea = proj_model(audio_wav2vec_fea)
    pos_idx_ranges = fantasytalking.split_audio_sequence( audio_proj_fea.size(1), num_frames=num_frames )
    audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding( audio_proj_fea, pos_idx_ranges, expand_length=4 )  # [b,21,9+8,768]    
    wav2vec, proj_model= None, None
    gc.collect()
    torch.cuda.empty_cache()

    return audio_proj_split, audio_context_lens