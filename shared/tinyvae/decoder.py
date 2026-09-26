"""Tiny preview decoders adapted from GOvEy1nw's Wan2GP PR #2107.

See LICENSES and sources.json for the upstream implementations and weights.
"""
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

REGISTRY = json.loads(Path(__file__).with_name('decoders.json').read_text(encoding='utf-8'))


def decoder_for(architecture, model_def):
    # These variants use different conditioning/latent contracts from the
    # baseline architectures. Register them only after checking that contract.
    if model_def.get('ltx2_msr') or model_def.get('joyai_echo') or model_def.get('ltx2_edit_anything'):
        return None
    return REGISTRY['architectures'].get(architecture)


def prepare_decoder(name, gen=None):
    from shared.utils import files_locator as fl
    from shared.utils.download import download_file, check_download_cancelled

    config = REGISTRY['decoders'][name]
    relative = 'preview_decoders/' + config['filename']
    path = fl.locate_file(relative, error_if_none=False)
    if path is None:
        path = fl.get_smart_download_location(config['filename'], 'preview_decoders')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        download_file(f"https://huggingface.co/{config['repo']}/resolve/main/{relative}", path, gen=gen)
    check_download_cancelled(gen)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != config['sha256']:
        raise ValueError(f'TinyVAE checkpoint checksum mismatch: {path}')
    return path


def frame_indices(count):
    return list(dict.fromkeys(i * count // min(4, count) for i in range(min(4, count))))


class TinyVAE(nn.Module):
    def __init__(self, name, net):
        super().__init__()
        self.name = name
        self.net = net
        self.config = REGISTRY['decoders'][name]
        self._convertWeightsFloatTo = None

    def forward(self, latents, image=False, abort_check=None):
        # Input is WanGP's existing C,T,H,W preview contract. Image batches
        # occupy T; temporal video decoders must not mix those images.
        value = latents.permute(1, 0, 2, 3).to(next(self.net.parameters()))
        if self.config['kind'] != 'video':
            decoded = torch.cat([self.net(frame) for frame in value[frame_indices(len(value))].split(1)])
        elif image:
            decoded = torch.cat([self.net.decode_video(frame[:, None], parallel=False)[:, 0] for frame in value[frame_indices(len(value))].split(1)])
        else:
            count = (len(value) - 1) * self.net.t_upscale + 1
            decoded = self.net.decode_video(value[None], parallel=False, output_indices=frame_indices(count), abort_check=abort_check)
            if decoded is None:
                return None
            decoded = decoded[0]
        if abort_check is not None and abort_check():
            return None
        height, width = decoded.shape[-2:]
        # Match the existing RGB preview strip and keep transport unchanged.
        decoded = F.interpolate(decoded, size=(200, max(1, round(width * 200 / height))), mode='bilinear', align_corners=False)
        pixels = decoded.clamp(0, 1).mul_(255).round_().to(torch.uint8).cpu()
        pixels = pixels.permute(2, 0, 3, 1).flatten(1, 2).numpy()
        return Image.fromarray(pixels)


def load_decoder(name, path):
    from mmgp import offload
    config = REGISTRY['decoders'][name]
    with torch.device('meta'):
        if config['kind'] == 'image':
            from .taesd import Decoder
            net = Decoder(config['channels'], use_midblock_gn=name == 'taef2')
        elif config['kind'] == 'h3':
            from safetensors import safe_open
            from .taeh3 import build_h3_decoder
            with safe_open(path, framework='pt', device='cpu') as weights:
                shapes = {key: torch.empty(weights.get_slice(key).get_shape(), device='meta') for key in weights.keys()}
            net = build_h3_decoder(shapes)
        else:
            from .taehv import TAEHV
            net = TAEHV(patch_size=config['patch'], latent_channels=config['channels'], decoder_time_upscale=config['temporal'])
        model = TinyVAE(name, net)

    def decoder_weights(state):
        if config['kind'] == 'video':
            state = net.patch_tgrow_layers({key: value for key, value in state.items() if key.startswith('decoder.')})
        return {'net.' + key: value for key, value in state.items()}

    offload.load_model_data(model, path, preprocess_sd=decoder_weights, writable_tensors=False, default_dtype=None)
    return model.eval().requires_grad_(False)
