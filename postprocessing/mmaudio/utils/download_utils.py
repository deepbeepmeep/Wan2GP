import hashlib
import logging
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger()

links = [
    {
        'name': 'mmaudio_small_16k.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_16k.pth',
        'md5': 'af93cde404179f58e3919ac085b8033b',
    },
    {
        'name': 'mmaudio_small_44k.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_44k.pth',
        'md5': 'babd74c884783d13701ea2820a5f5b6d',
    },
    {
        'name': 'mmaudio_medium_44k.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_medium_44k.pth',
        'md5': '5a56b6665e45a1e65ada534defa903d0',
    },
    {
        'name': 'mmaudio_large_44k.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k.pth',
        'md5': 'fed96c325a6785b85ce75ae1aafd2673'
    },
    {
        'name': 'mmaudio_large_44k_v2.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth',
        'md5': '01ad4464f049b2d7efdaa4c1a59b8dfe'
    },
    {
        'name': 'v1-16.pth',
        'url': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth',
        'md5': '69f56803f59a549a1a507c93859fd4d7'
    },
    {
        'name': 'best_netG.pt',
        'url': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt',
        'md5': 'eeaf372a38a9c31c362120aba2dde292'
    },
    {
        'name': 'v1-44.pth',
        'url': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth',
        'md5': 'fab020275fa44c6589820ce025191600'
    },
    {
        'name': 'synchformer_state_dict.pth',
        'url':
        'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth',
        'md5': '5b2f5594b0730f70e41e549b7c94390c'
    },
]


def download_model_if_needed(model_path: Path):
    base_name = model_path.name

    for link in links:
        if link['name'] == base_name:
            target_link = link
            break
    else:
        raise ValueError(f'No link found for {base_name}')

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists() or hashlib.md5(open(model_path,
                                                   'rb').read()).hexdigest() != target_link['md5']:
        log.info(f'Downloading {base_name} to {model_path}...')
        r = requests.get(target_link['url'], stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(model_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            raise RuntimeError('Error while downloading %s' % base_name)
