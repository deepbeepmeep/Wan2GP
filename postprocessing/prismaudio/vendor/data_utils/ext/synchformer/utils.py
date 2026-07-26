from hashlib import md5
from pathlib import Path

import requests
from tqdm import tqdm

PARENT_LINK = 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
FNAME2LINK = {
    # S3: Synchability: AudioSet (run 2)
    '24-01-22T20-34-52.pt':
    f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/24-01-22T20-34-52.pt',
    'cfg-24-01-22T20-34-52.yaml':
    f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/cfg-24-01-22T20-34-52.yaml',
    # S2: Synchformer: AudioSet (run 2)
    '24-01-04T16-39-21.pt':
    f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt',
    'cfg-24-01-04T16-39-21.yaml':
    f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml',
    # S2: Synchformer: AudioSet (run 1)
    '23-08-28T11-23-23.pt':
    f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/23-08-28T11-23-23.pt',
    'cfg-23-08-28T11-23-23.yaml':
    f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/cfg-23-08-28T11-23-23.yaml',
    # S2: Synchformer: LRS3 (run 2)
    '23-12-23T18-33-57.pt':
    f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.pt',
    'cfg-23-12-23T18-33-57.yaml':
    f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/cfg-23-12-23T18-33-57.yaml',
    # S2: Synchformer: VGS (run 2)
    '24-01-02T10-00-53.pt':
    f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/24-01-02T10-00-53.pt',
    'cfg-24-01-02T10-00-53.yaml':
    f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/cfg-24-01-02T10-00-53.yaml',
    # SparseSync: ft VGGSound-Full
    '22-09-21T21-00-52.pt':
    f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/22-09-21T21-00-52.pt',
    'cfg-22-09-21T21-00-52.yaml':
    f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/cfg-22-09-21T21-00-52.yaml',
    # SparseSync: ft VGGSound-Sparse
    '22-07-28T15-49-45.pt':
    f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/22-07-28T15-49-45.pt',
    'cfg-22-07-28T15-49-45.yaml':
    f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/cfg-22-07-28T15-49-45.yaml',
    # SparseSync: only pt on LRS3
    '22-07-13T22-25-49.pt':
    f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/22-07-13T22-25-49.pt',
    'cfg-22-07-13T22-25-49.yaml':
    f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/cfg-22-07-13T22-25-49.yaml',
    # SparseSync: feature extractors
    'ResNetAudio-22-08-04T09-51-04.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-08-04T09-51-04.pt',  # 2s
    'ResNetAudio-22-08-03T23-14-49.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-49.pt',  # 3s
    'ResNetAudio-22-08-03T23-14-28.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-28.pt',  # 4s
    'ResNetAudio-22-06-24T08-10-33.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T08-10-33.pt',  # 5s
    'ResNetAudio-22-06-24T17-31-07.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T17-31-07.pt',  # 6s
    'ResNetAudio-22-06-24T23-57-11.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T23-57-11.pt',  # 7s
    'ResNetAudio-22-06-25T04-35-42.pt':
    f'{PARENT_LINK}/sync/ResNetAudio-22-06-25T04-35-42.pt',  # 8s
}


def check_if_file_exists_else_download(path, fname2link=FNAME2LINK, chunk_size=1024):
    '''Checks if file exists, if not downloads it from the link to the path'''
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        link = fname2link.get(path.name, None)
        if link is None:
            raise ValueError(f'Cant find the checkpoint file: {path}.',
                             f'Please download it manually and ensure the path exists.')
        with requests.get(fname2link[path.name], stream=True) as r:
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                with open(path, 'wb') as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)


def get_md5sum(path):
    hash_md5 = md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096 * 8), b''):
            hash_md5.update(chunk)
    md5sum = hash_md5.hexdigest()
    return md5sum
