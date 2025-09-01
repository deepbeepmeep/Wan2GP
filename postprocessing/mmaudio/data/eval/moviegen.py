import json
import logging
import os
from pathlib import Path
from typing import Union

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

from ...utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class MovieGenData(Dataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        sync_root: Union[str, Path],
        jsonl_root: Union[str, Path],
        *,
        duration_sec: float = 10.0,
        read_clip: bool = True,
    ):
        self.video_root = Path(video_root)
        self.sync_root = Path(sync_root)
        self.jsonl_root = Path(jsonl_root)
        self.read_clip = read_clip

        videos = sorted(os.listdir(self.video_root))
        videos = [v[:-4] for v in videos]  # remove extensions
        self.captions = {}

        for v in videos:
            with open(self.jsonl_root / (v + '.jsonl')) as f:
                data = json.load(f)
                self.captions[v] = data['audio_prompt']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')

        self.duration_sec = duration_sec

        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_augment = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.sync_augment = v2.Compose([
            v2.Resize((_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.videos = videos

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]

        reader = StreamingMediaDecoder(self.video_root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(f'CLIP video too short {video_id}')

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(f'Sync video too short {video_id}')

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(f'CLIP video wrong length {video_id}, '
                               f'expected {self.clip_expected_length}, '
                               f'got {clip_chunk.shape[0]}')
        clip_chunk = self.clip_augment(clip_chunk)

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_id}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_augment(sync_chunk)

        data = {
            'name': video_id,
            'caption': caption,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.sample(idx)

    def __len__(self):
        return len(self.captions)
