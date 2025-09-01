import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

log = logging.getLogger()


class AudioCapsData(Dataset):

    def __init__(self, audio_path: Union[str, Path], csv_path: Union[str, Path]):
        df = pd.read_csv(csv_path).to_dict(orient='records')

        audio_files = sorted(os.listdir(audio_path))
        audio_files = set(
            [Path(f).stem for f in audio_files if f.endswith('.wav') or f.endswith('.flac')])

        self.data = []
        for row in df:
            self.data.append({
                'name': row['name'],
                'caption': row['caption'],
            })

        self.audio_path = Path(audio_path)
        self.csv_path = Path(csv_path)

        log.info(f'Found {len(self.data)} matching audio files in {self.audio_path}')

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __len__(self):
        return len(self.data)
