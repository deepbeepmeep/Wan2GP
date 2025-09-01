import logging
import os
from pathlib import Path
from typing import Union

import open_clip
import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

log = logging.getLogger()


class WavTextClipsDataset(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        captions_tsv: Union[str, Path],
        clips_tsv: Union[str, Path],
        sample_rate: int,
        num_samples: int,
        normalize_audio: bool = False,
        reject_silent: bool = False,
        tokenizer_id: str = 'ViT-H-14-378-quickgelu',
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.normalize_audio = normalize_audio
        self.reject_silent = reject_silent
        self.tokenizer = open_clip.get_tokenizer(tokenizer_id)

        audios = sorted(os.listdir(self.root))
        audios = set([
            Path(audio).stem for audio in audios
            if audio.endswith('.wav') or audio.endswith('.flac')
        ])
        self.captions = {}

        # read the caption tsv
        df_list = pd.read_csv(captions_tsv, sep='\t', dtype={'id': str}).to_dict('records')
        for record in df_list:
            id = record['id']
            caption = record['caption']
            self.captions[id] = caption

        # read the clip tsv
        df_list = pd.read_csv(clips_tsv, sep='\t', dtype={
            'id': str,
            'name': str
        }).to_dict('records')
        self.clips = []
        for record in df_list:
            record['id'] = record['id']
            record['name'] = record['name']
            id = record['id']
            name = record['name']
            if name not in self.captions:
                log.warning(f'Audio {name} not found in {captions_tsv}')
                continue
            record['caption'] = self.captions[name]
            self.clips.append(record)

        log.info(f'Found {len(self.clips)} audio files in {self.root}')

        self.resampler = {}

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            clip = self.clips[idx]
            audio_name = clip['name']
            audio_id = clip['id']
            caption = clip['caption']
            start_sample = clip['start_sample']
            end_sample = clip['end_sample']

            audio_path = self.root / f'{audio_name}.flac'
            if not audio_path.exists():
                audio_path = self.root / f'{audio_name}.wav'
                assert audio_path.exists()

            audio_chunk, sample_rate = torchaudio.load(audio_path)
            audio_chunk = audio_chunk.mean(dim=0)  # mono
            abs_max = audio_chunk.abs().max()
            if self.normalize_audio:
                audio_chunk = audio_chunk / abs_max * 0.95

            if self.reject_silent and abs_max < 1e-6:
                log.warning(f'Rejecting silent audio')
                return None

            audio_chunk = audio_chunk[start_sample:end_sample]

            # resample
            if sample_rate == self.sample_rate:
                audio_chunk = audio_chunk
            else:
                if sample_rate not in self.resampler:
                    # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                    self.resampler[sample_rate] = torchaudio.transforms.Resample(
                        sample_rate,
                        self.sample_rate,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method='sinc_interp_kaiser',
                        beta=14.769656459379492,
                    )
                audio_chunk = self.resampler[sample_rate](audio_chunk)

            if audio_chunk.shape[0] < self.num_samples:
                raise ValueError('Audio is too short')
            audio_chunk = audio_chunk[:self.num_samples]

            tokens = self.tokenizer([caption])[0]

            output = {
                'waveform': audio_chunk,
                'id': audio_id,
                'caption': caption,
                'tokens': tokens,
            }

            return output
        except Exception as e:
            log.error(f'Error reading {audio_path}: {e}')
            return None

    def __len__(self):
        return len(self.clips)
