import bisect

import torch
from torch.utils.data.dataset import Dataset


# modified from https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
class MultiModalDataset(Dataset):
    datasets: list[Dataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, video_datasets: list[Dataset], audio_datasets: list[Dataset]):
        super().__init__()
        self.video_datasets = list(video_datasets)
        self.audio_datasets = list(audio_datasets)
        self.datasets = self.video_datasets + self.audio_datasets

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_datasets[0].compute_latent_stats()
