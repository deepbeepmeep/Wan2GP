"""
Dumps things to tensorboard and console
"""

import datetime
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from PIL import Image
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

from .email_utils import EmailSender
from .time_estimator import PartialTimeEstimator, TimeEstimator
from .timezone import my_timezone


def tensor_to_numpy(image: torch.Tensor):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np


def detach_to_cpu(x: torch.Tensor):
    return x.detach().cpu()


def fix_width_trunc(x: float):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))


def plot_spectrogram(spectrogram: np.ndarray, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(spectrogram, origin="lower", aspect="auto", interpolation="nearest")


class TensorboardLogger:

    def __init__(self,
                 exp_id: str,
                 run_dir: Union[Path, str],
                 py_logger: logging.Logger,
                 *,
                 is_rank0: bool = False,
                 enable_email: bool = False):
        self.exp_id = exp_id
        self.run_dir = Path(run_dir)
        self.py_log = py_logger
        self.email_sender = EmailSender(exp_id, enable=(is_rank0 and enable_email))
        if is_rank0:
            self.tb_log = SummaryWriter(run_dir)
        else:
            self.tb_log = None

        # Get current git info for logging
        try:
            import git
            repo = git.Repo(".")
            git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
        except (ImportError, RuntimeError, TypeError):
            print('Failed to fetch git info. Defaulting to None')
            git_info = 'None'

        self.log_string('git', git_info)

        # log the SLURM job id if available
        job_id = os.environ.get('SLURM_JOB_ID', None)
        if job_id is not None:
            self.log_string('slurm_job_id', job_id)
            self.email_sender.send(f'Job {job_id} started', f'Job started {run_dir}')

        # used when logging metrics
        self.batch_timer: TimeEstimator = None
        self.data_timer: PartialTimeEstimator = None

        self.nan_count = defaultdict(int)

    def log_scalar(self, tag: str, x: float, it: int):
        if self.tb_log is None:
            return
        if math.isnan(x) and 'grad_norm' not in tag:
            self.nan_count[tag] += 1
            if self.nan_count[tag] == 10:
                self.email_sender.send(
                    f'Nan detected in {tag} @ {self.run_dir}',
                    f'Nan detected in {tag} at iteration {it}; run_dir: {self.run_dir}')
        else:
            self.nan_count[tag] = 0
        self.tb_log.add_scalar(tag, x, it)

    def log_metrics(self,
                    prefix: str,
                    metrics: dict[str, float],
                    it: int,
                    ignore_timer: bool = False):
        msg = f'{self.exp_id}-{prefix} - it {it:6d}: '
        metrics_msg = ''
        for k, v in sorted(metrics.items()):
            self.log_scalar(f'{prefix}/{k}', v, it)
            metrics_msg += f'{k: >10}:{v:.7f},\t'

        if self.batch_timer is not None and not ignore_timer:
            self.batch_timer.update()
            avg_time = self.batch_timer.get_and_reset_avg_time()
            data_time = self.data_timer.get_and_reset_avg_time()

            # add time to tensorboard
            self.log_scalar(f'{prefix}/avg_time', avg_time, it)
            self.log_scalar(f'{prefix}/data_time', data_time, it)

            est = self.batch_timer.get_est_remaining(it)
            est = datetime.timedelta(seconds=est)
            if est.days > 0:
                remaining_str = f'{est.days}d {est.seconds // 3600}h'
            else:
                remaining_str = f'{est.seconds // 3600}h {(est.seconds%3600) // 60}m'
            eta = datetime.datetime.now(timezone(my_timezone)) + est
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S %Z%z')
            time_msg = f'avg_time:{avg_time:.3f},data:{data_time:.3f},remaining:{remaining_str},eta:{eta_str},\t'
            msg = f'{msg} {time_msg}'

        msg = f'{msg} {metrics_msg}'
        self.py_log.info(msg)

    def log_histogram(self, tag: str, hist: torch.Tensor, it: int):
        if self.tb_log is None:
            return
        # hist should be a 1D tensor
        hist = hist.cpu().numpy()
        fig, ax = plt.subplots()
        x_range = np.linspace(0, 1, len(hist))
        ax.bar(x_range, hist, width=1 / (len(hist) - 1))
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
        plt.tight_layout()
        self.tb_log.add_figure(tag, fig, it)
        plt.close()

    def log_image(self, prefix: str, tag: str, image: np.ndarray, it: int):
        image_dir = self.run_dir / f'{prefix}_images'
        image_dir.mkdir(exist_ok=True, parents=True)

        image = Image.fromarray(image)
        image.save(image_dir / f'{it:09d}_{tag}.png')

    def log_audio(self,
                  prefix: str,
                  tag: str,
                  waveform: torch.Tensor,
                  it: Optional[int] = None,
                  *,
                  subdir: Optional[Path] = None,
                  sample_rate: int = 16000) -> Path:
        if subdir is None:
            audio_dir = self.run_dir / prefix
        else:
            audio_dir = self.run_dir / subdir / prefix
        audio_dir.mkdir(exist_ok=True, parents=True)

        if it is None:
            name = f'{tag}.flac'
        else:
            name = f'{it:09d}_{tag}.flac'

        torchaudio.save(audio_dir / name,
                        waveform.cpu().float(),
                        sample_rate=sample_rate,
                        channels_first=True)
        return Path(audio_dir)

    def log_spectrogram(
        self,
        prefix: str,
        tag: str,
        spec: torch.Tensor,
        it: Optional[int],
        *,
        subdir: Optional[Path] = None,
    ):
        if subdir is None:
            spec_dir = self.run_dir / prefix
        else:
            spec_dir = self.run_dir / subdir / prefix
        spec_dir.mkdir(exist_ok=True, parents=True)

        if it is None:
            name = f'{tag}.png'
        else:
            name = f'{it:09d}_{tag}.png'

        plot_spectrogram(spec.cpu().float())
        plt.tight_layout()
        plt.savefig(spec_dir / name)
        plt.close()

    def log_string(self, tag: str, x: str):
        self.py_log.info(f'{tag} - {x}')
        if self.tb_log is None:
            return
        self.tb_log.add_text(tag, x)

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)

        self.email_sender.send(f'Error occurred in {self.run_dir}', x)

    def complete(self):
        self.email_sender.send(f'Job completed in {self.run_dir}', 'Job completed')
