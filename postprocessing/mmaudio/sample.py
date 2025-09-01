import json
import logging
import os
import random

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from .data.data_setup import setup_test_datasets
from .runner import Runner
from .utils.dist_utils import info_if_rank_zero
from .utils.logger import TensorboardLogger

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])


def sample(cfg: DictConfig):
    # initial setup
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=cfg.enable_email and not cfg.debug)

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # cuda setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Configuration: {cfg}')
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')

    # construct the trainer
    runner = Runner(cfg, log=log, run_path=run_dir, for_training=False).enter_val()

    # load the last weights if needed
    if cfg['weights'] is not None:
        info_if_rank_zero(log, f'Loading weights from the disk: {cfg["weights"]}')
        runner.load_weights(cfg['weights'])
        cfg['weights'] = None
    else:
        weights = runner.get_final_ema_weight_path()
        if weights is not None:
            info_if_rank_zero(log, f'Automatically finding weight: {weights}')
            runner.load_weights(weights)

    # setup datasets
    dataset, sampler, loader = setup_test_datasets(cfg)
    data_cfg = cfg.data.ExtractedVGG_test
    with open_dict(data_cfg):
        if cfg.output_name is not None:
            # append to the tag
            data_cfg.tag = f'{data_cfg.tag}-{cfg.output_name}'

    # loop
    audio_path = None
    for curr_iter, data in enumerate(tqdm(loader)):
        new_audio_path = runner.inference_pass(data, curr_iter, data_cfg)
        if audio_path is None:
            audio_path = new_audio_path
        else:
            assert audio_path == new_audio_path, 'Different audio path detected'

    info_if_rank_zero(log, f'Inference completed. Audio path: {audio_path}')
    output_metrics = runner.eval(audio_path, curr_iter, data_cfg)

    if local_rank == 0:
        # write the output metrics to run_dir
        output_metrics_path = os.path.join(run_dir, f'{data_cfg.tag}-output_metrics.json')
        with open(output_metrics_path, 'w') as f:
            json.dump(output_metrics, f, indent=4)
