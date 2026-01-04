import logging
from omegaconf import DictConfig

log = logging.getLogger()


def get_dataset_cfg(cfg: DictConfig):
    dataset_name = cfg.dataset
    data_cfg = cfg.datasets[dataset_name]

    potential_overrides = [
        'image_directory',
        'mask_directory',
        'json_directory',
        'size',
        'save_all',
        'use_all_masks',
        'use_long_term',
        'mem_every',
    ]

    for override in potential_overrides:
        if cfg[override] is not None:
            log.info(f'Overriding config {override} from {data_cfg[override]} to {cfg[override]}')
            data_cfg[override] = cfg[override]
        # escalte all potential overrides to the top-level config
        if override in data_cfg:
            cfg[override] = data_cfg[override]

    return data_cfg
