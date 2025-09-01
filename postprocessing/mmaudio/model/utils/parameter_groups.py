import logging

log = logging.getLogger()


def get_parameter_groups(model, cfg, print_log=False):
    """
    Assign different weight decays and learning rates to different parameters.
    Returns a parameter group which can be passed to the optimizer.
    """
    weight_decay = cfg.weight_decay
    # embed_weight_decay = cfg.embed_weight_decay
    # backbone_lr_ratio = cfg.backbone_lr_ratio
    base_lr = cfg.learning_rate

    backbone_params = []
    embed_params = []
    other_params = []

    # embedding_names = ['summary_pos', 'query_init', 'query_emb', 'obj_pe']
    # embedding_names = [e + '.weight' for e in embedding_names]

    # inspired by detectron2
    memo = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid duplicating parameters
        if param in memo:
            continue
        memo.add(param)

        if name.startswith('module'):
            name = name[7:]

        inserted = False
        # if name.startswith('pixel_encoder.'):
        #     backbone_params.append(param)
        #     inserted = True
        #     if print_log:
        #         log.info(f'{name} counted as a backbone parameter.')
        # else:
        #     for e in embedding_names:
        #         if name.endswith(e):
        #             embed_params.append(param)
        #             inserted = True
        #             if print_log:
        #                 log.info(f'{name} counted as an embedding parameter.')
        #             break

        # if not inserted:
        other_params.append(param)

    parameter_groups = [
        # {
        #     'params': backbone_params,
        #     'lr': base_lr * backbone_lr_ratio,
        #     'weight_decay': weight_decay
        # },
        # {
        #     'params': embed_params,
        #     'lr': base_lr,
        #     'weight_decay': embed_weight_decay
        # },
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        },
    ]

    return parameter_groups
