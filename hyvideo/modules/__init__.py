from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG


def load_model(model, i2v_condition_type, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            i2v_condition_type = i2v_condition_type,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[model],
            **factor_kwargs,
        )
        return model
    else:
        raise NotImplementedError()
