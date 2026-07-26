import json

def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    elif model_type == 'diffusion_uncond':
        from .diffusion import create_diffusion_uncond_from_config
        return create_diffusion_uncond_from_config(model_config)
    # elif model_type == 'diffusion_infill':
    #     from .diffusion import create_diffusion_infill_from_config
    #     return create_diffusion_infill_from_config(model_config)
    elif model_type == 'diffusion_cond' or model_type == 'diffusion_cond_inpaint' or model_type == "diffusion_prior" or model_type == "diffusion_infill" or model_type == "mm_diffusion_cond":
        from .diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    elif model_type == 'diffusion_autoencoder':
        from .autoencoders import create_diffAE_from_config
        return create_diffAE_from_config(model_config)
    elif model_type == 'lm':
        from .lm import create_audio_lm_from_config
        return create_audio_lm_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_model_from_config_path(model_config_path):
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    return create_model_from_config(model_config)

def create_pretransform_from_config(pretransform_config, sample_rate):
    pretransform_type = pretransform_config.get('type', None)

    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        from .pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)
        chunked = pretransform_config.get("chunked", False)

        pretransform = AutoencoderPretransform(autoencoder, scale=scale, model_half=model_half, iterate_batch=iterate_batch, chunked=chunked)
    elif pretransform_type == 'wavelet':
        from .pretransforms import WaveletPretransform

        wavelet_config = pretransform_config["config"]
        channels = wavelet_config["channels"]
        levels = wavelet_config["levels"]
        wavelet = wavelet_config["wavelet"]

        pretransform = WaveletPretransform(channels, levels, wavelet)
    elif pretransform_type == 'pqmf':
        from .pretransforms import PQMFPretransform
        pqmf_config = pretransform_config["config"]
        pretransform = PQMFPretransform(**pqmf_config)
    elif pretransform_type == 'dac_pretrained':
        from .pretransforms import PretrainedDACPretransform
        pretrained_dac_config = pretransform_config["config"]
        pretransform = PretrainedDACPretransform(**pretrained_dac_config)
    elif pretransform_type == "audiocraft_pretrained":
        from .pretransforms import AudiocraftCompressionPretransform

        audiocraft_config = pretransform_config["config"]
        pretransform = AudiocraftCompressionPretransform(**audiocraft_config)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')
    
    enable_grad = pretransform_config.get('enable_grad', False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'tanh':
        from .bottleneck import TanhBottleneck
        bottleneck = TanhBottleneck()
    elif bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'rvq':
        from .bottleneck import RVQBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        bottleneck = RVQBottleneck(**quantizer_params)
    elif bottleneck_type == "dac_rvq":
        from .bottleneck import DACRVQBottleneck

        bottleneck = DACRVQBottleneck(**bottleneck_config["config"])
    
    elif bottleneck_type == 'rvq_vae':
        from .bottleneck import RVQVAEBottleneck

        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }

        quantizer_params.update(bottleneck_config["config"])

        bottleneck = RVQVAEBottleneck(**quantizer_params)
        
    elif bottleneck_type == 'dac_rvq_vae':
        from .bottleneck import DACRVQVAEBottleneck
        bottleneck = DACRVQVAEBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'l2_norm':
        from .bottleneck import L2Bottleneck
        bottleneck = L2Bottleneck()
    elif bottleneck_type == "wasserstein":
        from .bottleneck import WassersteinBottleneck
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    elif bottleneck_type == "fsq":
        from .bottleneck import FSQBottleneck
        bottleneck = FSQBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')
    
    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck
