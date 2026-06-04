import typing as tp

from models.TTS.stable_audio3.models.diffusion import DiTWrapper, ConditionedDiffusionModelWrapper
from models.TTS.stable_audio3.models.autoencoders import (
    AudioAutoencoder,
    SAMEEncoder,
    SAMEDecoder,
)
from models.TTS.stable_audio3.models.conditioners import (
    MultiConditioner,
    NumberConditioner,
    T5GemmaConditioner,
)
from models.TTS.stable_audio3.models.bottleneck import SoftNormBottleneck
from models.TTS.stable_audio3.models.pretransforms import (
    PatchedPretransform,
    AutoencoderPretransform,
)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any], text_encoder_weights_path=None, text_encoder_tokenizer_dir=None, dtype=None):
    model_config = config["model"]
    diffusion_config = model_config.get("diffusion", None)
    diffusion_model_config = diffusion_config.get("config", None)
    diffusion_objective = diffusion_config.get("diffusion_objective", "v")
    modular_local_cond_configs = diffusion_config.get("modular_local_cond_configs", [])

    diffusion_model = DiTWrapper(
        diffusion_objective=diffusion_objective,
        modular_local_cond_configs=modular_local_cond_configs,
        **diffusion_model_config,
    )

    io_channels = model_config.get("io_channels", None)
    sample_rate = config.get("sample_rate", None)
    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    local_add_cond_ids = diffusion_config.get("local_add_cond_ids", [])
    modular_local_cond_ids = [c["id"] for c in modular_local_cond_configs]
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    distribution_shift_options = diffusion_config.get(
        "distribution_shift_options", None
    )
    sampling_distribution_shift_options = diffusion_config.get(
        "sampling_distribution_shift_options", None
    )
    mask_padding_attention = diffusion_config.get("mask_padding_attention", False)
    use_effective_length_for_schedule = diffusion_config.get(
        "use_effective_length_for_schedule", False
    )

    pretransform = create_pretransform_from_config(model_config, sample_rate)
    min_input_length = pretransform.downsampling_ratio

    conditioning_config = model_config.get("conditioning", None)

    conditioner = create_multi_conditioner_from_conditioning_config(
        conditioning_config,
        text_encoder_weights_path=text_encoder_weights_path,
        text_encoder_tokenizer_dir=text_encoder_tokenizer_dir,
        dtype=dtype,
    )

    min_input_length *= diffusion_model.model.patch_size

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        local_add_cond_ids=local_add_cond_ids,
        modular_local_cond_ids=modular_local_cond_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        distribution_shift_options=distribution_shift_options,
        sampling_distribution_shift_options=sampling_distribution_shift_options,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        diffusion_objective=diffusion_objective,
    )


def create_autoencoder_from_config(config, sample_rate):
    # AE-only configs (SAME-S/SAME-L) have encoder/decoder at the top of config["model"].
    # Full SA3 configs nest them inside config["model"]["pretransform"]["config"].
    if "encoder" in config:
        autoencoder_config = config
    else:
        autoencoder_config = config["pretransform"]["config"]
    encoder = SAMEEncoder(**autoencoder_config["encoder"]["config"])
    decoder = SAMEDecoder(**autoencoder_config["decoder"]["config"])

    latent_dim = autoencoder_config.get("latent_dim", None)
    downsampling_ratio = autoencoder_config.get("downsampling_ratio", None)
    io_channels = autoencoder_config.get("io_channels", None)
    in_channels = autoencoder_config.get("in_channels", None)
    out_channels = autoencoder_config.get("out_channels", None)
    soft_clip = autoencoder_config["decoder"].get("soft_clip", False)
    pretransform = PatchedPretransform(**autoencoder_config["pretransform"]["config"])
    bottleneck = SoftNormBottleneck(**autoencoder_config["bottleneck"]["config"])

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip,
    )


def create_pretransform_from_config(config, sample_rate):
    pretransform_block = config["pretransform"]
    autoencoder = create_autoencoder_from_config(config, sample_rate)
    chunked = pretransform_block.get("chunked", False)
    iterate_batch = pretransform_block.get("iterate_batch", False)
    return AutoencoderPretransform(
        autoencoder, chunked=chunked, iterate_batch=iterate_batch
    )


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any],
    text_encoder_weights_path=None,
    text_encoder_tokenizer_dir=None,
    dtype=None,
) -> MultiConditioner:
    """Create a MultiConditioner from a conditioning config dictionary."""
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5gemma":
            if text_encoder_weights_path is not None:
                conditioner_config["weights_path"] = text_encoder_weights_path
            if text_encoder_tokenizer_dir is not None:
                conditioner_config["tokenizer_path"] = text_encoder_tokenizer_dir
            if dtype is not None:
                conditioner_config["dtype"] = dtype
            conditioners[id] = T5GemmaConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(
        conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys
    )

