H3_PHASE_2_NOISE_LEVEL_START_DEFAULT = 0.9035

H3_MASK_MODE_SETTING = "h3_mask_mode"
H3_MASK_MODE_SHARED_TIMESTEP = "shared_timestep"
H3_MASK_MODE_GROUPED_ROWS = "grouped_rows"
H3_MASK_MODE_DEFAULT = H3_MASK_MODE_GROUPED_ROWS
H3_MASK_MODES = (H3_MASK_MODE_SHARED_TIMESTEP, H3_MASK_MODE_GROUPED_ROWS)


def h3_grouped_masking_enabled(custom_settings):
    mode = H3_MASK_MODE_DEFAULT if custom_settings is None else custom_settings.get(H3_MASK_MODE_SETTING, H3_MASK_MODE_DEFAULT)
    if mode not in H3_MASK_MODES:
        raise ValueError(f"Unsupported MiniMax H3 mask denoising mode {mode!r}")
    return mode == H3_MASK_MODE_GROUPED_ROWS
