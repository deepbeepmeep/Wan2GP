H3_PHASE_2_NOISE_LEVEL_START_DEFAULT = 0.9035

# Experimental developer switch. Group masked H3 target rows by their 2x2 latent
# patch state so each contiguous group can reuse one AdaLN timestep row.
H3_GROUPED_MASKED_DENOISING = True

# Select an H3 patch cell when any source-mask pixel covered by that cell is
# editable. Disable to use majority coverage for each 16x16 latent position.
H3_GROUPED_MASK_ANY_PIXEL_CELL = True

# Expand the snapped editable mask by one complete H3 patch cell (32x32 pixels
# with the current 16x VAE scale and 2x2 latent patch).
H3_GROUPED_MASK_CELL_DILATION = False
