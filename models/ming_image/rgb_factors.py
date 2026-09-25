# Latent-to-RGB preview fit on 1,000 E:/ML/images JPEGs (800 train, 200 validation).
# Fit against the released Ming VAE mode latents after scaling by 8.0064.
# Reproduce with: python -m models.ming_image.regress_preview --images E:/ML/images \
#   --checkpoints D:/ml/wangp/ckpts --output _temp_codex/ming_opt/rgb_preview_1000.json
# See specs/MING_IMAGE_OPTIMIZATION.md for validation metrics.
RGB_FACTORS = [
    [0.02716483, 0.08257899, 0.19002339],
    [-0.04156913, -0.0416391, -0.02433927],
    [0.11596507, 0.0778253, 0.03903173],
    [0.14605501, 0.1380829, 0.07980736],
    [0.00337972, -0.01581605, -0.03437042],
    [0.08873618, 0.0104203, -0.01470665],
    [-0.12293938, -0.12913099, -0.07804169],
    [-0.04046585, 0.01898169, 0.02781412],
    [-0.13812313, -0.14329094, -0.17977682],
    [-0.06185946, 0.01024195, 0.03351302],
    [0.01300573, 0.06071904, 0.01739311],
    [0.0140004, 0.05279799, 0.08491608],
    [-0.10177479, 0.00714422, 0.04575863],
    [-0.03413372, -0.06290488, -0.02718202],
    [0.09658136, 0.05194259, 0.08794784],
    [0.10597223, 0.0767343, 0.08300512],
]
RGB_BIAS = [-0.08448407, -0.111338, -0.16915486]
