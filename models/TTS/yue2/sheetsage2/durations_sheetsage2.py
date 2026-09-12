import numpy as np


DURATION_TEMPLATES = np.array(
    [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
        96, 128, 192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096,
    ],
    dtype=np.int32,
)
duration_boundaries = (DURATION_TEMPLATES[:-1] + DURATION_TEMPLATES[1:]) / 2


