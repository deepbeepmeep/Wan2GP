"""TripoSplat tuning settings, shared by the handler (UI declarations) and the pipeline (defaults)."""

CUSTOM_SETTINGS = [
    {"id": "num_gaussians", "label": "Gaussian Count", "name": "Gaussians", "type": "dropdown", "default": 262144,
     "choices": [("32K (Lightest File)", 32768), ("64K", 65536), ("128K", 131072), ("192K", 196608), ("256K (Most Detail)", 262144)],
     "info": "Number of 3D Gaussians: more Gaussians keep finer details, fewer give lighter files that display faster. The model runs once; "
             "only the final decoding depends on it."},
]

DEFAULT_CUSTOM_SETTINGS = {one["id"]: one["default"] for one in CUSTOM_SETTINGS}
