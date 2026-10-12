"""TRELLIS.2 workflow choices and tuning settings, shared by the handler (UI declarations) and the pipeline (defaults)."""

MODEL_MODES = {
    "choices": [("512 (Fastest, Coarse Detail)", 0), ("1024 (Direct)", 1), ("1024 Cascade (Recommended)", 2), ("1536 Cascade (Finest, Slowest)", 3)],
    "default": 2,
    "label": "Pipeline",
}

CUSTOM_SETTINGS = [
    {"id": "texture_mode", "label": "Surface Output", "name": "Surface", "type": "dropdown", "default": "texture",
     "choices": [("PBR Textures (UV Baked)", "texture"), ("Vertex Colors", "vertex"), ("Shape Only (No Texture)", "none")]},
    {"id": "max_faces", "label": "Max Faces (Decimation Target)", "name": "Max Faces", "type": "int", "default": 100000, "min": 20000, "max": 2000000, "inc": 10000},
    {"id": "texture_size", "label": "Texture Size", "name": "Texture Size", "type": "dropdown", "default": 2048,
     "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)]},
    {"id": "max_tokens", "label": "Max Tokens of the 1536 Cascade (lowers its resolution by steps of 128 above it)", "name": "Max Tokens",
     "type": "int", "default": 49152, "min": 16384, "max": 98304, "inc": 4096},
    {"id": "alpha_mode", "label": "Transparency", "name": "Transparency", "type": "dropdown", "default": "opaque",
     "choices": [("Opaque (Alpha Ignored, as the Official Exporter)", "opaque"), ("Alpha Blending (Glass, Thin Materials)", "blend")]},
]

DEFAULT_CUSTOM_SETTINGS = {one["id"]: one["default"] for one in CUSTOM_SETTINGS}
