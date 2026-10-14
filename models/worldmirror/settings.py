"""WorldMirror 2.0 output modes and tuning settings, shared by the handler (UI declarations) and the pipeline (defaults)."""

MODEL_MODES = {"choices": [("Gaussian Splats", "gaussians"), ("Point Cloud", "points")], "default": "gaussians", "label": "Output"}

RESOLUTIONS = [("518 px (Fastest, Lightest)", 518), ("644 px", 644), ("756 px", 756), ("840 px", 840), ("952 px (Upstream Default, Finest)", 952)]

CUSTOM_SETTINGS = [
    {"id": "max_views", "name": "Max Views", "label": "Max Views (Images + Video Frames)", "type": "int", "min": 1, "max": 64, "inc": 1, "default": 24,
     "info": "Most views reconstructed together: extra images are evenly subsampled, the video fills the remaining views. VRAM grows with it."},
    {"id": "max_resolution", "name": "Resolution", "label": "Processing Resolution (Long Side)", "type": "dropdown", "default": 952, "choices": RESOLUTIONS,
     "info": "Long side of the views given to the model (never above the first view's own size). Higher keeps finer detail and more points but uses more VRAM."},
    {"id": "filtering", "name": "Filtering", "label": "Filtering", "type": "dropdown", "default": "sky_edges",
     "choices": [("Remove Sky and Depth Edges (Recommended)", "sky_edges"), ("Remove Depth Edges Only", "edges"), ("Keep Everything", "none")],
     "info": "Drops the sky and the floating points that stretch between foreground and background at depth discontinuities."},
]

DEFAULT_CUSTOM_SETTINGS = {one["id"]: one["default"] for one in CUSTOM_SETTINGS}
