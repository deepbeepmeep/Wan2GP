"""AnimateAnyMesh tuning settings, shared by the handler (UI declarations) and the pipeline (defaults)."""

CUSTOM_SETTINGS = [
    {"id": "fps", "name": "FPS", "label": "Playback Frame Rate (frames per second)", "type": "int", "default": 10, "min": 1, "max": 60, "inc": 1,
     "info": "Speed of the animation: the 16 generated frames last 1.5 s at 10 fps (the upstream rendering rate). It does not change the motion itself."},
    {"id": "playback", "name": "Playback", "label": "Playback", "type": "dropdown", "default": "forward",
     "choices": [("Forward (16 Frames)", "forward"), ("Forward then Backward (Seamless Loop, 31 Frames)", "pingpong")],
     "info": "Viewers loop the animation: Forward jumps back to the first frame, Forward then Backward plays the motion back in reverse so that the loop has no jump."},
    {"id": "trajectories", "name": "Trajectories", "label": "Motion Trajectories (0 = Auto: Vertices / 8, at least 512)", "type": "int", "default": 0, "min": 0, "max": 16384, "inc": 64,
     "info": "Number of vertices whose trajectories the model generates; the other vertices follow them. Fewer than the automatic count amplify the motion but can distort the shape."},
    {"id": "max_vertices", "name": "Max Vertices", "label": "Max Vertices (denser meshes are animated through a simplified copy)", "type": "int", "default": 50000, "min": 2000,
     "max": 200000, "inc": 1000,
     "info": "The model was trained on meshes of up to 50000 vertices. Above this count a simplified copy is animated and every original vertex follows its nearest simplified vertices."},
]

DEFAULT_CUSTOM_SETTINGS = {one["id"]: one["default"] for one in CUSTOM_SETTINGS}
