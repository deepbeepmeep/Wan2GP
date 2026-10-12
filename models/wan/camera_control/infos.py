CAMERA_PRESET_CHOICES = [
    ("Dolly Zoom", "dolly_zoom"),
    ("Left Front Zoom", "left_front_zoom"),
    ("Right Front Zoom", "right_front_zoom"),
    ("Close Crane Above", "close_crane_above"),
    ("Close Crane Below", "close_crane_below"),
    ("Arc Right 45 deg", "arc_right_45"),
    ("Arc Left 45 deg", "arc_left_45"),
    ("Push In", "push_in"),
    ("Pull Back", "pull_back"),
    ("Truck Right", "truck_right"),
    ("Truck Left", "truck_left"),
    ("Pedestal Up", "pedestal_up"),
    ("Pedestal Down", "pedestal_down"),
    ("Pan Right 45 deg", "pan_right_45"),
    ("Pan Left 45 deg", "pan_left_45"),
    ("Tilt Up 45 deg", "tilt_up_45"),
    ("Tilt Down 45 deg", "tilt_down_45"),
    ("Zoom In", "zoom_in"),
    ("Zoom Out", "zoom_out"),
    ("Bird View", "bird_view"),
    ("Crane Above Right", "crane_above_right"),
    ("Crane Above Left", "crane_above_left"),
    ("Crane Below Right", "crane_below_right"),
    ("Crane Below Left", "crane_below_left"),
]

CAMERA_STRENGTH_SETTING = {"id": "camera_strength", "name": "Camera strength", "label": "Camera Movement Strength (%)", "type": "float", "default": 100.0, "min": 0.0, "max": 200.0, "inc": 5.0, "info": "Scales the preset camera move; ignored with a custom camera path."}
CAMERA_CONTROL_SPECIALITY = {"name": "camera control", "aliases": ["camera path", "camera trajectory", "camera movement", "orbit", "dolly", "3D camera"], "description": "The camera follows a chosen 3D path (preset moves or a custom camera path file) while the video animates the start image."}
CAMERA_PATH_GUIDE = {"id": "custom_guide", "name": "Custom Camera Path", "label": "Custom Camera Path (.npz cam_c2w + intrinsics, or CameraCtrl .txt poses)", "type": "file", "default": None, "required": False, "file_types": [".npz", ".txt"]}

_CAMERA_INPUTS = """- **Camera Movement**: a preset move. Push In / Pull Back move the camera forward or backward, Truck moves it sideways, Pedestal up or
  down, Pan and Tilt turn it on the spot, Arc orbits around the middle of the scene, Crane and Bird View combine rising and turning,
  Zoom In / Zoom Out change the focal length and Dolly Zoom moves forward while widening the lens.
- **Camera Movement Strength (%)**: scales the preset, 100 is the default amplitude, 0 keeps the camera still and 200 doubles the move.
- **Custom Camera Path**: replaces the preset with your own path, one pose per frame (at least as many poses as frames):
  - a `.npz` file with `cam_c2w` (camera-to-world 4x4 matrices, OpenCV axes: x right, y down, z forward) and `intrinsics`
    (fx, fy, cx, cy in pixels), as written by Vista4D;
  - a CameraCtrl / RealEstate10K `.txt` pose file (VideoX-Fun's format, the first line is skipped).
  The path is used relative to its first camera."""

FUN_CAMERA_INFOS = f"""
# Wan2.1 Fun Control-Camera 1.3B

Animates a start image (or a scene described by the prompt) while the camera follows a chosen 3D path. The path is turned into
camera rays for every pixel and frame (Plucker embeddings), so the camera motion is precise and geometrically consistent instead of
being described in words. This light 1.3B model fits small GPUs.

## Inputs

- **Start Image** (recommended): becomes the first frame. Without it the whole scene comes from the prompt.
{_CAMERA_INPUTS}

## Recommended Settings

832x480 (or another resolution around 480p), 81 frames at 16 fps, 50 steps, guidance 6, shift 3. The whole clip follows one path:
sliding windows are not available.

## Limitations

The model was trained on camera translations and rotations with a fixed lens: physical moves (push in, truck, pedestal, pan, tilt,
arc) work best, focal-length changes (Zoom In, Zoom Out, Dolly Zoom) are followed weakly. As a 1.3B model its image quality is below
the 14B models, and strong moves can blur details that the start image does not show.

**License:** Apache 2.0 (Wan2.1-Fun-V1.1-1.3B-Control-Camera by Alibaba PAI).
"""

FUN_CAMERA_PROMPT_INFOS = """
# Fun Control-Camera Prompts

Describe the content of the scene and what moves in it: subjects, their actions, light and style. The camera path controls the
camera: you can mention the move ("the camera slowly orbits to the left") but never contradict the selected path.

## Example

`A quiet Italian river at noon: an old stone bridge lined with yellow houses, a rowing boat gliding on the green water, clouds drifting
slowly, realistic photography, high detail.`
"""

FUN_CAMERA_DEEPY_INFOS = """Wan2.1 Fun Control-Camera 1.3B: video along a 3D camera path (Plucker rays). `image_prompt_type`: `S` = `image_start` is the first frame (recommended), empty = text only. `model_mode` = camera preset (push_in, pull_back, truck_left/right, pedestal_up/down, pan_left_45/right_45, tilt_up_45/down_45, arc_left_45/right_45, crane_*, bird_view, zoom_in/out, dolly_zoom, left/right_front_zoom, close_crane_above/below); `custom_settings.camera_strength` scales it (100 default). `custom_guide` = optional custom path (.npz cam_c2w + intrinsics, or CameraCtrl .txt) replacing the preset. 832x480, 81 frames, 16 fps, 50 steps, guidance 6, shift 3; no sliding windows. Physical moves work best; focal zooms are weak."""

FUN_CAMERA_DEEPY_PROMPT_INFOS = """Describe the scene content, subject actions, light and style; the camera path drives the camera, so do not describe a different move. Example: `An old stone bridge over a green river, a rowing boat gliding, clouds drifting, realistic photography.`"""

UNI3C_INFOS = f"""
# Uni3C Camera Control 14B

Uni3C adds a ControlNet to the Wan2.1 Image2Video 14B 720p model to move the camera precisely around the scene of a start image.
WanGP estimates the depth of the start image (Depth Anything 3 Metric), lifts it to a 3D point cloud, renders that cloud along the
chosen camera path and gives the renders, the areas they do not cover and the camera rays to the ControlNet. The video keeps the
geometry of the scene while the model fills in what the start image does not show.

## Inputs

- **Start Image** (required): becomes the first frame and defines the 3D scene.
{_CAMERA_INPUTS}
  For a custom path, translations are in meters of the estimated scene (the start camera sits at the origin).

## Recommended Settings

832x480 or 768x480, 81 frames at 16 fps, 50 steps, guidance 5, shift 5. Larger resolutions work and are slower. The whole clip
follows one path: sliding windows are not available.

## Limitations

The geometry comes from one image: hidden areas, the sky and thin structures are invented and large moves (big orbits, long push
ins) reveal more of them. People and objects can move as the prompt says, but the rendered guide is static. The human-motion (SMPL)
control of the Uni3C paper is not supported.

**License:** Apache 2.0 (Uni3C by Alibaba DAMO Academy; Wan2.1 Image2Video 14B; Depth Anything 3 Metric by ByteDance).
"""

UNI3C_PROMPT_INFOS = """
# Uni3C Prompts

Describe the start image and what should happen in it: subjects, actions, light and style. The camera follows the selected path, so
keep camera words consistent with it or leave them out. Upstream's default prompt is a good neutral choice when nothing moves:
`This video describes a slow and stable camera movement with high quality and high definition.`

## Example

`An old stone bridge lined with yellow houses over a calm green river, a rowing team gliding under the arches, clouds drifting in a
blue sky, realistic photography, high detail.`
"""

UNI3C_DEEPY_INFOS = """Uni3C 14B (Wan2.1 I2V 720p + ControlNet): moves the camera around the scene of a required start image (`image_prompt_type` `S`, `image_start`). WanGP estimates depth, lifts a point cloud and renders it along the path to guide the video. `model_mode` = camera preset (push_in, pull_back, truck_left/right, pedestal_up/down, pan_left_45/right_45, tilt_up_45/down_45, arc_left_45/right_45, crane_*, bird_view, zoom_in/out, dolly_zoom, left/right_front_zoom, close_crane_above/below); `custom_settings.camera_strength` scales it (100 default). `custom_guide` = optional custom path (.npz cam_c2w + intrinsics in meters of the scene, or CameraCtrl .txt). 832x480, 81 frames, 16 fps, 50 steps, guidance 5, shift 5; no sliding windows. Hidden areas are invented; no SMPL human-motion control."""

UNI3C_DEEPY_PROMPT_INFOS = """Describe the start image and its motion (subjects, actions, light, style); keep camera words consistent with the path or omit them. Neutral prompt: `This video describes a slow and stable camera movement with high quality and high definition.`"""
