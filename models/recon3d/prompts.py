MOGE2_INFOS = """
# MoGe-2

MoGe-2 (Microsoft, MIT, ViT-L with normals, 331M parameters) reconstructs the visible 3D geometry of **one photo**: a metric point for
every pixel (in metres), surface normals and the camera field of view. It needs no prompt and runs in a few seconds with little VRAM.

## Inputs

- **Photo** (required): any picture, indoor or outdoor scene, object or person. Only what the camera sees is reconstructed (a 2.5D
  relief, not the back of objects). Sky and other undefined areas are left out.
- **Asset Name / Description**: names the files and is stored with the settings; it does not change the result.

## Output

- **Point Cloud (Gaussian Splats)**: the points as small round splats, viewable in the 3D gallery. The plain point cloud (`_points.ply`,
  x y z + RGB) and the camera (`_cameras.json`) are saved beside it.
- **Mesh (Textured GLB)**: the pixel grid triangulated into a surface textured with the photo, for 3D editors and game engines.

Axes follow glTF: +Y up, the scene in front of the camera at the origin (looking down -Z), units in metres.

## Settings

- **Working Resolution**: larger photos are reduced to this longest side; points and mesh vertices come from this resolution
  (1024 gives about 0.8M points).
- **Inference Detail**: number of transformer tokens; High is the upstream default, Ultra sharpens fine structures at a higher cost.
- **Depth Discontinuities**: Removed drops the pixels on depth jumps, which otherwise appear as stretched streaks between a foreground
  object and the background.
- **Max Points**: point clouds above it keep a random subset (the mesh is not reduced).

**License:** MIT (models/recon3d/LICENSE_MoGe).
"""

MAPANYTHING_INFOS = """
# MapAnything

MapAnything (Meta, Apache-2.0 model trained on commercially usable data, 1.2B parameters) is a metric multi-view reconstruction model:
from **several photos of the same scene or the frames of a video**, it predicts in one pass the 3D points of every pixel in a common
world frame, the camera position and orientation of every view and their focal lengths, in metres. One photo alone also works
(monocular metric depth).

## Inputs

- **Photos**: two or more pictures of the same place or object taken from different positions, with overlap between them.
- **Reference Video**: a short clip moving around a scene or object; **Video Frames Used** frames are sampled evenly over it.
  Use photos, a video or both (the photos come first, the first one sets the world frame); choose "No Photos" to use the video alone.
- **Asset Name / Description**: names the files only.

Moving cameras with static scenes work best. Moving people or objects, reflections and textureless walls reduce the quality.

## Output

Gaussian splats of the merged points, plus `_points.ply` (plain x y z + RGB point cloud) and `_cameras.json` (every camera's pose and
intrinsics). Axes follow glTF (+Y up), the world frame is that of the first view, units in metres.

## Settings

- **Max Points**: above it a random subset is kept (16 views at 518 px give about 3M raw points).
- **Video Frames Used**: more frames cover more of the scene but take longer and more memory (attention over all views).
- **Low-Confidence Filter**: drops the least confident pixels of each view (blurry, far or ambiguous areas). Pixels on depth and normal
  edges and the ambiguous areas the model flags (sky, reflections) are always removed.

**License:** the Apache-2.0 release of MapAnything (models/recon3d/LICENSE_MapAnything).
"""

DA3_INFOS = """
# Depth Anything 3 Points

Depth Anything 3 (ByteDance, DA3-LARGE-1.1, 0.41B parameters, the checkpoint WanGP already uses for depth control) estimates the depth
of every pixel and the cameras of **one or several photos or video frames** in one pass, and back-projects them into one 3D point
cloud. It is fast and light; MapAnything gives metric scale and usually more accurate multi-view geometry.

## Inputs

- **Photos** and/or a **Reference Video** (frames sampled evenly over it, see Video Frames Used); choose "No Photos" to use the video
  alone. Views of the same static scene with overlap work best.
- **Asset Name / Description**: names the files only.

## Output

Gaussian splats of the points, plus `_points.ply` (plain x y z + RGB) and `_cameras.json` (poses and intrinsics). Axes follow glTF
(+Y up), the world frame is that of the first view. The scale is relative (not metric).

## Settings

- **Max Points**: above it a random subset is kept.
- **Video Frames Used**: frames sampled from the video.
- **Low-Confidence Filter**: the threshold of the official exporter: about this share of the least confident pixels is removed (sky and
  far background first).
- **Working Resolution**: longest side of the processed views; higher gives more and finer points, slower.

**License:** the DA3-LARGE-1.1 model card states Apache-2.0 while the Depth Anything 3 README lists it as CC BY-NC 4.0
(non-commercial): check its terms before commercial use.
"""

PROMPT_INFOS = """
# Asset Name

These reconstruction models use only the photos and video: the text is not a prompt. Write a short name such as `Living room scan` or
`Garden statue`; it becomes the output file name and is stored with the settings.
"""

MOGE2_DEEPY_INFOS = """Photo-to-3D geometry: one `image_refs` photo -> metric point cloud as Gaussian splats (`model_mode` "points", default) or a textured GLB mesh of the visible surface ("mesh"); side files `_points.ply` and `_cameras.json`. Only the visible side is reconstructed (2.5D). The prompt is only the asset name. Custom settings: `max_size` 512-2048 (1024), `detail` 0/5/9/30 (9), `edges` remove/keep, `max_points` (1000000)."""

MAPANYTHING_DEEPY_INFOS = """Multi-view metric reconstruction: several `image_refs` photos of one static scene and/or a `video_guide` video (frames sampled evenly; `video_prompt_type` "I" photos, "V" video, "IV" both) -> merged point cloud as Gaussian splats in metres plus every camera (`_cameras.json`) and `_points.ply`. One photo also works. The prompt is only the asset name. Custom settings: `max_points` (1000000), `max_frames` (16), `confidence` 0/10/25/50 percent filter (0)."""

DA3_DEEPY_INFOS = """Fast depth-based reconstruction (Depth Anything 3 LARGE-1.1, licence ambiguous: Apache-2.0 card vs CC BY-NC README): `image_refs` photos and/or a `video_guide` video (`video_prompt_type` "I" photos, "V" video, "IV" both) -> point cloud as Gaussian splats with cameras (`_cameras.json`) and `_points.ply`; relative scale. The prompt is only the asset name. Custom settings: `max_points` (1000000), `max_frames` (16), `confidence` 0/10/25/40/60 (40), `process_res` 504/756/1008 (504)."""

DEEPY_PROMPT_INFOS = """The prompt is optional: a short asset name such as `Living room scan`: it names the output files and does not change the reconstruction, which depends only on the photos and video."""
