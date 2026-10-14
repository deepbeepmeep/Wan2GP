_LICENSE_NOTE = ("**License:** Tencent HY-WORLD 2.0 Community License (models/worldmirror/LICENSE): not licensed in the European Union, the "
                 "United Kingdom and South Korea; outputs may not be used to improve other AI models.")

WORLDMIRROR_INFOS = f"""
# WorldMirror 2.0

Tencent HY-World 2.0 WorldMirror (1.2B) reconstructs a scene or an object in 3D from several photos or a video in one pass: it estimates
the camera of every view, its depth and normals, and Gaussian splats. Reconstruction, not generation: what no view shows stays empty.

## Inputs

- **Photos**: one or more photos of the same static scene.
- **Reference Video**: a video of the scene; views are picked along the camera motion (about one per significant move, at least one
  every 4 s), each replaced by the sharpest frame within 3 frames. Photos, a video or both can be used: the photos come first and the
  video fills the remaining views.
- The first view is the reference: the scene is placed in its camera frame, and views with another aspect ratio are cropped to it.
- **Prompt**: the scene name, used for file names and metadata only.

Best inputs: 8 to 32 overlapping views around a static scene or object, sharp and well exposed; a slow orbit or walk-through video.
Moving objects, reflections, glass and textureless walls reconstruct poorly; a single image gives a depth-based 2.5D result.

## Output

- **Gaussian Splats** (default): the model's own Gaussian head, one Gaussian per pixel merged on a fine voxel grid, the sky and
  depth-edge floaters removed; at most 5 million Gaussians (SH degree 0, up to about 320 MB for 24 views at 952 px).
- **Point Cloud**: the depth points of all views, merged per voxel (at most 2 million), shown as small round splats.

Each result also saves `_points.ply` (plain coloured points) and `_cameras.json` (per view: source, size, camera-to-world matrix in the
file's axes with the camera looking down -Z, pixel intrinsics and fields of view). Files use glTF axes: +Y up, the first camera looking
towards -Z. The thumbnail is rendered from the first view.

## Settings

- **Max Views** (24): more views cover more of the scene and use more VRAM: about 8 GB at 952 px with 24 views (weights included),
  plus about 0.15 GB per extra view; about 6 GB at 518 px.
- **Processing Resolution** (952 px, the upstream default): 518 px is several times faster and lighter but coarser.
- **Filtering**: removes the sky and the points stretched across depth discontinuities (recommended); *Keep Everything* keeps
  them for inspection.

{_LICENSE_NOTE}
"""

WORLDMIRROR_PROMPT_INFOS = """
# Scene Name

WorldMirror 2.0 does not read the prompt: it only names the files and is stored in their metadata. Write a short name such as
`garden statue` or `kitchen corner`. The views decide the result: photos or a video of one static scene from several viewpoints with
plenty of overlap. To reconstruct a generated scene, generate a short orbit video of it first, then use that video as the Reference Video.
"""

WORLDMIRROR_DEEPY_INFOS = ("Multi-view 3D reconstruction (not generation): View Images (Photos: `image_refs`, `video_prompt_type` I) and/or a "
                           "Reference Video (`video_guide`, `video_prompt_type` V; IV for both) of one static scene -> Gaussian "
                           "splats PLY (`model_mode` `gaussians`, default) or point cloud (`points`), plus `_points.ply` and `_cameras.json`. "
                           "No steps or guidance. custom_settings: `max_views` (24), `max_resolution` 518/644/756/840/952 (952), `filtering` "
                           "sky_edges/edges/none. Best: 8-32 overlapping sharp views. Tencent license: not for EU/UK/South Korea; outputs may "
                           "not train other AI models.")

WORLDMIRROR_DEEPY_PROMPT_INFOS = ("`prompt`: a short scene name such as `garden statue`; only used for file names. The views (photos or a video "
                                  "of a static scene from several viewpoints) decide the reconstruction.")
