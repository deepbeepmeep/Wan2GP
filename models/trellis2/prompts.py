TRELLIS2_INFOS = """
# TRELLIS.2

TRELLIS.2 (Microsoft, 4B parameters, MIT) turns one image of an object into a 3D asset: a GLB mesh with PBR materials (base color,
metallic, roughness and optionally alpha), generated in four stages: a coarse occupancy structure, the shape, then the surface
materials, and finally mesh extraction, remeshing into a clean closed surface, simplification, UV unwrapping and texture baking.

## Inputs

- **Object Image** (required): one picture of a single object. A transparent image (PNG/WebP with alpha) is used as is, its alpha
  being the object mask; otherwise the background is removed automatically. The object is then cropped and centered. Whole, unoccluded objects on a plain background, seen from a three-quarter view, work best; cropped, cluttered or
  very dark images give incomplete shapes.
- **Asset Name / Description**: names the files and is stored in the metadata. TRELLIS.2 is conditioned on the image only, so this
  text does not change the result.

## Pipelines

- **512**: fastest, coarse geometry (512 voxels across the object).
- **1024**: shape generated directly at 1024.
- **1024 Cascade** (recommended): a 512 shape is generated first and refined at 1024, which keeps the 512 proportions with 1024
  details.
- **1536 Cascade**: the same at 1536 for the finest details; slower and more memory hungry. For complex objects its resolution is
  lowered by steps of 128 until the shape fits in **Max Tokens**.

## Settings

- **Number of Inference Steps** (12): steps of each stage (structure, shape, texture).
- **Guidance** (7.5): image guidance of the structure and shape stages, applied to the early steps with a rescale, as in the
  official pipeline. Lower values give simpler shapes, higher values follow the image more strictly.
- **Texture Guidance** (1 = off, the official setting): above 1 the materials follow the image more strongly but can saturate.
- **Surface Output**: *PBR Textures* bakes textures of **Texture Size** on a UV atlas; *Vertex Colors* skips the UV unwrapping
  (faster, base color only, needs a dense mesh); *Shape Only* skips the texture stage.
- **Max Faces** (100,000): the raw mesh (millions of faces) is simplified to this face count; lower it for real-time use, raise it
  for printing or close-ups. The UV unwrapping runs on the CPU and slows down quickly with the face count (about 30 s at 100,000
  faces, 1.5 min at 200,000).
- **Transparency**: *Opaque* (default, as the official exporter) ignores the generated alpha; *Alpha Blending* keeps it for glass
  or thin materials, which some viewers sort or render less reliably.
- **Seed / Batch Size**: each asset of a batch uses the seed plus its index.

## Output

GLB files (Y up) that open in any glTF viewer, game engine or 3D editor. The asset is generated in the canonical pose learned from
the training data: it usually faces +Z, but nearly symmetric objects can come out turned by 90 or 180 degrees depending on the seed.
The surface is rebuilt as a clean shell around the generated one (slightly thicker thin parts), then simplified. Known limitations:
very thin parts can merge or show small holes, and transparent or reflective objects get approximate materials.

**License:** TRELLIS.2 is MIT (models/trellis2/LICENSE), as is the TRELLIS sparse-structure decoder it reuses. Its DINOv3 image encoder
is under Meta's DINOv3 License (LICENSE_DINOv3.md beside the weights).
"""

TRELLIS2_PROMPT_INFOS = """
# TRELLIS.2 Asset Name

TRELLIS.2 builds the 3D asset from the Object Image alone: the text is not a prompt. Write a short name or description of the asset,
such as `Bronze lion statue` or `Red armchair, low poly reference`; it becomes the output file name and is stored with the settings.

To create an asset from text, first generate the object with an image model (a single centered object, plain background, whole
object in frame, three-quarter view, soft even light), then send that image to TRELLIS.2 as its Object Image.
"""

TRELLIS2_DEEPY_INFOS = """Image-to-3D: one `image_refs` image of a single object (an RGBA image's alpha is used as the mask, else background removal) -> GLB mesh with PBR textures (Y up). The prompt is only the asset name (file name), not a conditioning. `model_mode`: 0 = 512 (fast, coarse), 1 = 1024, 2 = 1024 Cascade (default, best balance), 3 = 1536 Cascade (finest, slowest). Defaults: 12 steps, guidance 7.5, texture guidance 1 (off). Custom settings: `texture_mode` texture/vertex/none, `max_faces` (100000; CPU UV unwrap ~30 s at 100k, ~90 s at 200k), `texture_size` 1024/2048/4096, `max_tokens` (1536 cascade cap), `alpha_mode` opaque/blend (opaque default; blend keeps generated transparency). Best inputs: whole object, plain background, three-quarter view; for text-to-3D generate such an image first."""

TRELLIS2_DEEPY_PROMPT_INFOS = """The prompt is optional: a short asset name such as `Bronze lion statue` that names the output file; it does not change the 3D result, which depends only on the reference image."""
