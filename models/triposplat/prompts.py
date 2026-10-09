TRIPOSPLAT_INFOS = """
# TripoSplat

TripoSplat (VAST / Tripo, MIT) turns one picture of an object into a 3D Gaussian splat: up to 262,144 small coloured, semi-transparent
3D blobs that together render the object from any angle. Splats look photographic and keep fine, fuzzy or thin details (fur, foliage,
hair, fabric), but they are not a mesh: they suit viewing, previews, AR/VR scenes and splat-capable engines, not 3D printing, UV
texturing or rigging.

## Inputs

- **Object Image** (required): one picture of a single object. Its background is removed automatically, then the object is cropped
  and centered. A whole, uncropped object on a plain background, in a front or three-quarter view with even light, works best. An
  image with a transparent background keeps its own cutout.
- **Asset Name / Description**: names the files and is stored with the settings. TripoSplat only reads the image, so this text does
  not change the result.

## Settings

- **Number of Inference Steps** (20): 10-20 is the recommended range; fewer steps are faster but less detailed.
- **Guidance** (3.0): higher values follow the image more closely and add detail, too high can oversaturate the colours; 1 turns
  guidance off (twice as fast).
- **Shift Scale** (3.0): spends more of the steps on the coarse shape; 1 is a uniform schedule.
- **Gaussian Count**: 256K keeps the most detail; 32K-128K give lighter files (about 68 bytes per Gaussian) that display faster, for
  background props. The model runs once whatever the count: only the final decoding changes.
- **Seed / Number of Samples**: each sample of a batch uses the seed plus its index.

## Output

A Gaussian splat PLY (the standard 3D Gaussian Splatting layout, colours without view dependence, Y up), viewable in the 3D gallery
and in splat viewers and editors such as SuperSplat or SparkJS. The pictured side usually faces +Z (the default front view), but the
model places the object in its own frame and some seeds turn it a quarter turn. Stylised objects, props, toys and creatures work
best; faces of people often come out blurred, and the hidden sides are inferred.

**License:** TripoSplat is MIT (models/triposplat/LICENSE). Its DINOv3 image encoder is under Meta's DINOv3 License (LICENSE_DINOv3.md beside the weights).
"""

TRIPOSPLAT_PROMPT_INFOS = """
# TripoSplat Asset Name

TripoSplat builds the 3D Gaussian splat from the Object Image alone: the text is not a prompt. Write a short name of the asset, such
as `Ceramic teapot` or `Toy pirate ship`; it becomes the output file name and is stored with the settings.

To create an asset from text, first generate the object with an image model (one centered object, whole in frame, plain background,
three-quarter view, soft even light), then send that image to TripoSplat as its Object Image.
"""

TRIPOSPLAT_DEEPY_INFOS = ("Image to 3D Gaussian splat (PLY, Y up), not a mesh. Needs exactly one reference image (`image_refs`): one whole object "
                          "on a plain background (removed automatically). `prompt` = asset name only, not read by the model. Defaults: 20 steps, "
                          "guidance 3 (1 = off, faster), flow_shift 3. custom_settings: `num_gaussians` 32768/65536/131072/196608/262144 "
                          "(detail vs file size, ~68 bytes per Gaussian). batch_size = candidate splats. Best for stylised objects, props, "
                          "creatures; human faces come out blurred.")

TRIPOSPLAT_DEEPY_PROMPT_INFOS = ("`prompt`: a short asset name such as `Toy pirate ship`; it names the file and does not change the splat, which "
                                 "depends only on the reference image (one centred, uncropped object, plain background, three-quarter view).")

ASSET_NAME_INSTRUCTIONS = """You name 3D assets. Look at the image and name the single main object it shows in 2 to 6 words, for example: Toy
pirate ship, Ceramic teapot with blue flowers, Stone cottage. Keep the words of the user's text when it already names the object. Answer with
the name only, without punctuation, quotes or commentary."""
