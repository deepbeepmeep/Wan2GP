"""Help texts and prompt enhancer instructions of AnimateAnyMesh."""

INFOS = """
# AnimateAnyMesh

AnimateAnyMesh (HUST and Alibaba DAMO, ICCV 2025, Apache-2.0) animates a static 3D mesh from a text description of its motion. It
needs no skeleton or rig: it generates the trajectory of every vertex over 16 frames and returns the original mesh, with its
textures, UVs and vertex colours, as a morph-target animation (GLB) that plays in the 3D gallery and in any glTF viewer.

## Inputs

- **Input Mesh** (required): a .glb, .obj, .ply or .stl file, or a 3D gallery asset sent with "To Input Mesh". Any object works:
  characters, animals, creatures, vehicles, plants, objects. Every part of the file is animated together as one object. The
  object must stand upright with +Y up (the glTF convention): a lying or Z-up file (common for STL) moves much less or wrongly,
  so rotate it first. Meshes of the 3D gallery (Hunyuan3D, TRELLIS.2...) can be sent directly.
- **Motion Description** (the prompt): what the object does, in English, such as `The dragon flaps its wings and flies`.

## Settings

- **Number of Inference Steps** (63, the upstream setting): steps of the flow sampler, two model evaluations each.
- **Guidance** (3.0): how strongly the motion follows the description. Raise it (4-5) when the motion is too small; very high
  values can distort the mesh.
- **Seed / Number of Samples**: results vary a lot from one seed to another. Generate several samples (each uses the seed plus its
  index) and keep the best one; upstream authors typically tried about three prompt/seed combinations per mesh.
- **Playback Frame Rate** (10 fps): speed of the 16 frames; it does not change the motion.
- **Playback**: *Forward* (16 frames, the loop jumps back to the start) or *Forward then Backward* (31 frames, a seamless loop).
- **Motion Trajectories** (0 = automatic: one eighth of the vertices, at least 512): fewer trajectories give larger motions but can
  distort the shape.
- **Max Vertices** (50000, the training limit): denser meshes are animated through a simplified copy that every original vertex
  follows.

## Output and limits

A GLB with 16 (or 31) morph targets driven by one looping animation; the rest pose is the input mesh. The mesh deforms but its
topology is unchanged, so a closed mesh cannot split into pieces. Motions are short (16 frames) and work best for the motions most
common in animated 3D assets: walking, running, jumping, flying, swimming, waving, dancing, rotating, bending, swaying. Rare or very
specific motions may be ignored: rephrase the description or try other seeds. Left/right details are not always followed (a
wave of the right hand can raise both arms), and very low-poly meshes (a few hundred vertices) tend to move little.

**License:** Apache-2.0 (models/animate_any_mesh/LICENSE).
"""

PROMPT_INFOS = """
# AnimateAnyMesh Motion Description

Describe the motion in one short English sentence (the CLIP text encoder reads at most 77 tokens), with the subject first and the
action in the present tense. Describe only the motion: the mesh already defines the appearance, so colours, materials, background
and camera words have no effect.

Examples:
- `The object is flying` (the upstream example, here with a dragon)
- `The dog is running forward`
- `The character waves its right hand`
- `The bird flaps its wings`
- `The fish swims, its tail swinging from side to side`
- `The tree sways in the wind`
- `The robot is dancing`
- `The car bounces up and down`

Tips: name the moving part when it matters (`its tail`, `its head`, `its arms`); keep one main action; if the motion is too small,
raise Guidance or lower Motion Trajectories; if it does not match, try other seeds or rephrase with more common words.
"""

DEEPY_INFOS = """Mesh animation: `custom_guide` = a static upright (+Y up) mesh file (.glb/.obj/.ply/.stl, required) + `prompt` = an English motion description -> animated GLB (the same mesh, textures kept, with 16 morph-target frames; no rig). Defaults: 63 steps, guidance 3 (raise to 4-5 for small motions). Custom settings: `fps` (10), `playback` forward / pingpong (seamless 31-frame loop), `trajectories` (0 = auto vertices/8, fewer = larger motion but distortion), `max_vertices` (50000; denser meshes are animated through a simplified copy). Results vary a lot per seed: generate several samples."""

DEEPY_PROMPT_INFOS = """One short English sentence describing only the motion, subject first, present tense: `The dragon flaps its wings and flies`, `The dog is running`, `The character waves its right hand`. No appearance, camera or style words."""

MOTION_ENHANCER = """You write motion descriptions for AnimateAnyMesh, a model that animates an existing 3D mesh from a short English caption of its motion.
Rewrite the user's request as ONE English sentence of at most 25 words that describes only how the object moves.
- Start with the subject: keep the subject the user names (for example "The dragon"), otherwise write "The object".
- Use the present tense and common motion verbs (walks, runs, jumps, flies, flaps its wings, swims, waves, turns its head, dances, bends, sways, rotates).
- Name the moving parts when the request implies them (its wings, its tail, its right arm).
- Keep every motion, direction and exclusion the user states, and keep their exact wording when it is already English.
- Write in English even when the request is in another language: the model's text encoder only understands English.
- Do not describe appearance, colours, materials, background, lighting, camera or style, and do not add actions the user did not ask for beyond what makes the requested motion concrete.
Return only the sentence, without quotes or comments."""
