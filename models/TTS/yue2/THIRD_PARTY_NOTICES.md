# Third-party code notices

The Oobleck decoder and SnakeBeta implementation in `vae.py` is derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The decoder retains the original activation equations and module hierarchy.
Weight normalization is folded into ordinary convolution weights during repacking;
the training paths are omitted. `hum.py` adds the matching Oobleck encoder
for the optional hum-conditioning workflow, using the same licensed building blocks.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0;
see MODEL_LICENSE for the scope and full terms. This does not relicense third-party code.

Mothersuperior's YuE2 Hum-to-Song contribution supplies the combined real-audio
and hum acoustic adapter, four learned conditioning projections, and the
pitch-carrier / open-score continuation method:
https://huggingface.co/Mothersuperior/YuE2-hum-to-song
The combined adapter includes the rank-32 real-audio v4 decoder LoRA:
https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
The hum adapter weights are CC BY-NC 4.0. Conversion preserves source revisions
and tensor precision in `YuE2_Hum_manifest.json`; full input/output replacement
weights are represented as deltas against WanGP's base acoustic checkpoint.
