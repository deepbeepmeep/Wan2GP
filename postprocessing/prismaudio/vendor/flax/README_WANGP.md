This is a reduced Flax/Linen subset vendored for PrismAudio's VideoPrism feature extractor.

It contains only the modules exercised by VideoPrism's `from flax import linen as nn` path, including Linen modules, params, dropout, scan, and remat. It intentionally excludes Flax training/checkpoint APIs so WanGP does not install `orbax-checkpoint`.

Derived from Flax 0.12.7. See `LICENSE`.
