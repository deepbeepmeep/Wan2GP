# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from .triton_runtime import install_prebuilt_helper

# Shared modules are imported before attention libraries or model kernels start Triton.
install_prebuilt_helper()
