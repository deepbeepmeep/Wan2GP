# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Compatibility alias for shared download routes."""
import sys
from shared.utils import downloads
sys.modules[__name__] = downloads
