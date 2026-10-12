# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Compatibility import; the shared chat implementation lives in Deepy."""
import sys
from shared.deepy import chat

sys.modules[__name__] = chat
