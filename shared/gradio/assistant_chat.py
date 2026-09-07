"""Compatibility import; the shared chat implementation lives in Deepy."""
import sys
from shared.deepy import chat

sys.modules[__name__] = chat
