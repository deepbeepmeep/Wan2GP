"""Compatibility alias for shared download routes."""
import sys
from shared.utils import downloads
sys.modules[__name__] = downloads
