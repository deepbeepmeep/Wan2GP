"""
Sparse SageAttention module for efficient attention computation.

This module provides sparse attention mechanisms with INT8 quantization
and Triton kernel implementations for improved performance.

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

from .core import sparse_sageattn

__all__ = ['sparse_sageattn']
