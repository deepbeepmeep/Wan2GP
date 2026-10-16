# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared canvas styling, comment preservation and resolution lookup for prompt helpers."""
from pathlib import Path


def get_css(prefix):
    return Path(__file__).with_suffix('.css').read_text(encoding='utf-8').replace('layout-', prefix + '-')


def get_javascript():
    return Path(__file__).with_suffix('.js').read_text(encoding='utf-8')
