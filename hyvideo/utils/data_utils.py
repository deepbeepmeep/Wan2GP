import numpy as np
import math
from PIL import Image
import torch
import copy
import string
import random


def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def black_image(width, height):
    """generate a black image

    Args:
        width (int): image width
        height (int): image height

    Returns:
        _type_: a black image
    """
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """get the closest ratio in the buckets

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    aspect_ratio = float(height) / float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return buckets[closest_ratio_id], float(closest_ratio)


def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    """generate crop size list

    Args:
        base_size (int, optional): the base size for generate bucket. Defaults to 256.
        patch_size (int, optional): the stride to generate bucket. Defaults to 32.
        max_ratio (float, optional): th max ratio for h or w based on base_size . Defaults to 4.0.

    Returns:
        list: generate crop size list
    """
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def align_floor_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.floor(value / alignment) * alignment)
