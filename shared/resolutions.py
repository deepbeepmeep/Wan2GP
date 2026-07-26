import json
import math
import os
import re

from shared.match_archi import match_nvidia_architecture


RESOLUTION_FILE = "resolutions.json"

DEFAULT_RESOLUTION_CHOICES_4K = [
    # 4K
    ("3840x2176 (16:9)", "3840x2176"),
    ("2176x3840 (9:16)", "2176x3840"),
    ("2880x2880 (1:1)", "2880x2880"),
    ("3840x1664 (21:9)", "3840x1664"),
    ("1664x3840 (9:21)", "1664x3840"),
    ("2048x2048 (1:1)", "2048x2048"),
    # 1440p
    ("1920x1920 (1:1)", "1920x1920"),
    ("2560x1440 (16:9)", "2560x1440"),
    ("1440x2560 (9:16)", "1440x2560"),
    ("1920x1440 (4:3)", "1920x1440"),
    ("1440x1920 (3:4)", "1440x1920"),
    ("2160x1440 (3:2)", "2160x1440"),
    ("1440x2160 (2:3)", "1440x2160"),
    ("1456x1456 (1:1)", "1456x1456"),
    ("2688x1152 (21:9)", "2688x1152"),
    ("1152x2688 (9:21)", "1152x2688"),
]

DEFAULT_RESOLUTION_CHOICES = [
    # 1080p
    ("1920x1088 (16:9)", "1920x1088"),
    ("1088x1920 (9:16)", "1088x1920"),
    ("1440x1440 (1:1)", "1440x1440"),
    ("1536x1024 (3:2)", "1536x1024"),
    ("1024x1536 (2:3)", "1024x1536"),
    ("1920x832 (21:9)", "1920x832"),
    ("832x1920 (9:21)", "832x1920"),
    ("2048x768 (8:3)", "2048x768"),
    ("1024x1792 (4:7)", "1024x1792"),
    ("1088x1088 (1:1)", "1088x1088"),
    # 720p
    ("1024x1024 (1:1)", "1024x1024"),
    ("1280x720 (16:9)", "1280x720"),
    ("720x1280 (9:16)", "720x1280"),
    ("1600x400 (4:1)", "1600x400"),
    ("1280x544 (21:9)", "1280x544"),
    ("544x1280 (9:21)", "544x1280"),
    ("1104x832 (4:3)", "1104x832"),
    ("832x1104 (3:4)", "832x1104"),
    ("960x960 (1:1)", "960x960"),
    # 540p
    ("960x544 (16:9)", "960x544"),
    ("544x960 (9:16)", "544x960"),
    # 480p
    ("832x624 (4:3)", "832x624"),
    ("624x832 (3:4)", "624x832"),
    ("720x720 (1:1)", "720x720"),
    ("832x480 (16:9)", "832x480"),
    ("480x832 (9:16)", "480x832"),
    # 384p
    ("672x384 (16:9)", "672x384"),
    ("384x672 (9:16)", "384x672"),
    ("512x512 (1:1)", "512x512"),
    # 320p
    ("576x320 (16:9)", "576x320"),
    ("320x576 (9:16)", "320x576"),
    ("448x448 (1:1)", "448x448"),
    # 256p
    ("448x256 (7:4)", "448x256"),
    ("256x448 (4:7)", "256x448"),
    ("320x320 (1:1)", "320x320"),
]

GROUP_THRESHOLDS = {
    "256p": 448 * 256,
    "320p": 448 * 448,
    "384p": 512 * 512,
    "480p": 832 * 624,
    "540p": 960 * 544,
    "720p": 1024 * 1024,
    "1080p": 1920 * 1088,
    "1440p": 2560 * 1440,
    "2160p": 3840 * 2176,
}

GROUP_TIERS = {
    "256p": 256,
    "320p": 320,
    "384p": 384,
    "480p": 480,
    "540p": 540,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "2160p": 2160,
}

CATEGORY_ALIASES = {
    "2k": 1440,
    "4k": 2160,
}

_custom_resolutions = None


def is_resolution_value(value):
    return isinstance(value, str) and re.fullmatch(r"\d+x\d+", value.strip().lower()) is not None


def parse_resolution(value):
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def normalize_resolution_choices(resolution_choices, source_name, printer=print):
    if resolution_choices is None:
        return None
    if not isinstance(resolution_choices, list):
        printer(f'"{source_name}" should be a list of 2 elements lists ["Label","WxH"]')
        return None

    normalized = []
    for tup in resolution_choices:
        if not isinstance(tup, (list, tuple)) or len(tup) != 2 or not isinstance(tup[0], str) or not isinstance(tup[1], str):
            printer(f'"{source_name}" contains an invalid list of two elements: {tup}')
            return None
        if not is_resolution_value(tup[1]):
            printer(f'"{source_name}" contains a resolution value that is not in the format "WxH": {tup[1]}')
            return None
        normalized.append((tup[0], tup[1].lower()))
    return normalized


def load_custom_resolution_choices(resolution_file=RESOLUTION_FILE, printer=print):
    global _custom_resolutions
    if _custom_resolutions is not None:
        return _custom_resolutions
    if not os.path.isfile(resolution_file):
        return []

    try:
        with open(resolution_file, "r", encoding="utf-8") as f:
            resolution_choices = json.load(f)
    except Exception as e:
        printer(f'Invalid "{resolution_file}" : {e}')
        resolution_choices = None
    normalized = normalize_resolution_choices(resolution_choices, resolution_file, printer)
    if normalized is None:
        return []
    _custom_resolutions = normalized
    return _custom_resolutions


def reset_custom_resolution_cache():
    global _custom_resolutions
    _custom_resolutions = None


def dedupe_resolution_choices(resolution_choices):
    seen = set()
    deduped = []
    for label, resolution in resolution_choices:
        if resolution in seen:
            continue
        seen.add(resolution)
        deduped.append((label, resolution))
    return deduped


def normalize_block_size(block_size):
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"Invalid resolution block size: {block_size}")
    return block_size


def align_dimension_to_block(value, block_size):
    if block_size <= 1:
        return value
    return max(block_size, value // block_size * block_size)


def align_resolution_value(resolution, block_size):
    block_size = normalize_block_size(block_size)
    width, height = parse_resolution(resolution)
    return f"{align_dimension_to_block(width, block_size)}x{align_dimension_to_block(height, block_size)}"


def align_resolution_label(label, resolution, aligned_resolution):
    if resolution == aligned_resolution:
        return label
    exact_pattern = re.compile(re.escape(resolution), re.IGNORECASE)
    if exact_pattern.search(label):
        return exact_pattern.sub(aligned_resolution, label, count=1)
    return re.sub(r"\d+x\d+", aligned_resolution, label, count=1, flags=re.IGNORECASE)


def align_resolution_choices(resolution_choices, block_size):
    block_size = normalize_block_size(block_size)
    if block_size <= 1:
        return resolution_choices
    return dedupe_resolution_choices(
        (align_resolution_label(label, resolution, aligned_resolution), aligned_resolution)
        for label, resolution in resolution_choices
        for aligned_resolution in [align_resolution_value(resolution, block_size)]
    )


def builtin_resolution_choices(include_4k=False):
    return list(DEFAULT_RESOLUTION_CHOICES_4K if include_4k else []) + list(DEFAULT_RESOLUTION_CHOICES)


def all_global_resolution_choices():
    return dedupe_resolution_choices(builtin_resolution_choices(include_4k=True) + load_custom_resolution_choices())


def default_global_resolution_choices(enable_4k_resolutions=False):
    return dedupe_resolution_choices(builtin_resolution_choices(include_4k=enable_4k_resolutions) + load_custom_resolution_choices())


def keep_resolution_on_model_switch_enabled(value):
    if isinstance(value, str):
        return value.strip().casefold() not in {"0", "false", "no", "off"}
    return value not in (False, 0)


def categorize_resolution(resolution_str):
    width, height = parse_resolution(resolution_str)
    pixel_count = width * height

    for group, threshold in GROUP_THRESHOLDS.items():
        if pixel_count <= threshold:
            return group
    return next(reversed(GROUP_THRESHOLDS))


def _category_tier(category):
    key = str(category).strip().lower()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    if key.endswith("p"):
        key = key[:-1]
    if not key.isdigit():
        return None
    tier = int(key)
    return tier if tier in GROUP_TIERS.values() else None


def _normalize_category_expression(expression):
    def replace_token(match):
        tier = _category_tier(match.group(0))
        return str(tier) if tier is not None else match.group(0)

    return re.sub(r"[a-z0-9]+", replace_token, str(expression).strip().lower())


def normalize_category_expressions(category_expressions):
    if category_expressions is None:
        return []
    if isinstance(category_expressions, str):
        return [category_expressions]
    if isinstance(category_expressions, (list, tuple)):
        return [str(expression) for expression in category_expressions]
    return []


def category_allowed(category, category_expressions):
    expressions = normalize_category_expressions(category_expressions)
    if not expressions:
        return True
    tier = GROUP_TIERS[category]
    return any(match_nvidia_architecture({_normalize_category_expression(expression): True}, tier) for expression in expressions)


def filter_resolution_choices_by_categories(resolution_choices, category_expressions):
    return [
        resolution
        for resolution in resolution_choices
        if category_allowed(categorize_resolution(resolution[1]), category_expressions)
    ]


def closest_resolution(target_resolution, resolution_choices):
    if not resolution_choices:
        return target_resolution
    if not is_resolution_value(target_resolution or ""):
        return resolution_choices[0][1]

    target_width, target_height = parse_resolution(target_resolution)
    target_ratio = target_width / target_height
    target_pixels = target_width * target_height
    target_group = categorize_resolution(target_resolution)
    group_order = list(GROUP_THRESHOLDS)
    target_group_index = group_order.index(target_group)
    grouped_choices = {}
    for choice in resolution_choices:
        grouped_choices.setdefault(categorize_resolution(choice[1]), []).append(choice)
    closest_group = min(grouped_choices, key=lambda group: (abs(group_order.index(group) - target_group_index), abs(GROUP_THRESHOLDS[group] - GROUP_THRESHOLDS[target_group])))

    def score(choice):
        width, height = parse_resolution(choice[1])
        ratio_score = abs(math.log((width / height) / target_ratio))
        pixel_score = abs((width * height) - target_pixels) / target_pixels
        return ratio_score, pixel_score

    return min(grouped_choices[closest_group], key=score)[1]


def resolve_resolution_choices(current_resolution_choice, model_def, enable_4k_resolutions=False, block_size=None):
    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None:
        resolution_choices = normalize_resolution_choices(model_resolutions, "model.resolutions") or []
        if model_def.get("resolutions_categories", None) is not None:
            resolution_choices = dedupe_resolution_choices(resolution_choices + filter_resolution_choices_by_categories(all_global_resolution_choices(), model_def["resolutions_categories"]))
    elif model_def.get("resolutions_categories", None) is not None:
        resolution_choices = filter_resolution_choices_by_categories(all_global_resolution_choices(), model_def["resolutions_categories"])
    else:
        resolution_choices = default_global_resolution_choices(enable_4k_resolutions)
    resolution_choices = align_resolution_choices(resolution_choices, model_def.get("vae_block_size", 16) if block_size is None else block_size)

    if not resolution_choices:
        return [], None
    if current_resolution_choice is not None and not any(current_resolution_choice == resolution[1] for resolution in resolution_choices):
        current_resolution_choice = closest_resolution(current_resolution_choice, resolution_choices)
    elif current_resolution_choice is None and resolution_choices:
        current_resolution_choice = resolution_choices[0][1]

    return resolution_choices, current_resolution_choice


def resolve_model_switch_resolution(source_resolution, target_model_def, enable_4k_resolutions=False, block_size=None):
    if not is_resolution_value(source_resolution or ""):
        return None
    return resolve_resolution_choices(source_resolution, target_model_def, enable_4k_resolutions, block_size)[1]


def group_resolution_choices(resolution_choices, selected_resolution):
    grouped_resolutions = {}
    for resolution in resolution_choices:
        group = categorize_resolution(resolution[1])
        grouped_resolutions.setdefault(group, []).append(resolution)

    if not grouped_resolutions:
        return [], [], None

    available_groups = [group for group in GROUP_THRESHOLDS if group in grouped_resolutions]
    available_groups.reverse()

    if selected_resolution is None and available_groups:
        selected_group = available_groups[0]
    else:
        selected_group = categorize_resolution(selected_resolution)
        if selected_group not in grouped_resolutions and available_groups:
            selected_group = available_groups[0]

    return available_groups, grouped_resolutions.get(selected_group, []), selected_group


def group_choices(resolution_choices, selected_group):
    return [resolution for resolution in resolution_choices if categorize_resolution(resolution[1]) == selected_group]


def remember_last_resolution(last_resolution_per_group, resolution):
    last_resolution_per_group[categorize_resolution(resolution)] = resolution
    return last_resolution_per_group
