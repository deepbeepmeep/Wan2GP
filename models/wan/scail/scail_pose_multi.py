from __future__ import annotations

from typing import Sequence

import numpy as np


def get_largest_bbox_indices(bboxes: Sequence[Sequence[float]], num_bboxes: int = 2) -> list[int]:
    """Return indices of the largest bboxes by area (normalized coords are fine)."""
    if num_bboxes <= 0:
        return []

    bboxes_with_area: list[tuple[int, float]] = []
    for i, bbox in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = bbox
        except Exception:
            continue
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        bboxes_with_area.append((i, area))

    bboxes_with_area.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _area in bboxes_with_area[: min(num_bboxes, len(bboxes_with_area))]]


def change_poses_to_limit_num(poses, bboxes, num_bboxes: int = 2):
    """Trim DWpose outputs to the top-N bboxes per frame (SCAIL-Pose multi-human helper)."""
    bboxes = list(bboxes)
    for frame_idx, (pose, bbox_list) in enumerate(zip(poses, bboxes)):
        if not bbox_list:
            continue

        largest_indices = get_largest_bbox_indices(bbox_list, num_bboxes)
        if not largest_indices:
            continue

        bodies = pose.get("bodies", {})
        if "candidate" in bodies:
            bodies["candidate"] = bodies["candidate"][largest_indices]
        if "subset" in bodies:
            bodies["subset"] = bodies["subset"][largest_indices]
        pose["bodies"] = bodies

        faces = pose.get("faces", None)
        if isinstance(faces, np.ndarray):
            pose["faces"] = faces[largest_indices]
        elif isinstance(faces, list):
            pose["faces"] = [faces[i] for i in largest_indices if i < len(faces)]

        hands = pose.get("hands", None)
        hand_indices = [j for i in largest_indices for j in (2 * i, 2 * i + 1)]
        if isinstance(hands, np.ndarray):
            pose["hands"] = hands[hand_indices]
        elif isinstance(hands, list):
            pose["hands"] = [hands[i] for i in hand_indices if i < len(hands)]

        bboxes[frame_idx] = [bbox_list[i] for i in largest_indices if i < len(bbox_list)]

    return poses, bboxes

