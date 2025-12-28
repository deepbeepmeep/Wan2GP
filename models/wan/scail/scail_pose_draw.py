"""
2D pose overlay drawing utilities from the official SCAIL-Pose repo.

Upstream:
- https://github.com/zai-org/SCAIL-Pose (pose_draw/draw_utils.py)
- https://github.com/zai-org/SCAIL-Pose (pose_draw/draw_pose_utils.py)

Only the functions required by `render_nlf_as_images` are included.
"""

from __future__ import annotations

import math
import random

import cv2
import matplotlib
import numpy as np

eps = 0.01


def draw_bodypose_augmentation(canvas, candidate, subset, drop_aug=True, shift_aug=False, all_cheek_aug=False):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    if drop_aug:
        arr_drop = list(range(17))
        k_drop = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
        drop_indices = random.sample(arr_drop, k_drop)
    else:
        drop_indices = []
    if shift_aug:
        shift_indices = random.sample(list(range(17)), 2)
    else:
        shift_indices = []
    if all_cheek_aug:
        drop_indices = list(range(13))

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            if i in drop_indices:
                continue

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if i in shift_indices:
                mX = mX + random.uniform(-length / 4, length / 4)
                mY = mY + random.uniform(-length / 4, length / 4)
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        if all_cheek_aug:
            if i not in [0, 14, 15, 16, 17]:
                continue
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape
    stickwidth_thin = min(max(int(min(H, W) / 300), 1), 2)

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=stickwidth_thin,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth_thin, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks, optimized_face=True):
    H, W, C = canvas.shape
    stickwidth = min(max(int(min(H, W) / 200), 1), 3)
    stickwidth_thin = min(max(int(min(H, W) / 300), 1), 2)

    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk_idx, lmk in enumerate(lmks):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if optimized_face:
                    if lmk_idx in list(range(17, 27)) + list(range(36, 70)):
                        cv2.circle(canvas, (x, y), stickwidth_thin, (255, 255, 255), thickness=-1)
                else:
                    cv2.circle(canvas, (x, y), stickwidth, (255, 255, 255), thickness=-1)
    return canvas


def draw_pose(
    pose,
    H,
    W,
    show_feet=False,
    show_body=True,
    show_hand=True,
    show_face=True,
    show_cheek=False,
    dw_bgr=False,
    dw_hand=False,
    aug_body_draw=False,
    optimized_face=False,
):
    final_canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for i in range(len(pose["bodies"]["candidate"])):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        bodies = pose["bodies"]
        faces = pose["faces"][i : i + 1]
        hands = pose["hands"][2 * i : 2 * i + 2]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i : i + 1]

        if show_cheek:
            if show_body:
                raise AssertionError("show_cheek and show_body cannot be True at the same time")
            canvas = draw_bodypose_augmentation(
                canvas, candidate, subset, drop_aug=True, shift_aug=False, all_cheek_aug=True
            )

        if show_hand:
            canvas = draw_handpose(canvas, hands)

        if show_face:
            canvas = draw_facepose(canvas, faces, optimized_face=optimized_face)

        final_canvas = final_canvas + canvas
    return final_canvas


def scale_image_hw_keep_size(img, scale_h, scale_w):
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale_h), int(W * scale_w)
    scaled = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    result = np.zeros_like(img)

    if new_H >= H:
        y_start_src = (new_H - H) // 2
        y_end_src = y_start_src + H
        y_start_dst = 0
        y_end_dst = H
    else:
        y_start_src = 0
        y_end_src = new_H
        y_start_dst = (H - new_H) // 2
        y_end_dst = y_start_dst + new_H

    if new_W >= W:
        x_start_src = (new_W - W) // 2
        x_end_src = x_start_src + W
        x_start_dst = 0
        x_end_dst = W
    else:
        x_start_src = 0
        x_end_src = new_W
        x_start_dst = (W - new_W) // 2
        x_end_dst = x_start_dst + new_W

    result[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = scaled[y_start_src:y_end_src, x_start_src:x_end_src]
    return result


def draw_pose_to_canvas_np(
    poses,
    pool,
    H,
    W,
    reshape_scale,
    show_feet_flag=False,
    show_body_flag=True,
    show_hand_flag=True,
    show_face_flag=True,
    show_cheek_flag=False,
    dw_bgr=False,
    dw_hand=False,
    aug_body_draw=False,
):
    canvas_np_lst = []
    for pose in poses:
        if reshape_scale > 0 and pool is not None:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(
            pose,
            H,
            W,
            show_feet_flag,
            show_body_flag,
            show_hand_flag,
            show_face_flag,
            show_cheek_flag,
            dw_bgr,
            dw_hand,
            aug_body_draw,
            optimized_face=True,
        )
        canvas_np_lst.append(canvas)
    return canvas_np_lst

