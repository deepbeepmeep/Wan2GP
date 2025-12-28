"""
Official SCAIL-Pose 3D retarget camera intrinsics solver.

Upstream: https://github.com/zai-org/SCAIL-Pose (NLFPoseExtract/align3d.py)
Note: the upstream file imports `sympy` but doesn't use it; this adaptation omits that import.
"""

from __future__ import annotations

import numpy as np


def solve_new_camera_params_central(three_d_points, focal_length, imshape, new_2d_points):
    from scipy.optimize import minimize

    def objective(params):
        m, s, p, q = params
        K_new = np.array(
            [
                [focal_length * m, 0, imshape[1] / 2 + p],
                [0, focal_length * m * s, imshape[0] / 2 + q],
                [0, 0, 1],
            ]
        )

        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 * 8 + error

    initial_params = [1.0, 1.0, 0.0, 0.0]
    result = minimize(
        objective,
        initial_params,
        bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])],
    )

    m, s, p, q = result.x
    K_final = np.array(
        [
            [focal_length * m, 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1],
        ]
    )
    return K_final, m


def solve_new_camera_params_down(three_d_points, focal_length, imshape, new_2d_points):
    from scipy.optimize import minimize

    def objective(params):
        m, s, p, q = params
        K_new = np.array(
            [
                [focal_length * m, 0, imshape[1] / 2 + p],
                [0, focal_length * m * s, imshape[0] / 2 + q],
                [0, 0, 1],
            ]
        )

        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 + error * 4

    initial_params = [1.0, 1.0, 0.0, 0.0]
    result = minimize(
        objective,
        initial_params,
        bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])],
    )

    m, s, p, q = result.x
    K_final = np.array(
        [
            [focal_length * m, 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1],
        ]
    )
    return K_final, m

