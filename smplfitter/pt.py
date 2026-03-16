"""Import-time stub for optional `smplfitter.pt` dependency."""

from __future__ import annotations

import torch


class BodyModel(torch.nn.Module):
    def __init__(self, *args, num_vertices: int = 6890, num_joints: int = 24, **kwargs):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_joints = num_joints

    def forward(self, pose_rotvecs, shape_betas, trans):
        batch = pose_rotvecs.shape[0] if hasattr(pose_rotvecs, "shape") else 1
        device = pose_rotvecs.device if hasattr(pose_rotvecs, "device") else "cpu"
        vertices = torch.zeros((batch, self.num_vertices, 3), device=device)
        joints = torch.zeros((batch, self.num_joints, 3), device=device)
        return {"vertices": vertices, "joints": joints}


class BodyFitter:
    def __init__(self, body_model: BodyModel):
        self.body_model = body_model

    def fit(
        self,
        vertices,
        joints,
        *,
        requested_keys=None,
        **kwargs,
    ):
        batch = vertices.shape[0] if hasattr(vertices, "shape") else 1
        device = vertices.device if hasattr(vertices, "device") else "cpu"
        requested = set(requested_keys or [])
        result = {}
        if "pose_rotvecs" in requested or not requested:
            result["pose_rotvecs"] = torch.zeros((batch, self.body_model.num_joints, 3), device=device)
        if "shape_betas" in requested or not requested:
            result["shape_betas"] = torch.zeros((batch, 10), device=device)
        if "trans" in requested or not requested:
            result["trans"] = torch.zeros((batch, 3), device=device)
        return result
