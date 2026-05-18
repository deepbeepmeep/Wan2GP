from functools import partial

import torch


def get_plucker_embedding(intrinsics, cam_c2w, height, width, height_dit=None, width_dit=None, flip_flag=None):
    custom_meshgrid = partial(torch.meshgrid, indexing="ij")
    batch_size, num_frames = intrinsics.shape[:2]

    use_dit_hw = height_dit is not None and width_dit is not None
    if not use_dit_hw:
        height_dit = height
        width_dit = width
    else:
        patch_height = height / height_dit
        patch_width = width / width_dit

    j, i = custom_meshgrid(
        torch.linspace(0, height_dit - 1, height_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
        torch.linspace(0, width_dit - 1, width_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
    )
    i = i.reshape(1, 1, height_dit * width_dit).expand(batch_size, num_frames, height_dit * width_dit) + 0.5
    j = j.reshape(1, 1, height_dit * width_dit).expand(batch_size, num_frames, height_dit * width_dit) + 0.5

    if use_dit_hw:
        i = i * patch_width + (patch_width / 2)
        j = j * patch_height + (patch_height / 2)

    if flip_flag is not None and torch.sum(flip_flag).item() > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, height_dit - 1, height_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
            torch.linspace(width_dit - 1, 0, width_dit, device=cam_c2w.device, dtype=cam_c2w.dtype),
        )
        i_flip = i_flip.reshape(1, 1, height_dit * width_dit).expand(batch_size, 1, height_dit * width_dit) + 0.5
        j_flip = j_flip.reshape(1, 1, height_dit * width_dit).expand(batch_size, 1, height_dit * width_dit) + 0.5
        if use_dit_hw:
            i_flip = i_flip * patch_width + (patch_width / 2)
            j_flip = j_flip * patch_height + (patch_height / 2)
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ cam_c2w[..., :3, :3].transpose(-1, -2)
    rays_o = cam_c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(batch_size, num_frames, height_dit, width_dit, 6)

