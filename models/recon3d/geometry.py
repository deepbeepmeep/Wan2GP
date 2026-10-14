# Point-map helpers of the reconstruction models: edge masks (torch ports of the depth_edge / normals_edge / points_to_normals functions
# of facebookresearch/map-anything, Apache-2.0, and of depth_map_edge of EasternJournalist/utils3d, MIT, see LICENSE_utils3d), grid
# meshes of point maps (utils3d build_mesh_from_map), OpenCV to glTF axes, camera records and point subsampling.
import json
import math

import torch
import torch.nn.functional as F

OPENCV_TO_GLTF = (1.0, -1.0, -1.0)  # 180 degrees about X: +X right, +Y down, +Z forward -> +X right, +Y up, +Z towards the viewer


def _pad(tensor, value):
    """(H, W, ...) padded by one row and column on each side with value."""
    padded = tensor.new_full((tensor.shape[0] + 2, tensor.shape[1] + 2, *tensor.shape[2:]), value)
    padded[1:-1, 1:-1] = tensor
    return padded


def _max_pool(x):
    return F.max_pool2d(x[None, None], 3, stride=1, padding=1)[0, 0]


def depth_edges(depth, mask, rtol):
    """Pixels whose valid 3x3 neighbours differ in depth by more than rtol times their own depth."""
    diff = _max_pool(torch.where(mask, depth, -math.inf)) + _max_pool(torch.where(mask, -depth, -math.inf))
    return diff / depth > rtol


def disparity_edges(depth, mask, ltol):
    """utils3d depth_map_edge(ltol), as the MoGe demo removes edges: pixels whose 3x3 window holds a relative disparity Laplacian above
    ltol and one below -ltol, invalid pixels counting as zero disparity (infinite depth); NaNs are ignored by the window pooling."""
    disparity = torch.where(mask, 1 / depth, 0)
    laplacian = (disparity - F.avg_pool2d(disparity[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]) / disparity
    window_max = _max_pool(laplacian.nan_to_num(nan=-math.inf, posinf=math.inf, neginf=-math.inf))
    window_min = -_max_pool((-laplacian).nan_to_num(nan=-math.inf, posinf=math.inf, neginf=-math.inf))
    return (window_max > ltol) & (window_min < -ltol)


def points_to_normals(points, mask):
    """Normals (H, W, 3) of a point map from the cross products of its four neighbour pairs, and where they are defined."""
    pts, valid_pad = _pad(points, 0.0), _pad(mask, False)
    center = pts[1:-1, 1:-1]
    up, left, down, right = pts[:-2, 1:-1] - center, pts[1:-1, :-2] - center, pts[2:, 1:-1] - center, pts[1:-1, 2:] - center
    normal = torch.stack([torch.linalg.cross(up, left), torch.linalg.cross(left, down), torch.linalg.cross(down, right), torch.linalg.cross(right, up)])
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-12)
    m_up, m_left, m_down, m_right = valid_pad[:-2, 1:-1], valid_pad[1:-1, :-2], valid_pad[2:, 1:-1], valid_pad[1:-1, 2:]
    valid = torch.stack([m_up & m_left, m_left & m_down, m_down & m_right, m_right & m_up]) & mask[None]
    normal = (normal * valid[..., None]).sum(dim=0)
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-12)
    normal_mask = valid.any(dim=0)
    return torch.where(normal_mask[..., None], normal, 0.0), normal_mask


def normal_edges(normals, mask, tolerance):
    """Pixels within one pixel of a valid 3x3 window whose normals turn by more than tolerance degrees."""
    height, width = mask.shape
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)
    padded = F.pad(normals.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    padded_mask = F.pad(mask[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0] > 0
    angle = torch.zeros_like(mask, dtype=normals.dtype)
    for dy in range(3):
        for dx in range(3):
            cosine = (normals * padded[dy:dy + height, dx:dx + width]).sum(dim=-1).clamp(-1, 1)
            angle = torch.maximum(angle, torch.where(padded_mask[dy:dy + height, dx:dx + width], torch.arccos(cosine), 0.0))
    return _max_pool(angle) > math.radians(tolerance)


def grid_faces(mask):
    """Two counter-clockwise (seen from the camera) triangles per 2x2 block of valid pixels, as indices into the row-major pixels."""
    height, width = mask.shape
    index = torch.arange(height * width, device=mask.device).view(height, width)
    top_left, bottom_left, bottom_right, top_right = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    quads = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    return torch.cat([torch.stack([top_left, bottom_left, bottom_right], dim=-1)[quads], torch.stack([top_left, bottom_right, top_right], dim=-1)[quads]])


def opencv_to_gltf(vectors):
    return vectors * vectors.new_tensor(OPENCV_TO_GLTF)


def subsample(count, max_points, seed, device):
    """Indices of at most max_points of count points (a seeded random subset, kept in order), or None to keep them all."""
    if count <= max_points:
        return None
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randperm(count, generator=generator, device=device)[:max_points].sort().values


def cameras_json(intrinsics, poses, image_size, sources):
    """_cameras.json bytes: per view the pixel intrinsics (OpenCV convention, pixel centres at integer coordinates, of the processed
    image) and the camera-to-world matrix in glTF conventions (camera +X right, +Y up, looking down -Z; world +Y up), the world being
    the frame of the first camera."""
    flip = torch.diag(torch.tensor((*OPENCV_TO_GLTF, 1.0), device=poses.device))
    poses = flip @ poses @ flip
    height, width = image_size
    cameras = []
    for index, (matrix, pose) in enumerate(zip(intrinsics.tolist(), poses.tolist())):
        fx, fy, cx, cy = matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]
        cameras.append({"index": index, "source": sources[index], "width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "fov_y_degrees": math.degrees(2 * math.atan(height / (2 * fy))), "camera_to_world": pose})
    record = {"world": "glTF axes (+Y up, +Z towards the viewer), metres, origin and axes of the first camera",
              "camera_to_world": "row-major 4x4, glTF/OpenGL camera axes (+X right, +Y up, looking down -Z)",
              "intrinsics": "pixels of the processed image, OpenCV convention (pixel centres at integer coordinates)", "cameras": cameras}
    return json.dumps(record, indent=1).encode("utf-8")


def photo_view(points, intrinsics, image_size, azimuth=20.0, elevation=10.0):
    """Preview camera (view, projection) of a single-photo reconstruction (glTF points): the photo's camera turned by azimuth and
    elevation degrees about the median point, with the field of view of the photo's longer side so that the square thumbnail holds
    the whole frame."""
    from shared.mesh3d.render import look_at, perspective

    center = points[::max(1, len(points) // 200000)].median(dim=0).values
    distance = float(center.norm())
    x, y, z = (-center / distance).tolist()
    yaw, pitch = math.atan2(x, z) + math.radians(azimuth), math.asin(max(-1.0, min(1.0, y))) + math.radians(elevation)
    eye = center + distance * center.new_tensor([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    height, width = image_size
    fov = max(2 * math.atan(height / (2 * float(intrinsics[1, 1]))), 2 * math.atan(width / (2 * float(intrinsics[0, 0]))))
    return look_at(eye, center), perspective(min(math.degrees(fov) * 1.1, 150.0), 1.0, 0.01 * distance, 100 * distance, device=points.device)
