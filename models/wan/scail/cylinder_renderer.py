"""
SCAIL Cylinder Renderer - Render 3D skeleton as cylindrical bones.
Uses Taichi for GPU-accelerated rendering with proper occlusion handling.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

# Lazy import to avoid printing version message at app startup
ti = None
TAICHI_AVAILABLE = None  # Will be set on first use


def _ensure_taichi_imported():
    global ti, TAICHI_AVAILABLE
    if TAICHI_AVAILABLE is not None:
        return TAICHI_AVAILABLE
    try:
        import taichi as _ti
        ti = _ti
        TAICHI_AVAILABLE = True
    except ImportError:
        TAICHI_AVAILABLE = False
        print("[SCAIL] Taichi not available, using CPU rendering (slower)")
    return TAICHI_AVAILABLE


# Skeleton topology - pairs of keypoint indices that form bones
# Based on COCO 18-keypoint format
SKELETON_CONNECTIONS = [
    # Head
    (0, 1),   # Nose -> Neck
    (0, 14),  # Nose -> R_Eye
    (0, 15),  # Nose -> L_Eye
    (14, 16), # R_Eye -> R_Ear
    (15, 17), # L_Eye -> L_Ear

    # Torso
    (1, 2),   # Neck -> R_Shoulder
    (1, 5),   # Neck -> L_Shoulder
    (2, 8),   # R_Shoulder -> R_Hip
    (5, 11),  # L_Shoulder -> L_Hip
    (8, 11),  # R_Hip -> L_Hip

    # Right arm
    (2, 3),   # R_Shoulder -> R_Elbow
    (3, 4),   # R_Elbow -> R_Wrist

    # Left arm
    (5, 6),   # L_Shoulder -> L_Elbow
    (6, 7),   # L_Elbow -> L_Wrist

    # Right leg
    (8, 9),   # R_Hip -> R_Knee
    (9, 10),  # R_Knee -> R_Ankle

    # Left leg
    (11, 12), # L_Hip -> L_Knee
    (12, 13), # L_Knee -> L_Ankle
]

# Colors for different body parts (RGB, 0-1 range)
BONE_COLORS = {
    # Head - yellow/orange
    (0, 1): (1.0, 0.8, 0.2),
    (0, 14): (1.0, 0.7, 0.3),
    (0, 15): (1.0, 0.7, 0.3),
    (14, 16): (1.0, 0.6, 0.3),
    (15, 17): (1.0, 0.6, 0.3),

    # Torso - blue
    (1, 2): (0.2, 0.6, 1.0),
    (1, 5): (0.2, 0.6, 1.0),
    (2, 8): (0.3, 0.5, 0.9),
    (5, 11): (0.3, 0.5, 0.9),
    (8, 11): (0.4, 0.4, 0.8),

    # Right arm - green
    (2, 3): (0.2, 0.9, 0.3),
    (3, 4): (0.3, 0.8, 0.4),

    # Left arm - cyan
    (5, 6): (0.2, 0.9, 0.8),
    (6, 7): (0.3, 0.8, 0.7),

    # Right leg - red
    (8, 9): (0.9, 0.3, 0.2),
    (9, 10): (0.8, 0.4, 0.3),

    # Left leg - magenta
    (11, 12): (0.9, 0.2, 0.7),
    (12, 13): (0.8, 0.3, 0.6),
}


class CylinderRendererCPU:
    """CPU-based cylinder renderer fallback."""

    def __init__(self, resolution: Tuple[int, int] = (512, 896)):
        self.width, self.height = resolution

    def render_skeleton(
        self,
        keypoints_3d: np.ndarray,
        canvas_size: Optional[Tuple[int, int]] = None,
        cylinder_radius: float = 0.015
    ) -> np.ndarray:
        """
        Render 3D skeleton as colored cylinders.

        Args:
            keypoints_3d: 3D keypoints, shape (num_joints, 3) in normalized coords
            canvas_size: Output image size (width, height)
            cylinder_radius: Radius of cylinder bones (normalized)

        Returns:
            RGB image as numpy array, shape (H, W, 3), uint8
        """
        if canvas_size is None:
            canvas_size = (self.width, self.height)

        width, height = canvas_size
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

        # Render each bone
        for (i, j) in SKELETON_CONNECTIONS:
            if keypoints_3d[i, 0] < 0 or keypoints_3d[j, 0] < 0:
                continue  # Skip invalid keypoints

            p1 = keypoints_3d[i]
            p2 = keypoints_3d[j]
            color = np.array(BONE_COLORS.get((i, j), (0.5, 0.5, 0.5)))

            self._render_cylinder(canvas, depth_buffer, p1, p2, color, cylinder_radius, width, height)

        # Convert to uint8
        canvas = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        return canvas

    def _render_cylinder(
        self,
        canvas: np.ndarray,
        depth_buffer: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        color: np.ndarray,
        radius: float,
        width: int,
        height: int
    ):
        """Render a single cylinder between two 3D points."""
        # Project to 2D
        x1, y1, z1 = p1[0] * width, p1[1] * height, p1[2]
        x2, y2, z2 = p2[0] * width, p2[1] * height, p2[2]

        # Draw thick line with depth-based shading
        num_steps = int(max(abs(x2 - x1), abs(y2 - y1), 1) * 2)
        pixel_radius = int(radius * min(width, height))

        for t in np.linspace(0, 1, num_steps):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            z = z1 + t * (z2 - z1)

            # Draw cylinder cross-section (circle)
            for dy in range(-pixel_radius, pixel_radius + 1):
                for dx in range(-pixel_radius, pixel_radius + 1):
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > pixel_radius * pixel_radius:
                        continue

                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        # Adjust depth based on position in cylinder
                        dist = np.sqrt(dist_sq)
                        z_offset = np.sqrt(max(0, pixel_radius * pixel_radius - dist_sq)) / (pixel_radius + 1)
                        local_z = z - z_offset * 0.1

                        if local_z < depth_buffer[py, px]:
                            depth_buffer[py, px] = local_z

                            # Shading based on distance from center
                            shade = 0.5 + 0.5 * (1 - dist / pixel_radius)
                            canvas[py, px] = color * shade


# Lazy-defined Taichi renderer class (defined after ti is imported)
_CylinderRendererTaichi = None


def _get_taichi_renderer_class():
    """Get or create the Taichi renderer class (lazy definition)."""
    global _CylinderRendererTaichi
    if _CylinderRendererTaichi is not None:
        return _CylinderRendererTaichi

    @ti.data_oriented
    class CylinderRendererTaichi:
        """GPU-accelerated cylinder renderer using Taichi."""

        def __init__(self, resolution: Tuple[int, int] = (512, 896)):
            self.width, self.height = resolution

            # Initialize Taichi (silent mode, only show errors)
            ti.init(arch=ti.gpu, offline_cache=True, log_level=ti.ERROR)

            # Allocate fields
            self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(self.height, self.width))
            self.depth_buffer = ti.field(dtype=ti.f32, shape=(self.height, self.width))

            # Pre-compute skeleton connections and colors
            self.num_bones = len(SKELETON_CONNECTIONS)
            self.bone_indices = ti.Vector.field(2, dtype=ti.i32, shape=self.num_bones)
            self.bone_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.num_bones)

            for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
                self.bone_indices[idx] = [i, j]
                color = BONE_COLORS.get((i, j), (0.5, 0.5, 0.5))
                self.bone_colors[idx] = list(color)

            # Keypoints field (max 18 keypoints)
            self.keypoints = ti.Vector.field(3, dtype=ti.f32, shape=18)

        @ti.kernel
        def _clear_buffers(self):
            """Clear canvas and depth buffer."""
            for i, j in self.canvas:
                self.canvas[i, j] = ti.Vector([0.0, 0.0, 0.0])
                self.depth_buffer[i, j] = 1e10

        @ti.kernel
        def _render_bones(self, radius: ti.f32):
            """Render all bones as cylinders."""
            for bone_idx in range(self.num_bones):
                i = self.bone_indices[bone_idx][0]
                j = self.bone_indices[bone_idx][1]

                # Skip invalid keypoints
                if self.keypoints[i][0] < 0 or self.keypoints[j][0] < 0:
                    continue

                p1 = self.keypoints[i]
                p2 = self.keypoints[j]
                color = self.bone_colors[bone_idx]

                # Project to 2D
                x1, y1, z1 = p1[0] * self.width, p1[1] * self.height, p1[2]
                x2, y2, z2 = p2[0] * self.width, p2[1] * self.height, p2[2]

                # Rasterize cylinder
                pixel_radius = ti.cast(radius * ti.min(self.width, self.height), ti.i32)
                length = ti.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                num_steps = ti.cast(ti.max(length * 2, 1.0), ti.i32)

                for step in range(num_steps):
                    t = step / ti.max(num_steps - 1.0, 1.0)
                    cx = ti.cast(x1 + t * (x2 - x1), ti.i32)
                    cy = ti.cast(y1 + t * (y2 - y1), ti.i32)
                    cz = z1 + t * (z2 - z1)

                    # Draw cylinder cross-section
                    for dy in range(-pixel_radius, pixel_radius + 1):
                        for dx in range(-pixel_radius, pixel_radius + 1):
                            dist_sq = dx * dx + dy * dy
                            if dist_sq <= pixel_radius * pixel_radius:
                                px = cx + dx
                                py = cy + dy

                                if 0 <= px < self.width and 0 <= py < self.height:
                                    dist = ti.sqrt(ti.cast(dist_sq, ti.f32))
                                    z_offset = ti.sqrt(ti.max(0.0, pixel_radius ** 2 - dist_sq)) / (pixel_radius + 1)
                                    local_z = cz - z_offset * 0.1

                                    if local_z < self.depth_buffer[py, px]:
                                        self.depth_buffer[py, px] = local_z
                                        shade = 0.5 + 0.5 * (1 - dist / pixel_radius)
                                        self.canvas[py, px] = color * shade

        def render_skeleton(
            self,
            keypoints_3d: np.ndarray,
            canvas_size: Optional[Tuple[int, int]] = None,
            cylinder_radius: float = 0.015
        ) -> np.ndarray:
            """
            Render 3D skeleton as colored cylinders.

            Args:
                keypoints_3d: 3D keypoints, shape (num_joints, 3) in normalized coords
                canvas_size: Output image size (ignored, uses init resolution)
                cylinder_radius: Radius of cylinder bones (normalized)

            Returns:
                RGB image as numpy array, shape (H, W, 3), uint8
            """
            # Copy keypoints to Taichi field
            for i in range(min(len(keypoints_3d), 18)):
                self.keypoints[i] = keypoints_3d[i].tolist()

            # Clear and render
            self._clear_buffers()
            self._render_bones(cylinder_radius)

            # Copy result
            canvas = self.canvas.to_numpy()
            canvas = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
            return canvas

    _CylinderRendererTaichi = CylinderRendererTaichi
    return _CylinderRendererTaichi


class CylinderRenderer:
    """
    Main cylinder renderer class.
    Automatically selects GPU (Taichi) or CPU implementation.
    """

    def __init__(self, resolution: Tuple[int, int] = (512, 896)):
        """
        Initialize renderer.

        Args:
            resolution: Output resolution as (width, height)
        """
        self.resolution = resolution

        if _ensure_taichi_imported():
            try:
                TaichiRenderer = _get_taichi_renderer_class()
                self._renderer = TaichiRenderer(resolution)
                self._use_gpu = True
            except Exception as e:
                print(f"[SCAIL] Failed to initialize Taichi renderer: {e}")
                self._renderer = CylinderRendererCPU(resolution)
                self._use_gpu = False
        else:
            self._renderer = CylinderRendererCPU(resolution)
            self._use_gpu = False

    def render_skeleton(
        self,
        keypoints_3d: np.ndarray,
        canvas_size: Optional[Tuple[int, int]] = None,
        cylinder_radius: float = 0.015
    ) -> np.ndarray:
        """
        Render 3D skeleton as colored cylinders.

        Args:
            keypoints_3d: 3D keypoints, shape (num_joints, 3) in normalized [0,1] coords
            canvas_size: Output image size (width, height), or None for default
            cylinder_radius: Radius of cylinder bones (normalized)

        Returns:
            RGB image as numpy array, shape (H, W, 3), uint8
        """
        result = self._renderer.render_skeleton(keypoints_3d, canvas_size, cylinder_radius)

        # Resize if canvas_size differs from rendered size (Taichi renderer uses fixed resolution)
        if canvas_size is not None:
            target_w, target_h = canvas_size
            current_h, current_w = result.shape[:2]
            if current_h != target_h or current_w != target_w:
                result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return result

    def render_sequence(
        self,
        keypoints_sequence: List[np.ndarray],
        canvas_size: Optional[Tuple[int, int]] = None,
        cylinder_radius: float = 0.015
    ) -> List[np.ndarray]:
        """
        Render a sequence of skeletons.

        Args:
            keypoints_sequence: List of 3D keypoints arrays
            canvas_size: Output image size (width, height)
            cylinder_radius: Radius of cylinder bones

        Returns:
            List of rendered RGB images
        """
        return [
            self.render_skeleton(kp, canvas_size, cylinder_radius)
            for kp in keypoints_sequence
        ]
