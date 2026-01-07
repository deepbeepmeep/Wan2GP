"""
SCAIL Pose Aligner - Align 3D pose to reference image proportions.
Adapted from steadydancer/pose_align.py for 3D keypoints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def _safe_ratio(num: float, den: float) -> float:
    """Safe division with fallback to 1.0."""
    if den == 0 or not np.isfinite(den):
        return 1.0
    val = num / den
    return float(val) if np.isfinite(val) else 1.0


def _nan_to_one(val: float) -> float:
    """Replace NaN with 1.0."""
    return 1.0 if not np.isfinite(val) else float(val)


def _dist_3d(keypoints: np.ndarray, a: int, b: int) -> float:
    """Compute 3D distance between two keypoints."""
    pa, pb = keypoints[a, :3], keypoints[b, :3]
    # Validity: camera-Z must be positive; X/Y can be negative in camera space.
    if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
        return np.nan
    if pa[2] <= 0 or pb[2] <= 0:
        return np.nan
    return float(np.linalg.norm(pa - pb))


class ScailPoseAligner:
    """
    Align extracted 3D pose keypoints to reference image body proportions.

    This adapts the steadydancer alignment approach for 3D keypoints,
    computing scale factors for different body parts and applying them
    to match the reference image's body proportions.
    """

    # Body keypoint indices (COCO format)
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    R_HIP = 8
    R_KNEE = 9
    R_ANKLE = 10
    L_HIP = 11
    L_KNEE = 12
    L_ANKLE = 13
    R_EYE = 14
    L_EYE = 15
    R_EAR = 16
    L_EAR = 17

    def __init__(self):
        pass

    def compute_alignment_params(
        self,
        ref_keypoints: np.ndarray,
        driving_keypoints: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute alignment parameters from reference and first driving frame.

        Args:
            ref_keypoints: Reference image 3D keypoints, shape (num_joints, 3)
            driving_keypoints: First frame driving video 3D keypoints, shape (num_joints, 3)

        Returns:
            Dictionary of scale factors for different body parts
        """
        body_ref = ref_keypoints.copy()
        body_drv = driving_keypoints.copy()

        # Compute scale factors for different body parts
        align_params = {
            "scale_neck": _safe_ratio(
                _dist_3d(body_ref, self.NOSE, self.NECK),
                _dist_3d(body_drv, self.NOSE, self.NECK)
            ),
            "scale_face_left": _safe_ratio(
                _dist_3d(body_ref, self.L_EAR, self.L_EYE) + _dist_3d(body_ref, self.L_EYE, self.NOSE),
                _dist_3d(body_drv, self.L_EAR, self.L_EYE) + _dist_3d(body_drv, self.L_EYE, self.NOSE)
            ),
            "scale_face_right": _safe_ratio(
                _dist_3d(body_ref, self.R_EAR, self.R_EYE) + _dist_3d(body_ref, self.R_EYE, self.NOSE),
                _dist_3d(body_drv, self.R_EAR, self.R_EYE) + _dist_3d(body_drv, self.R_EYE, self.NOSE)
            ),
            "scale_shoulder": _safe_ratio(
                _dist_3d(body_ref, self.R_SHOULDER, self.L_SHOULDER),
                _dist_3d(body_drv, self.R_SHOULDER, self.L_SHOULDER)
            ),
            "scale_arm_upper": np.nanmean([
                _safe_ratio(_dist_3d(body_ref, self.R_SHOULDER, self.R_ELBOW),
                           _dist_3d(body_drv, self.R_SHOULDER, self.R_ELBOW)),
                _safe_ratio(_dist_3d(body_ref, self.L_SHOULDER, self.L_ELBOW),
                           _dist_3d(body_drv, self.L_SHOULDER, self.L_ELBOW))
            ]),
            "scale_arm_lower": np.nanmean([
                _safe_ratio(_dist_3d(body_ref, self.R_ELBOW, self.R_WRIST),
                           _dist_3d(body_drv, self.R_ELBOW, self.R_WRIST)),
                _safe_ratio(_dist_3d(body_ref, self.L_ELBOW, self.L_WRIST),
                           _dist_3d(body_drv, self.L_ELBOW, self.L_WRIST))
            ]),
            "scale_body_len": _safe_ratio(
                _dist_3d(body_ref, self.NECK, self.R_HIP) if not np.isnan(_dist_3d(body_ref, self.NECK, self.R_HIP))
                else _dist_3d(body_ref, self.NECK, self.L_HIP),
                _dist_3d(body_drv, self.NECK, self.R_HIP) if not np.isnan(_dist_3d(body_drv, self.NECK, self.R_HIP))
                else _dist_3d(body_drv, self.NECK, self.L_HIP)
            ),
            "scale_leg_upper": np.nanmean([
                _safe_ratio(_dist_3d(body_ref, self.R_HIP, self.R_KNEE),
                           _dist_3d(body_drv, self.R_HIP, self.R_KNEE)),
                _safe_ratio(_dist_3d(body_ref, self.L_HIP, self.L_KNEE),
                           _dist_3d(body_drv, self.L_HIP, self.L_KNEE))
            ]),
            "scale_leg_lower": np.nanmean([
                _safe_ratio(_dist_3d(body_ref, self.R_KNEE, self.R_ANKLE),
                           _dist_3d(body_drv, self.R_KNEE, self.R_ANKLE)),
                _safe_ratio(_dist_3d(body_ref, self.L_KNEE, self.L_ANKLE),
                           _dist_3d(body_drv, self.L_KNEE, self.L_ANKLE))
            ]),
        }

        # Replace NaN values with mean of other scales
        finite_vals = [v for v in align_params.values() if np.isfinite(v)]
        mean_scale = np.mean(finite_vals) if finite_vals else 1.0
        align_params = {k: mean_scale if not np.isfinite(v) else v for k, v in align_params.items()}

        # Compute spatial offset (translate to match reference position)
        if body_ref[self.NECK, 2] > 0 and body_drv[self.NECK, 2] > 0:
            offset = body_ref[self.NECK, :3] - body_drv[self.NECK, :3]
        else:
            offset = np.zeros(3)
        align_params["offset"] = offset

        return align_params

    def apply_alignment(
        self,
        keypoints: np.ndarray,
        align_params: Dict[str, float],
        center_idx: int = 1  # NECK
    ) -> np.ndarray:
        """
        Apply alignment parameters to transform keypoints.

        Args:
            keypoints: 3D keypoints to transform, shape (num_joints, 3)
            align_params: Dictionary of scale factors from compute_alignment_params
            center_idx: Index of center keypoint for scaling (default: NECK)

        Returns:
            Transformed keypoints, shape (num_joints, 3)
        """
        result = keypoints.copy()
        center = result[center_idx, :3].copy()
        if not np.all(np.isfinite(center)) or center[2] <= 0:
            # No reliable center; apply only translation to valid keypoints.
            offset = align_params.get("offset", np.zeros(3))
            valid_mask = np.isfinite(result[:, 2]) & (result[:, 2] > 0)
            result[valid_mask, :3] += offset
            return result

        # Scale different body parts from their respective parent joints
        # Head/face region
        self._scale_from_center(result, [self.NOSE], center, align_params["scale_neck"])
        self._scale_from_center(result, [self.L_EYE, self.L_EAR], result[self.NOSE, :3], align_params["scale_face_left"])
        self._scale_from_center(result, [self.R_EYE, self.R_EAR], result[self.NOSE, :3], align_params["scale_face_right"])

        # Shoulders
        self._scale_from_center(result, [self.R_SHOULDER, self.L_SHOULDER], center, align_params["scale_shoulder"])

        # Arms
        self._scale_from_center(result, [self.R_ELBOW], result[self.R_SHOULDER, :3], align_params["scale_arm_upper"])
        self._scale_from_center(result, [self.R_WRIST], result[self.R_ELBOW, :3], align_params["scale_arm_lower"])
        self._scale_from_center(result, [self.L_ELBOW], result[self.L_SHOULDER, :3], align_params["scale_arm_upper"])
        self._scale_from_center(result, [self.L_WRIST], result[self.L_ELBOW, :3], align_params["scale_arm_lower"])

        # Torso/hips
        self._scale_from_center(result, [self.R_HIP, self.L_HIP], center, align_params["scale_body_len"])

        # Legs
        self._scale_from_center(result, [self.R_KNEE], result[self.R_HIP, :3], align_params["scale_leg_upper"])
        self._scale_from_center(result, [self.R_ANKLE], result[self.R_KNEE, :3], align_params["scale_leg_lower"])
        self._scale_from_center(result, [self.L_KNEE], result[self.L_HIP, :3], align_params["scale_leg_upper"])
        self._scale_from_center(result, [self.L_ANKLE], result[self.L_KNEE, :3], align_params["scale_leg_lower"])

        # Apply offset
        offset = align_params.get("offset", np.zeros(3))
        valid_mask = np.isfinite(result[:, 2]) & (result[:, 2] > 0)
        result[valid_mask, :3] += offset

        return result

    def _scale_from_center(
        self,
        keypoints: np.ndarray,
        indices: List[int],
        center: np.ndarray,
        scale: float
    ):
        """Scale keypoints relative to a center point."""
        for idx in indices:
            if np.isfinite(keypoints[idx, 2]) and keypoints[idx, 2] > 0:  # Valid keypoint (camera-Z)
                diff = keypoints[idx, :3] - center
                keypoints[idx, :3] = center + diff * scale

    def align_sequence(
        self,
        ref_keypoints: np.ndarray,
        driving_keypoints_seq: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Align a sequence of driving keypoints to match reference proportions.

        Args:
            ref_keypoints: Reference image 3D keypoints, shape (num_joints, 3)
            driving_keypoints_seq: List of driving video keypoints, each shape (num_joints, 3)

        Returns:
            List of aligned keypoints
        """
        if not driving_keypoints_seq:
            return []

        # Compute alignment from first frame
        align_params = self.compute_alignment_params(ref_keypoints, driving_keypoints_seq[0])

        # Apply to all frames
        aligned_seq = []
        for kp in driving_keypoints_seq:
            aligned_kp = self.apply_alignment(kp, align_params)
            aligned_seq.append(aligned_kp)

        return aligned_seq
