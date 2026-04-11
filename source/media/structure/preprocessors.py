"""Structure video preprocessors (flow, canny, depth, pose, raw, uni3c)."""

import numpy as np
from pathlib import Path
from typing import List

from source.core.log import generation_logger
from source.runtime.wgp_bridge import (
    get_canny_video_annotator_class,
    get_depth_v2_video_annotator_class,
    get_flow_annotator_class,
    get_flow_viz_module,
    get_pose_body_face_video_annotator_class,
)

__all__ = [
    "get_structure_preprocessor",
    "process_structure_frames",
]


def get_structure_preprocessor(
    structure_type: str,
    motion_strength: float = 1.0,
    canny_intensity: float = 1.0,
    depth_contrast: float = 1.0,
):
    """
    Get preprocessor function for structure video guidance.

    We import and instantiate the preprocessor ourselves, then run it
    on the structure video frames to generate RGB visualizations.

    Args:
        structure_type: Type of preprocessing ("flow", "canny", or "depth")
        motion_strength: Only affects flow - scales flow vector magnitude
        canny_intensity: Only affects canny - scales edge boldness
        depth_contrast: Only affects depth - adjusts depth map contrast

    Returns:
        Function that takes a list of frames (np.ndarray)
        and returns a list of processed frames (RGB visualizations).
    """
    generation_logger.debug_anomaly("PREPROCESSOR_DEBUG", f"Initializing preprocessor with structure_type='{structure_type}'")

    if structure_type == "flow":
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
        FlowAnnotator = get_flow_annotator_class()

        # Ensure RAFT model is downloaded
        flow_model_path = wan_dir / "ckpts" / "flow" / "raft-things.pth"
        if not flow_model_path.exists():
            generation_logger.debug_anomaly("FLOW", f"RAFT model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                flow_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename="raft-things.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="flow"
                )
                generation_logger.debug_anomaly("FLOW", f"RAFT model downloaded successfully")
            except (OSError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to download RAFT model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/flow") from e

        cfg = {"PRETRAINED_MODEL": str(flow_model_path)}
        annotator = FlowAnnotator(cfg)

        flow_viz = get_flow_viz_module()

        def process_with_motion_strength(frames):
            """Process frames with motion_strength applied to flow visualizations."""
            # Get raw flow fields from RAFT
            flow_fields, _ = annotator.forward(frames)

            # Scale flow fields by motion_strength
            scaled_flows = [flow * motion_strength for flow in flow_fields]

            # Generate visualizations from scaled flows
            flow_visualizations = [flow_viz.flow_to_image(flow) for flow in scaled_flows]

            # Match FlowVisAnnotator behavior: duplicate first frame
            return flow_visualizations[:1] + flow_visualizations

        if abs(motion_strength - 1.0) > 1e-6:
            generation_logger.debug_anomaly("FLOW", f"Applying motion_strength={motion_strength} to flow visualizations")

        return process_with_motion_strength

    elif structure_type == "canny":
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
        CannyVideoAnnotator = get_canny_video_annotator_class()

        # Ensure scribble/canny model is downloaded
        canny_model_path = wan_dir / "ckpts" / "scribble" / "netG_A_latest.pth"
        if not canny_model_path.exists():
            generation_logger.debug_anomaly("CANNY", f"Scribble model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                canny_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename="netG_A_latest.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="scribble"
                )
                generation_logger.debug_anomaly("CANNY", f"Scribble model downloaded successfully")
            except (OSError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to download Scribble model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/scribble") from e

        cfg = {"PRETRAINED_MODEL": str(canny_model_path)}
        annotator = CannyVideoAnnotator(cfg)

        def process_canny(frames):
            # Get base canny edges
            edge_frames = annotator.forward(frames)

            # Apply intensity adjustment if not 1.0
            if abs(canny_intensity - 1.0) > 1e-6:
                adjusted_frames = []
                for frame in edge_frames:
                    # Scale pixel values by intensity factor
                    adjusted = (frame.astype(np.float32) * canny_intensity).clip(0, 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Applied canny intensity: {canny_intensity}")
                return adjusted_frames
            return edge_frames

        return process_canny

    elif structure_type == "depth":
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
        DepthV2VideoAnnotator = get_depth_v2_video_annotator_class()

        variant = "vitl"  # Could be configurable

        # Ensure depth model is downloaded
        depth_model_path = wan_dir / "ckpts" / "depth" / f"depth_anything_v2_{variant}.pth"
        if not depth_model_path.exists():
            generation_logger.debug_anomaly("DEPTH", f"Depth Anything V2 {variant} model not found, downloading from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                depth_model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id="DeepBeepMeep/Wan2.1",
                    filename=f"depth_anything_v2_{variant}.pth",
                    local_dir=str(wan_dir / 'ckpts'),
                    subfolder="depth"
                )
                generation_logger.debug_anomaly("DEPTH", f"Depth Anything V2 {variant} model downloaded successfully")
            except (OSError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to download Depth Anything V2 model: {e}. Please manually download from https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/depth") from e

        cfg = {
            "PRETRAINED_MODEL": str(depth_model_path),
            "MODEL_VARIANT": variant
        }
        annotator = DepthV2VideoAnnotator(cfg)

        def process_depth(frames):
            # Get base depth maps
            depth_frames = annotator.forward(frames)

            # Apply contrast adjustment if not 1.0
            if abs(depth_contrast - 1.0) > 1e-6:
                adjusted_frames = []
                for frame in depth_frames:
                    # Convert to float, normalize, apply contrast, denormalize
                    frame_float = frame.astype(np.float32) / 255.0
                    # Apply contrast around midpoint (0.5)
                    adjusted = ((frame_float - 0.5) * depth_contrast + 0.5).clip(0, 1)
                    adjusted = (adjusted * 255).astype(np.uint8)
                    adjusted_frames.append(adjusted)
                generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Applied depth contrast: {depth_contrast}")
                return adjusted_frames
            return depth_frames

        return process_depth

    elif structure_type == "pose":
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
        PoseBodyFaceVideoAnnotator = get_pose_body_face_video_annotator_class()

        det_model_path = wan_dir / "ckpts" / "pose" / "yolox_l.onnx"
        pose_model_path = wan_dir / "ckpts" / "pose" / "dw-ll_ucoco_384.onnx"

        missing_models = [
            str(model_path)
            for model_path in (det_model_path, pose_model_path)
            if not model_path.exists()
        ]
        if missing_models:
            raise RuntimeError(
                "Missing DWPose models required for pose preprocessing: "
                + ", ".join(missing_models)
            )

        cfg = {
            "DETECTION_MODEL": str(det_model_path),
            "POSE_MODEL": str(pose_model_path),
            "RESIZE_SIZE": 1024,
        }
        annotator = PoseBodyFaceVideoAnnotator(cfg)

        def process_pose(frames):
            return annotator.forward(frames)

        return process_pose

    elif structure_type in ("raw", "uni3c"):
        # Raw/uni3c use raw video frames as guidance - no preprocessing needed
        # For uni3c, frames are passed directly to WGP's uni3c encoder
        generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"{structure_type} type: returning frames without preprocessing")
        return lambda frames: frames

    else:
        raise ValueError(
            f"Unsupported structure_type: {structure_type}. Must be 'flow', 'canny', "
            "'depth', 'pose', 'raw', or 'uni3c'"
        )


def process_structure_frames(
    frames: List[np.ndarray],
    structure_type: str,
    motion_strength: float,
    canny_intensity: float,
    depth_contrast: float,
) -> List[np.ndarray]:
    """
    Process frames with chosen preprocessor, ensuring consistent output count.

    Handles the N-1 problem for optical flow (which returns N-1 flows for N frames).

    Args:
        frames: List of input frames to preprocess
        structure_type: Type of preprocessing ("flow", "canny", "depth", "pose", "raw", or "uni3c")
        motion_strength: Strength parameter for flow
        canny_intensity: Intensity parameter for canny
        depth_contrast: Contrast parameter for depth

    Returns:
        List of RGB visualization frames (length = len(frames))
    """
    # Handle raw type - no preprocessing needed
    if structure_type == "raw":
        generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Raw type: returning {len(frames)} frames without preprocessing")
        return frames

    generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Processing {len(frames)} frames with '{structure_type}' preprocessor...")

    preprocessor = get_structure_preprocessor(
        structure_type,
        motion_strength,
        canny_intensity,
        depth_contrast,
    )

    import time
    start_time = time.time()
    processed_frames = preprocessor(frames)
    duration = time.time() - start_time

    generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Preprocessing completed in {duration:.2f}s")

    # Handle N-1 case for optical flow
    if structure_type == "flow" and len(processed_frames) == len(frames) - 1:
        # Duplicate last flow frame to match input count
        processed_frames.append(processed_frames[-1].copy())
        generation_logger.debug_anomaly("STRUCTURE_PREPROCESS", f"Duplicated last flow frame ({len(frames)-1} \u2192 {len(frames)} frames)")

    # Validate output count
    if len(processed_frames) != len(frames):
        raise ValueError(
            f"Preprocessor '{structure_type}' returned {len(processed_frames)} frames "
            f"for {len(frames)} input frames. Expected {len(frames)} output frames."
        )

    return processed_frames
