# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import cv2
import torch
import numpy as np
from . import util
from .wholebody import Wholebody, HWC3, resize_image
from PIL import Image
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class OptimizedWholebody:
    """Optimized version of Wholebody for faster serial processing"""
    def __init__(self, onnx_det, onnx_pose, device='cuda:0'):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
        self.device = device
        
        # Pre-allocate session options for better performance
        self.session_det.set_providers(providers)
        self.session_pose.set_providers(providers)
        
        # Get input names once to avoid repeated lookups
        self.det_input_name = self.session_det.get_inputs()[0].name
        self.pose_input_name = self.session_pose.get_inputs()[0].name
        self.pose_output_names = [out.name for out in self.session_pose.get_outputs()]
    
    def __call__(self, ori_img):
        from .onnxdet import inference_detector
        from .onnxpose import inference_pose
        
        det_result = inference_detector(self.session_det, ori_img)
        keypoints, scores = inference_pose(self.session_pose, det_result, ori_img)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores, det_result


class PoseAnnotator:
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pose_estimation = Wholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        image = convert_to_numpy(image)
        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2])

    def process(self, ori_img, ori_shape):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            
            if len(candidate) == 0:
                # No detections - return empty results
                empty_ret_data = {}
                if self.use_body:
                    empty_ret_data["detected_map_body"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_face:
                    empty_ret_data["detected_map_face"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_body and self.use_face:
                    empty_ret_data["detected_map_bodyface"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_hand and self.use_body and self.use_face:
                    empty_ret_data["detected_map_handbodyface"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                return empty_ret_data, np.array([])
            
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]
            faces = candidate[:, 24:92]
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(detected_map_body[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_body"] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(detected_map_face[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose, H, W, use_body=True, use_face=True)
                detected_map_bodyface = cv2.resize(detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                                                   interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(pose, H, W, use_hand=True, use_body=True, use_face=True)
                detected_map_handbodyface = cv2.resize(detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
                                                       interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            return ret_data, det_result


class OptimizedPoseAnnotator(PoseAnnotator):
    """Optimized version using improved Wholebody class"""
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pose_estimation = OptimizedWholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)


class PoseBodyFaceAnnotator(PoseAnnotator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_body, self.use_face, self.use_hand = True, True, False
    
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_bodyface']


class OptimizedPoseBodyFaceVideoAnnotator:
    """Optimized video annotator with multiple optimization strategies"""
    def __init__(self, cfg, num_workers=2, chunk_size=8):
        self.cfg = cfg
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.use_body, self.use_face, self.use_hand = True, True, True
        
        # Initialize one annotator per worker to avoid ONNX session conflicts
        self.annotators = []
        for _ in range(num_workers):
            annotator = OptimizedPoseAnnotator(cfg)
            annotator.use_body, annotator.use_face, annotator.use_hand = True, True, True
            self.annotators.append(annotator)
        
        self._current_worker = 0
        self._worker_lock = threading.Lock()
    
    def _get_annotator(self):
        """Get next available annotator in round-robin fashion"""
        with self._worker_lock:
            annotator = self.annotators[self._current_worker]
            self._current_worker = (self._current_worker + 1) % len(self.annotators)
            return annotator
    
    def _process_single_frame(self, frame_data):
        """Process a single frame with error handling"""
        frame, frame_idx = frame_data
        try:
            annotator = self._get_annotator()
            
            # Convert frame
            frame = convert_to_numpy(frame)
            input_image = HWC3(frame[..., ::-1])
            resized_image = resize_image(input_image, annotator.resize_size)
            
            # Process
            ret_data, _ = annotator.process(resized_image, frame.shape[:2])
            
            if 'detected_map_handbodyface' in ret_data:
                return frame_idx, ret_data['detected_map_handbodyface']
            else:
                # Create empty frame if no detection
                h, w = frame.shape[:2]
                return frame_idx, np.zeros((h, w, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Return empty frame on error
            h, w = frame.shape[:2] if hasattr(frame, 'shape') else (480, 640)
            return frame_idx, np.zeros((h, w, 3), dtype=np.uint8)
    
    def forward(self, frames):
        """Process video frames with optimizations"""
        if len(frames) == 0:
            return []
        
        # For small number of frames, use serial processing to avoid threading overhead
        if len(frames) <= 4:
            annotator = self.annotators[0]
            ret_frames = []
            for frame in frames:
                frame = convert_to_numpy(frame)
                input_image = HWC3(frame[..., ::-1])
                resized_image = resize_image(input_image, annotator.resize_size)
                ret_data, _ = annotator.process(resized_image, frame.shape[:2])
                
                if 'detected_map_handbodyface' in ret_data:
                    ret_frames.append(ret_data['detected_map_handbodyface'])
                else:
                    h, w = frame.shape[:2]
                    ret_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            return ret_frames
        
        # For larger videos, use parallel processing
        frame_data = [(frame, idx) for idx, frame in enumerate(frames)]
        results = [None] * len(frames)
        
        # Process in chunks to manage memory
        for chunk_start in range(0, len(frame_data), self.chunk_size * self.num_workers):
            chunk_end = min(chunk_start + self.chunk_size * self.num_workers, len(frame_data))
            chunk_data = frame_data[chunk_start:chunk_end]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                chunk_results = list(executor.map(self._process_single_frame, chunk_data))
            
            # Store results in correct order
            for frame_idx, result in chunk_results:
                results[frame_idx] = result
        
        return results


class OptimizedPoseBodyFaceHandVideoAnnotator:
    """Optimized video annotator that includes hands, body, and face"""
    def __init__(self, cfg, num_workers=2, chunk_size=8):
        self.cfg = cfg
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.use_body, self.use_face, self.use_hand = True, True, True  # Enable hands
        
        # Initialize one annotator per worker to avoid ONNX session conflicts
        self.annotators = []
        for _ in range(num_workers):
            annotator = OptimizedPoseAnnotator(cfg)
            annotator.use_body, annotator.use_face, annotator.use_hand = True, True, True
            self.annotators.append(annotator)
        
        self._current_worker = 0
        self._worker_lock = threading.Lock()
    
    def _get_annotator(self):
        """Get next available annotator in round-robin fashion"""
        with self._worker_lock:
            annotator = self.annotators[self._current_worker]
            self._current_worker = (self._current_worker + 1) % len(self.annotators)
            return annotator
    
    def _process_single_frame(self, frame_data):
        """Process a single frame with error handling"""
        frame, frame_idx = frame_data
        try:
            annotator = self._get_annotator()
            
            # Convert frame
            frame = convert_to_numpy(frame)
            input_image = HWC3(frame[..., ::-1])
            resized_image = resize_image(input_image, annotator.resize_size)
            
            # Process
            ret_data, _ = annotator.process(resized_image, frame.shape[:2])
            
            if 'detected_map_handbodyface' in ret_data:
                return frame_idx, ret_data['detected_map_handbodyface']
            else:
                # Create empty frame if no detection
                h, w = frame.shape[:2]
                return frame_idx, np.zeros((h, w, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Return empty frame on error
            h, w = frame.shape[:2] if hasattr(frame, 'shape') else (480, 640)
            return frame_idx, np.zeros((h, w, 3), dtype=np.uint8)
    
    def forward(self, frames):
        """Process video frames with optimizations"""
        if len(frames) == 0:
            return []
        
        # For small number of frames, use serial processing to avoid threading overhead
        if len(frames) <= 4:
            annotator = self.annotators[0]
            ret_frames = []
            for frame in frames:
                frame = convert_to_numpy(frame)
                input_image = HWC3(frame[..., ::-1])
                resized_image = resize_image(input_image, annotator.resize_size)
                ret_data, _ = annotator.process(resized_image, frame.shape[:2])
                
                if 'detected_map_handbodyface' in ret_data:
                    ret_frames.append(ret_data['detected_map_handbodyface'])
                else:
                    h, w = frame.shape[:2]
                    ret_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            return ret_frames
        
        # For larger videos, use parallel processing
        frame_data = [(frame, idx) for idx, frame in enumerate(frames)]
        results = [None] * len(frames)
        
        # Process in chunks to manage memory
        for chunk_start in range(0, len(frame_data), self.chunk_size * self.num_workers):
            chunk_end = min(chunk_start + self.chunk_size * self.num_workers, len(frame_data))
            chunk_data = frame_data[chunk_start:chunk_end]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                chunk_results = list(executor.map(self._process_single_frame, chunk_data))
            
            # Store results in correct order
            for frame_idx, result in chunk_results:
                results[frame_idx] = result
        
        return results


# Choose which version you want to use:

# Option 1: Body + Face only (original behavior)
class PoseBodyFaceVideoAnnotator(OptimizedPoseBodyFaceVideoAnnotator):
    """Backward compatible class name - Body and Face only"""
# Option 2: Body + Face + Hands (if you want hands)
class PoseBodyFaceHandVideoAnnotator(OptimizedPoseBodyFaceHandVideoAnnotator):
    """Video annotator with hands, body, and face"""
    def __init__(self, cfg):
        super().__init__(cfg, num_workers=2, chunk_size=4)


# Keep the existing utility functions
import imageio

def save_one_video(file_path, videos, fps=8, quality=8, macro_block_size=None):
    try:
        video_writer = imageio.get_writer(file_path, fps=fps, codec='libx264', quality=quality, macro_block_size=macro_block_size)
        for frame in videos:
            video_writer.append_data(frame)
        video_writer.close()
        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False
    
def get_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return frames, fps