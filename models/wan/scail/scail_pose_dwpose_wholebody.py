# https://github.com/IDEA-Research/DWPose

import numpy as np
import onnxruntime as ort

from shared.utils import files_locator as fl

from .scail_pose_dwpose_onnxdet import inference_detector
from .scail_pose_dwpose_onnxpose import inference_pose, inference_pose_batch


class Wholebody:
    def __init__(self, device, use_batch=False):
        providers = [("CUDAExecutionProvider", {
            "device_id": device
        })]
        # providers = [("CPUExecutionProvider", {})]
        onnx_det = fl.locate_file("pose/yolox_l.onnx")
        onnx_pose = fl.locate_file("pose/dw-ll_ucoco_384.onnx")

        self.session_det = ort.InferenceSession(
            path_or_bytes=onnx_det, providers=providers
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=onnx_pose, providers=providers
        )
        self.use_batch = use_batch

    def _get_result_from_det_pose(self, det_result, keypoints, scores):
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)    # (1, 133, 3)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)   # (1, 3)，对第五第六个点做平均
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)   # 从第二个开始切片，这里维度为3，只切一片
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)    # 在17索引处插入neck
        # 调换骨骼索引
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]     # openpose需要检测17点+1脖子关键点
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores, det_result


    def __call__(self, oriImg):
        if not self.use_batch:
            det_result = inference_detector(self.session_det, oriImg)
            keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)   # keypoints: (n_bbox, 133, 2) scores: (n_bbox, 133)  不管输入是什么初步提取的关键点数量都是一致的
            return self._get_result_from_det_pose(det_result=det_result, keypoints=keypoints, scores=scores)
            
        else:
            raise NotImplementedError("DWposeDetector does not support batch mode")
