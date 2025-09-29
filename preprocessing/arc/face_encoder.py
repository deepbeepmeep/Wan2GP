# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms as T

import numpy as np

from insightface.utils import face_align
from insightface.app import FaceAnalysis
from facexlib.recognition import init_recognition_model


__all__ = [
    "FaceEncoderArcFace",
    "get_landmarks_from_image",
]


detector = None

def get_landmarks_from_image(image):
    """
    Detect landmarks with insightface.

    Args:
        image (np.ndarray or PIL.Image):
            The input image in RGB format.

    Returns:
        5 2D keypoints, only one face will be returned.
    """
    global detector
    if detector is None:
        detector = FaceAnalysis()
        detector.prepare(ctx_id=0, det_size=(640, 640))

    in_image = np.array(image).copy()
    
    faces = detector.get(in_image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Get the largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    
    # Return the 5 keypoints directly
    keypoints = face.kps  # 5 x 2

    return keypoints


class FaceEncoderArcFace():
    """ Official ArcFace, no_grad-only """

    def __repr__(self):
        return "ArcFace"


    def init_encoder_model(self, device, eval_mode=True):
        self.device = device
        self.encoder_model = init_recognition_model('arcface', device=device)

        if eval_mode:
            self.encoder_model.eval()


    @torch.no_grad()
    def input_preprocessing(self, in_image, landmarks, image_size=112):
        assert landmarks is not None, "landmarks are not provided!"

        in_image = np.array(in_image)
        landmark = np.array(landmarks)

        face_aligned = face_align.norm_crop(in_image, landmark=landmark, image_size=image_size)

        image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        face_aligned = image_transform(face_aligned).unsqueeze(0).to(self.device)

        return face_aligned


    @torch.no_grad()
    def __call__(self, in_image, need_proc=False, landmarks=None, image_size=112):

        if need_proc:
            in_image = self.input_preprocessing(in_image, landmarks, image_size)
        else:
            assert isinstance(in_image, torch.Tensor)

        in_image = in_image[:, [2, 1, 0], :, :].contiguous()
        image_embeds = self.encoder_model(in_image) # [B, 512], normalized

        return image_embeds