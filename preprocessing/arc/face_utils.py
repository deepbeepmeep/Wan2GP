# Copyright (c) 2022 Insightface Team
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
# SPDX-License-Identifier: Apache-2.0 

# Original file (insightface) was released under MIT License:

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cv2
from PIL import Image

def estimate_norm(lmk, image_size=112, arcface_dst=None):
    from skimage import transform as trans
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def get_arcface_dst(extend_face_crop=False, extend_ratio=0.8):
    arcface_dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    if extend_face_crop:
        arcface_dst[:,1] = arcface_dst[:,1] + 10
        arcface_dst = (arcface_dst - 112/2) * extend_ratio + 112/2
    return arcface_dst

def align_face(image_pil, face_kpts, extend_face_crop=False, extend_ratio=0.8, face_size=112):
    arcface_dst = get_arcface_dst(extend_face_crop, extend_ratio)
    M = estimate_norm(face_kpts, face_size, arcface_dst)
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    face_image_cv2 = cv2.warpAffine(image_cv2, M, (face_size, face_size), borderValue=0.0)
    face_image = Image.fromarray(cv2.cvtColor(face_image_cv2, cv2.COLOR_BGR2RGB))
    return face_image