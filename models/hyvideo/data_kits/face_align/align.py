import os
import sys
import torch
from .detface import DetFace

class AlignImage(object):
    def __init__(self, device='cuda', det_path=''):
        self.facedet = DetFace(pt_path=det_path, confThreshold=0.5, nmsThreshold=0.45, device=device)

    @torch.no_grad()
    def __call__(self, im, maxface=False):
        bboxes, kpss, scores = self.facedet.detect(im)
        face_num = bboxes.shape[0]

        five_pts_list = []
        scores_list = []
        bboxes_list = []
        for i in range(face_num):
            five_pts_list.append(kpss[i].reshape(5,2))
            scores_list.append(scores[i])
            bboxes_list.append(bboxes[i])

        if maxface and face_num>1:
            max_idx = 0
            max_area = (bboxes[0, 2])*(bboxes[0, 3])
            for i in range(1, face_num):
                area = (bboxes[i,2])*(bboxes[i,3])
                if area>max_area:
                    max_idx = i
            five_pts_list = [five_pts_list[max_idx]]
            scores_list = [scores_list[max_idx]]
            bboxes_list = [bboxes_list[max_idx]]

        return five_pts_list, scores_list, bboxes_list