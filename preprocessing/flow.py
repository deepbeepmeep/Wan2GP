# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import numpy as np
import argparse
from PIL import Image

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

class FlowAnnotator:
    def __init__(self, cfg, device=None):
        from .raft.raft import RAFT
        from .raft.utils.utils import InputPadder
        from .raft.utils import flow_viz

        params = {
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        }
        params = argparse.Namespace(**params)
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = RAFT(params)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pretrained_model, map_location="cpu", weights_only=True).items()})
        self.model = self.model.to(self.device).eval()
        self.InputPadder = InputPadder
        self.flow_viz = flow_viz

    def forward(self, frames):
        # frames / RGB
        frames = [torch.from_numpy(convert_to_numpy(frame).astype(np.uint8)).permute(2, 0, 1).float()[None].to(self.device) for frame in frames]
        flow_up_list, flow_up_vis_list = [], []
        with torch.no_grad():
            for i, (image1, image2) in enumerate(zip(frames[:-1], frames[1:])):
                padder = self.InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flow_up_vis = self.flow_viz.flow_to_image(flow_up)
                flow_up_list.append(flow_up)
                flow_up_vis_list.append(flow_up_vis)
        return flow_up_list, flow_up_vis_list  # RGB


class FlowVisAnnotator(FlowAnnotator):
    def forward(self, frames):
        flow_up_list, flow_up_vis_list = super().forward(frames)
        return flow_up_vis_list[:1] + flow_up_vis_list