import logging
from typing import Any, Mapping

import torch
from torch import nn

from .motionformer import MotionFormer


class Synchformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.vfeat_extractor = MotionFormer(extract_features=True,
                                            factorize_space_time=True,
                                            agg_space_module='TransformerEncoderLayer',
                                            agg_time_module='torch.nn.Identity',
                                            add_global_repr=False)

        # self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        # self.afeat_extractor = instantiate_from_config(afeat_extractor)
        # # bridging the s3d latent dim (1024) into what is specified in the config
        # # to match e.g. the transformer dim
        # self.vproj = instantiate_from_config(vproj)
        # self.aproj = instantiate_from_config(aproj)
        # self.transformer = instantiate_from_config(transformer)

    def forward(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis = self.vfeat_extractor(vis)
        return vis

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        # discard all entries except vfeat_extractor
        sd = {k: v for k, v in sd.items() if k.startswith('vfeat_extractor')}

        return super().load_state_dict(sd, strict)


if __name__ == "__main__":
    model = Synchformer().cuda().eval()
    sd = torch.load('./ext_weights/synchformer_state_dict.pth', weights_only=True)
    model.load_state_dict(sd)

    vid = torch.randn(2, 7, 16, 3, 224, 224).cuda()
    features = model.extract_vfeats(vid, for_loop=False).detach().cpu()
    print(features.shape)

    # extract and save the state dict only
    # sd = torch.load('./ext_weights/sync_model_audioset.pt')['model']
    # torch.save(sd, './ext_weights/synchformer_state_dict.pth')
