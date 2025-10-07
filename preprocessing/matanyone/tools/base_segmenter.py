import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from .mask_painter import mask_painter
from shared.utils import files_locator as fl 


class BaseSegmenter:
    def __init__(self, SAM_checkpoint, model_type, device='cuda:0'):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        self.device = device
        # SAM_checkpoint = None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        from accelerate import init_empty_weights

        # self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        with init_empty_weights():
            self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        from mmgp import offload
        # self.model.to(torch.float16)
        # offload.save_model(self.model, "ckpts/mask/sam_vit_h_4b8939_fp16.safetensors")
        
        offload.load_model_data(self.model, fl.locate_file("mask/sam_vit_h_4b8939_fp16.safetensors"))
        self.model.to(torch.float32) # need to be optimized, if not f32 crappy precision
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return
    
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False

    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if mode == 'point':
                masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                    point_labels=prompts['point_labels'], 
                                    multimask_output=multimask)
            elif mode == 'mask':
                masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'], 
                                    multimask_output=multimask)
            elif mode == 'both':   # both
                masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                    point_labels=prompts['point_labels'], 
                                    mask_input=prompts['mask_input'], 
                                    multimask_output=multimask)
            else:
                raise("Not implement now!")
            # masks (n, h, w), scores (n,), logits (n, 256, 256)
            return masks, scores, logits


if __name__ == "__main__":
    # load and show an image
    image = cv2.imread('/hhd3/gaoshang/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # numpy array (h, w, 3)

    # initialise BaseSegmenter
    SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = "cuda:4"
    base_segmenter = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device)
    
    # image embedding (once embedded, multiple prompts can be applied)
    base_segmenter.set_image(image)
    
    # examples
    # point only ------------------------
    mode = 'point'
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 1]), 
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=False)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)

    # both ------------------------
    mode = 'both'
    mask_input  = logits[np.argmax(scores), :, :]
    prompts = {'mask_input': mask_input [None, :, :]}
    prompts = {
        'point_coords': np.array([[500, 375], [1125, 625]]),
        'point_labels': np.array([1, 0]), 
        'mask_input': mask_input[None, :, :]
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_both.jpg', painted_image)

    # mask only ------------------------
    mode = 'mask'
    mask_input  = logits[np.argmax(scores), :, :]
    
    prompts = {'mask_input': mask_input[None, :, :]}
    
    masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_mask.jpg', painted_image)
