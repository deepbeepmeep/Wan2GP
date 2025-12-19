import time
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.modeling.image_encoder import window_partition, window_unpartition, get_rel_pos, Block as image_encoder_block
import matplotlib.pyplot as plt
import PIL
from .mask_painter import mask_painter
from shared.utils import files_locator as fl

# Detect bfloat16 support once at module load
_bfloat16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
    """VRAM-optimized forward pass for SAM image encoder blocks.

    Optimizations made by DeepBeepMeep
    """
    def split_mlp(mlp, x, divide=4):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        chunk_size = int(x.shape[0] / divide)
        x_chunks = torch.split(x, chunk_size)
        for i, x_chunk in enumerate(x_chunks):
            mlp_chunk = mlp.lin1(x_chunk)
            mlp_chunk = mlp.act(mlp_chunk)
            x_chunk[...] = mlp.lin2(mlp_chunk)
        return x.reshape(x_shape)

    def get_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, q_size, k_size) -> torch.Tensor:
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)
        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = torch.zeros(B, q_h, q_w, k_h, k_w, dtype=q.dtype, device=q.device)
        attn += rel_h[:, :, :, :, None]
        attn += rel_w[:, :, :, None, :]
        return attn.view(B, q_h * q_w, k_h * k_w)

    def pay_attention(self, x: torch.Tensor, split_heads=1) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        if not _bfloat16_supported:
            qkv = qkv.to(torch.float16)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        if split_heads == 1:
            attn_mask = None
            if self.use_rel_pos:
                attn_mask = get_decomposed_rel_pos(q, self.rel_pos_h.to(q), self.rel_pos_w.to(q), (H, W), (H, W))
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        else:
            chunk_size = self.num_heads // split_heads
            x = torch.empty_like(q)
            q_chunks = torch.split(q, chunk_size)
            k_chunks = torch.split(k, chunk_size)
            v_chunks = torch.split(v, chunk_size)
            x_chunks = torch.split(x, chunk_size)
            for x_chunk, q_chunk, k_chunk, v_chunk in zip(x_chunks, q_chunks, k_chunks, v_chunks):
                attn_mask = None
                if self.use_rel_pos:
                    attn_mask = get_decomposed_rel_pos(q_chunk, self.rel_pos_h.to(q), self.rel_pos_w.to(q), (H, W), (H, W))
                x_chunk[...] = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=attn_mask, scale=self.scale)
            del x_chunk, q_chunk, k_chunk, v_chunk
        del q, k, v, attn_mask
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        if not _bfloat16_supported:
            x = x.to(torch.bfloat16)

        return self.proj(x)

    shortcut = x
    x = self.norm1(x)
    # Window partition
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)
    x_shape = x.shape

    if x_shape[0] > 10:
        chunk_size = int(x.shape[0] / 4) + 1
        x_chunks = torch.split(x, chunk_size)
        for i, x_chunk in enumerate(x_chunks):
            x_chunk[...] = pay_attention(self.attn, x_chunk)
    else:
        x = pay_attention(self.attn, x, 4)

    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
    x += shortcut
    shortcut[...] = self.norm2(x)
    x += split_mlp(self.mlp, shortcut)

    return x


def set_image_encoder_patch():
    """Apply VRAM optimizations to SAM image encoder blocks."""
    if not hasattr(image_encoder_block, "patched"):
        image_encoder_block.forward = _patched_forward
        image_encoder_block.patched = True 


class BaseSegmenter:
    def __init__(self, SAM_checkpoint, model_type, device='cuda:0'):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        """
        print(f"Initializing BaseSegmenter to {device}")
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        # Apply VRAM optimizations before loading model
        set_image_encoder_patch()

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
