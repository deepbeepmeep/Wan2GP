# core.py
#
# Frames2Video model backend for Wan2GP.
# This file integrates directly into Wan2GP’s unified inference engine.
#
# Wan2GP will call:
#   run_frames2video(self, model_def, inputs)
#
# IMPORTANT:
#   - No UI logic here
#   - No file writing here
#   - No plugin tab logic
#   - No HuggingFace downloads here
#
# Wan2GP handles:
#   - model loading
#   - LoRAs
#   - quantization
#   - VRAM/offload
#   - video writing
#
# This file only implements the algorithmic heart of the morph.


import torch
import torchvision.transforms.functional as TF


# ------------------------------------------------------------
# 1. Inference Entry Point
# ------------------------------------------------------------
def run_frames2video(self, model_def, inputs):
    """
    Main Frames2Video inference path.

    Called from any2video.py when:
        model_def["frames2video_class"] == True

    Parameters:
        self   : Wan2GP pipeline object (contains VAE, device, etc.)
        model_def : dict with model metadata
        inputs : dict of Wan2GP inputs (image_start, image_end, etc.)

    Returns:
        A tensor shaped [C, F, H, W] in [-1, 1]
        Wan2GP will decode + write the video.
    """

    # --------------------------------------------------------
    # 1. Extract inputs from Wan2GP
    # --------------------------------------------------------
    start_img = inputs.get("image_start")
    end_img   = inputs.get("image_end")
    middle_imgs = inputs.get("middle_images")
    middle_ts   = inputs.get("middle_images_timestamps")
    max_area    = inputs.get("max_area")

    if start_img is None or end_img is None:
        raise ValueError("Frames2Video requires image_start and image_end.")

    # Convert PIL → normalized tensor [-1,1]
    def to_tensor(img):
        return TF.to_tensor(img).sub_(0.5).div_(0.5)

    start_tensor = to_tensor(start_img)
    end_tensor   = to_tensor(end_img)

    middle_tensors = []
    if middle_imgs:
        for img in middle_imgs:
            middle_tensors.append(to_tensor(img))

    # --------------------------------------------------------
    # 2. Move tensors to the correct device
    # --------------------------------------------------------
    device = self.device
    start_tensor = start_tensor.to(device)
    end_tensor   = end_tensor.to(device)
    middle_tensors = [t.to(device) for t in middle_tensors]

    # --------------------------------------------------------
    # 3. Call Wan2GP’s model.generate()
    # --------------------------------------------------------
    # We call the Wan2.2 I2V model with our custom inputs.
    # Wan2GP handles:
    #   - VAE
    #   - CLIP
    #   - LoRAs
    #   - quantization
    #   - offload
    #
    # We only pass the tensors.
    # --------------------------------------------------------

    video_tensor = self.generate(
        input_prompt="",                     # No text prompt for morphing
        img=start_tensor,
        img_end=end_tensor,
        middle_images=middle_tensors if middle_tensors else None,
        middle_images_timestamps=middle_ts if middle_ts else None,
        frame_num=81,                        # Default; can expose later
        max_area=max_area,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        seed=-1,
        offload_model=True,
    )

    # --------------------------------------------------------
    # 4. Return the latent video tensor
    # --------------------------------------------------------
    # Shape: [C, F, H, W], values in [-1,1]
    # Wan2GP will decode + write the video.
    # --------------------------------------------------------
    return video_tensor
