# Thanks to https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/film_grain.py
import torch

def add_film_grain(images: torch.Tensor, grain_intensity: float = 0, saturation: float = 0.5):
    device = images.device 

    images = images.permute(1, 2 ,3 ,0)
    images.add_(1.).div_(2.)
    grain = torch.randn_like(images, device=device)
    grain[:, :, :, 0] *= 2
    grain[:, :, :, 2] *= 3
    grain = grain * saturation + grain[:, :, :, 1].unsqueeze(3).repeat(
        1, 1, 1, 3
    ) * (1 - saturation)

    # Blend the grain with the image
    noised_images = images + grain_intensity * grain
    noised_images.clamp_(0, 1)
    noised_images.sub_(.5).mul_(2.)
    noised_images = noised_images.permute(3, 0, 1 ,2)
    return noised_images
