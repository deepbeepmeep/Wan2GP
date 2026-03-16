# Thanks to https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/film_grain.py
import torch

def add_film_grain(images: torch.Tensor, grain_intensity: float = 0, saturation: float = 0.5):
    device = images.device
    input_was_uint8 = images.dtype == torch.uint8
    if input_was_uint8:
        images = images.float().div_(255.0).mul_(2.0).sub_(1.0)

    images = images.permute(1, 2, 3, 0)
    images.add_(1.0).div_(2.0)
    grain = torch.randn_like(images, device=device)
    grain[:, :, :, 0] *= 2
    grain[:, :, :, 2] *= 3
    grain = grain * saturation + grain[:, :, :, 1].unsqueeze(3).repeat(
        1, 1, 1, 3
    ) * (1 - saturation)

    # Blend the grain with the image
    noised_images = images + grain_intensity * grain
    noised_images.clamp_(0, 1)
    noised_images.sub_(0.5).mul_(2.0)
    noised_images = noised_images.permute(3, 0, 1, 2)
    if input_was_uint8:
        noised_images = noised_images.add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8)
    return noised_images
