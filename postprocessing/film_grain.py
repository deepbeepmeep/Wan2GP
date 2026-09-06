# Thanks to https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/film_grain.py
import torch
from shared.utils.utils import get_default_workers, process_images_multithread


def is_film_grain_enabled(grain_intensity) -> bool:
    try:
        return float(grain_intensity) > 0
    except (TypeError, ValueError):
        return False


def add_film_grain(images: torch.Tensor, grain_intensity: float = 0, saturation: float = 0.5):
    """Add grain to CTHW images, reusing uint8 input storage in place."""
    if grain_intensity == 0:
        return images

    input_was_uint8 = images.dtype == torch.uint8
    output = images if input_was_uint8 else torch.empty_like(images)
    frame_count = images.shape[1]
    # Each worker needs two RGB frames and one monochrome plane; budget 256 MiB, or at least one frame.
    scratch_bytes = 7 * images.shape[2] * images.shape[3] * (4 if input_was_uint8 else images.element_size())
    workers = min(frame_count, max(1, int(get_default_workers())), max(1, (256 * 1024**2) // scratch_bytes)) if images.device.type == "cpu" else 1
    seeds = torch.randint(0, 2**63 - 1, (frame_count,), device="cpu").tolist()

    @torch.inference_mode()
    def process_frames(worker_no):
        frame = torch.empty_like(images[:, 0], dtype=torch.float32 if input_was_uint8 else images.dtype, memory_format=torch.contiguous_format)
        grain = torch.empty_like(frame)
        monochrome_grain = torch.empty_like(frame[1:2])
        generator = torch.Generator(device=images.device)
        for frame_no in range(worker_no, frame_count, workers):
            frame.copy_(images[:, frame_no])
            if input_was_uint8:
                frame.div_(255.0).mul_(2.0).sub_(1.0)
            frame.add_(1.0).div_(2.0)

            grain.normal_(generator=generator.manual_seed(seeds[frame_no]))
            grain[0].mul_(2)
            grain[2].mul_(3)
            monochrome_grain.copy_(grain[1:2]).mul_(1 - saturation)
            grain.mul_(saturation).add_(monochrome_grain)

            frame.add_(grain.mul_(grain_intensity)).clamp_(0, 1).sub_(0.5).mul_(2.0)
            if input_was_uint8:
                frame.add_(1.0).mul_(127.5).clamp_(0, 255)
            output[:, frame_no].copy_(frame)

    process_images_multithread(process_frames, range(workers), "upsample", wrap_in_list=False, max_workers=workers)
    return output
