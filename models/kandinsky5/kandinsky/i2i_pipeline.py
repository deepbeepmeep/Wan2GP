from typing import Optional, Union

import torch
from PIL import Image

from .pipeline import Kandinsky5Pipeline


class Kandinsky5I2IPipeline(Kandinsky5Pipeline):
    RESOLUTIONS = Kandinsky5Pipeline.IMAGE_RESOLUTIONS

    def __init__(
        self,
        device_map: Union[str, torch.device, dict],
        dit,
        text_embedder,
        vae,
        resolution: int = 1024,
        conf=None,
    ):
        if resolution not in [1024]:
            raise ValueError("Resolution can be only 1024")
        super().__init__(
            mode="i2i",
            device_map=device_map,
            dit=dit,
            text_embedder=text_embedder,
            vae=vae,
            resolution=resolution,
            conf=conf,
        )

    def __call__(
        self,
        text: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: float = 3.0,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        image: Optional[Union[Image.Image, str]] = None,
        joint_pass: bool = False,
    ):
        return super().__call__(
            text=text,
            image=image,
            time_length=0,
            width=width,
            height=height,
            frame_num=1,
            seed=seed,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            expand_prompts=expand_prompts,
            save_path=save_path,
            progress=progress,
            joint_pass=joint_pass,
        )
