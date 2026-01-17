from typing import Optional, Union

import torch
from PIL import Image

from .pipeline import Kandinsky5Pipeline


class Kandinsky5I2VPipeline(Kandinsky5Pipeline):
    def __init__(
        self,
        device_map: Union[str, torch.device, dict],
        dit,
        text_embedder,
        vae,
        conf=None,
    ):
        super().__init__(
            mode="i2v",
            device_map=device_map,
            dit=dit,
            text_embedder=text_embedder,
            vae=vae,
            resolution=conf.metrics.resolution,
            conf=conf,
        )

    def __call__(
        self,
        text: str,
        image: Union[str, Image.Image],
        time_length: int = 5,
        frame_num: Optional[int] = None,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = (
            "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, "
            "low quality, ugly, deformed, walking backwards"
        ),
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        joint_pass: bool = False,
    ):
        return super().__call__(
            text=text,
            image=image,
            time_length=time_length,
            width=None,
            height=None,
            frame_num=frame_num,
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
