from typing import Optional, Union

import torch

from .pipeline import Kandinsky5Pipeline


class Kandinsky5T2IPipeline(Kandinsky5Pipeline):
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
        super().__init__(
            mode="t2i",
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
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: float = 3.0,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        joint_pass: bool = False,
    ):
        return super().__call__(
            text=text,
            image=None,
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
