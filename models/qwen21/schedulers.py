# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The sampling schedule released with Qwen Image 2.1 Turbo."""
from diffusers import FlowMatchEulerDiscreteScheduler


class QwenImage21TurboScheduler(FlowMatchEulerDiscreteScheduler):
    # Qwen/Qwen-Image-2.1-Turbo, model_index.json at d65dbc9a7e8f6b5479e33dee6030eaab2a906509.
    sample_sigmas = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)

    def set_timesteps(self, num_inference_steps=None, device=None):
        if num_inference_steps not in (None, len(self.sample_sigmas)):
            raise ValueError("Qwen Image 2.1 Turbo scheduler requires 8 inference steps.")
        self.register_to_config(use_dynamic_shifting=False, shift=1.0, shift_terminal=None)
        self.set_shift(1.0)
        return super().set_timesteps(len(self.sample_sigmas), device=device, sigmas=self.sample_sigmas)
