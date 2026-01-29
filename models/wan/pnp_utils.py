import torch
from diffusers.utils.torch_utils import randn_tensor

class PnPHandler:
    def __init__(self, stochastic_plan, ths_uncertainty=0.0, p_norm=1, certain_percentage=0.999):
        self.stochastic_step_map = self._build_stochastic_step_map(stochastic_plan)
        self.ths_uncertainty = ths_uncertainty
        self.p_norm = p_norm
        self.certain_percentage = certain_percentage
        self.buffer = [None] # [certain_mask, pred_original_sample, latents_next]
        self.certain_flag = False

    def _build_stochastic_step_map(self, plan):
        step_map = {}
        if not plan:
            return step_map
        
        for entry in plan:
            if isinstance(entry, dict):
                start = entry.get("start", entry.get("begin"))
                end = entry.get("end", entry.get("stop"))
                steps = entry.get("steps", entry.get("anneal", entry.get("num_anneal_steps", 1)))
            else:
                start, end, steps = entry
            
            start_i = int(start)
            end_i = int(end)
            steps_i = int(steps)
            
            if steps_i > 0:
                for idx in range(start_i, end_i + 1):
                    step_map[idx] = steps_i
        return step_map

    def get_anneal_steps(self, step_index):
        return self.stochastic_step_map.get(step_index, 0)

    def reset_buffer(self):
        self.buffer = [None]
        self.certain_flag = False

    def process_step(self, latents, noise_pred, sigma, sigma_next, generator=None, device=None):
        """
        Returns (latents_next, buffer_updated)
        """
        # Predict original sample (x0) and next latent
        # x_t = t * x_1 + (1-t) * x_0 (Flow Matching)
        # v_t = x_1 - x_0
        # dx/dt = v_t
        # Here sigma is time t?? In Wan code usually t goes 1000->0.
        # Ref code: pred_original_sample = latents - sigma * noise_pred
        # latents_next = latents + (sigma_next - sigma) * noise_pred
        # This matches Flow Matching if sigma is time t.
        
        pred_original_sample = latents - sigma * noise_pred
        latents_next = latents + (sigma_next - sigma) * noise_pred

        if self.buffer[-1] is not None:
            # Calculate uncertainty
            # buffer[-1][1] is previous pred_original_sample
            diff = pred_original_sample - self.buffer[-1][1]
            # dim=1 is channels (C)
            uncertainty = torch.norm(diff, p=self.p_norm, dim=1) / latents.shape[1] # .shape[1] is channels
            
            certain_mask = uncertainty < self.ths_uncertainty
            if self.buffer[-1][0] is not None:
                certain_mask = certain_mask | self.buffer[-1][0]
            
            if certain_mask.sum() / certain_mask.numel() > self.certain_percentage:
                self.certain_flag = True
            
            certain_mask_float = certain_mask.to(latents.dtype).unsqueeze(1) # Broadcast channels
            
            # Blend
            latents_next = certain_mask_float * self.buffer[-1][2] + (1.0 - certain_mask_float) * latents_next
            pred_original_sample = certain_mask_float * self.buffer[-1][1] + (1.0 - certain_mask_float) * pred_original_sample
            
            # Pack for buffer
            # we need to squeeze the mask back if we store it
            certain_mask_stored = certain_mask # keep bool
        else:
            certain_mask_stored = None

        self.buffer.append([certain_mask_stored, pred_original_sample, latents_next])
        return latents_next

    def perturb_latents(self, latents, buffer_latent, sigma, generator=None, device=None):
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        # Re-noise: (1-sigma) * x0 + sigma * noise ?? 
        # Ref code: latents = (1.0 - sigma) * buffer[-1][1] + sigma * noise
        # This seems to assume x_t = (1-sigma)*x0 + sigma*noise? 
        # Wait, Flow matching usually is x_t = t * x1 + (1-t) * x0. 
        # If sigma is t, then x_sigma = sigma * x1 + (1-sigma) * x0.
        # If we target noise (epsilon), it varies.
        # The reference code uses this formula. We should probably stick to it if we assume sigma is correct.
        
        return (1.0 - sigma) * buffer_latent + sigma * noise
