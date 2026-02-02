import torch
import copy
from diffusers.utils.torch_utils import randn_tensor

def is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def normalize_self_refiner_plan(plan_str):
    entries = []
    for chunk in plan_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            return "", "Self-refine plan entries must be in 'start-end:steps' format."
        range_part, steps_part = chunk.split(":", 1)
        range_part = range_part.strip()
        steps_part = steps_part.strip()
        if not steps_part:
            return "", "Self-refine plan entries must include a step count."
        if "-" in range_part:
            start_s, end_s = range_part.split("-", 1)
        else:
            start_s = end_s = range_part
        start_s = start_s.strip()
        if not is_int_string(start_s):
            return "", "Self-refine plan start position must be an integer."
        end_s = end_s.strip()
        if not is_int_string(end_s):
            return "", "Self-refine plan end position must be an integer."
        if not is_int_string(steps_part):
            return "", "Self-refine plan steps part must be an integer."
        
        entries.append({
            "start": int(start_s),
            "end": int(end_s),
            "steps": int(steps_part),
        })
    plan = entries
    return plan, ""

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

    def process_step(self, latents, noise_pred, sigma, sigma_next, generator=None, device=None, latents_next=None, pred_original_sample=None):
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
        
        if pred_original_sample is None:
            pred_original_sample = latents - sigma * noise_pred
        
        if latents_next is None:
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
            
            certain_mask_stored = certain_mask # keep bool
        else:
            certain_mask_stored = None
        self.buffer.append([certain_mask_stored, pred_original_sample, latents_next])
        return latents_next

    def perturb_latents(self, latents, buffer_latent, sigma, generator=None, device=None):
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        
        return (1.0 - sigma) * buffer_latent + sigma * noise

    def run_refinement_loop(self, 
                            latents, 
                            noise_pred, 
                            current_sigma, 
                            next_sigma, 
                            m_steps, 
                            denoise_func, 
                            step_func, 
                            clone_func=None, 
                            restore_func=None, 
                            generator=None, 
                            device=None):
        
        # Save initial state if needed
        scheduler_state = None
        if clone_func:
            scheduler_state = clone_func()

        # Step 0 (Initial)
        latents_next_0, pred_original_sample_0 = step_func(noise_pred, latents)
        
        latents_next = self.process_step(
            latents, noise_pred, current_sigma, next_sigma, 
            latents_next=latents_next_0, pred_original_sample=pred_original_sample_0
        )
        
        if self.certain_flag:
            return latents_next

        # Refinement Loop
        for ii in range(1, m_steps):
            if restore_func and scheduler_state is not None:
                restore_func(scheduler_state)
            
            # Perturb
            latents_perturbed = self.perturb_latents(
                latents, self.buffer[-1][1], current_sigma, generator=generator, device=device
            )
            
            # Denoise
            n_pred = denoise_func(latents_perturbed)
            
            # Step
            latents_next_loop, pred_original_sample_loop = step_func(n_pred, latents_perturbed)
            
            # Refine
            latents_next = self.process_step(
                latents_perturbed, n_pred, current_sigma, next_sigma,
                latents_next=latents_next_loop, pred_original_sample=pred_original_sample_loop
            )
            
            if self.certain_flag:
                break
                
        return latents_next

    def step(self, step_index, latents, noise_pred, t, timesteps, target_shape, seed_g, sample_scheduler, scheduler_kwargs, denoise_func):
        # Reset per denoising step to avoid blending with stale buffers from prior timesteps.
        self.reset_buffer()
        # Calculate sigma for PnP
        current_sigma = t.item() / 1000.0
        next_sigma = (0. if step_index == len(timesteps)-1 else timesteps[step_index+1].item()) / 1000.0
        
        m_steps = self.get_anneal_steps(step_index)

        if m_steps > 1 and not self.certain_flag:

            def _get_prev_sample(step_out):
                if hasattr(step_out, "prev_sample"):
                    return step_out.prev_sample
                if isinstance(step_out, (tuple, list)):
                    return step_out[0]
                return step_out

            def _get_pred_original_sample(step_out, latents_in, n_pred_sliced):
                if hasattr(step_out, "pred_original_sample"):
                    return step_out.pred_original_sample
                t_val = t.item() if torch.is_tensor(t) else float(t)
                return latents_in - (t_val / 1000.0) * n_pred_sliced

            def step_func(n_pred_in, latents_in):
                # Correct slicing: 
                # [:, :channels] slices Dimension 1
                # [:, :, :frames] slices Dimension 2
                n_pred_sliced = n_pred_in[:, :latents_in.shape[1], :target_shape[1]]

                nonlocal sample_scheduler
                step_out = sample_scheduler.step(n_pred_sliced, t, latents_in, **scheduler_kwargs)
                latents_next_out = _get_prev_sample(step_out)
                pred_original_sample_out = _get_pred_original_sample(step_out, latents_in, n_pred_sliced)
                return latents_next_out, pred_original_sample_out

            def clone_func():
                if sample_scheduler is None:
                    return None
                if getattr(sample_scheduler, "is_stateful", True):
                    return copy.deepcopy(sample_scheduler)
                return None

            def restore_func(saved_state):
                nonlocal sample_scheduler
                if saved_state:
                     sample_scheduler = copy.deepcopy(saved_state)

            latents = self.run_refinement_loop(
                latents=latents,
                noise_pred=noise_pred,
                current_sigma=current_sigma,
                next_sigma=next_sigma,
                m_steps=m_steps,
                denoise_func=denoise_func,
                step_func=step_func,
                clone_func=clone_func,
                restore_func=restore_func,
                generator=seed_g,
                device=latents.device
            )
        else:
            # Standard logic
            # Correct slicing: [:, :channels, :frames]
            n_pred_sliced = noise_pred[:, :latents.shape[1], :target_shape[1]]
            step_out = sample_scheduler.step( n_pred_sliced, t, latents, **scheduler_kwargs)
            if hasattr(step_out, "prev_sample"):
                latents = step_out.prev_sample
            elif isinstance(step_out, (tuple, list)):
                latents = step_out[0]
            else:
                latents = step_out
        
        return latents, sample_scheduler

def create_self_refiner_handler(pnp_plan, pnp_f_uncertainty, pnp_p_norm, pnp_certain_percentage):
    if len(pnp_plan):
        stochastic_plan, error = normalize_self_refiner_plan(pnp_plan)
    else:
        # Default plan from paper/code
        stochastic_plan = [
            {"start": 1, "end": 5, "steps": 3},
            {"start": 6, "end": 13, "steps": 1},
        ]

    return PnPHandler(stochastic_plan, ths_uncertainty=pnp_f_uncertainty, p_norm=pnp_p_norm, certain_percentage=pnp_certain_percentage)
