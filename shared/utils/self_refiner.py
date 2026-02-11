import torch
import copy
import uuid
from diffusers.utils.torch_utils import randn_tensor

def is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def normalize_self_refiner_plan(plan_input, max_plans: int = 1):
    default_plan = [
        {"start": 1, "end": 5, "steps": 3},
        {"start": 6, "end": 13, "steps": 1},
    ]
    if len(plan_input) > max_plans:
        return [], f"Self-refiner supports up to {max_plans} plan(s); found {len(plan_input)}."
    if not plan_input or not isinstance(plan_input, list):
        return [default_plan], ""
    
    return [plan_input], ""

def ensure_refiner_list(plan_data):
    if not isinstance(plan_data, list):
        return []
    for rule in plan_data:
        if "id" not in rule:
            rule["id"] = str(uuid.uuid4())
    return plan_data

def add_refiner_rule(current_rules, range_val, steps_val):
    new_start, new_end = int(range_val[0]), int(range_val[1])
    
    if new_start >= new_end:
         from gradio import Info
         Info(f"Start step ({new_start}) must be smaller than End step ({new_end}).")
         return current_rules

    for rule in current_rules:
        if new_start <= rule['end'] and new_end >= rule['start']:
            from gradio import Info
            Info(f"Overlap detected! Steps {new_start}-{new_end} conflict with existing rule {rule['start']}-{rule['end']}.")
            return current_rules

    new_rule = {
        "id": str(uuid.uuid4()),
        "start": new_start,
        "end": new_end,
        "steps": int(steps_val)
    }
    updated_list = current_rules + [new_rule]
    return sorted(updated_list, key=lambda x: x['start'])

def remove_refiner_rule(current_rules, rule_id):
    return [r for r in current_rules if r["id"] != rule_id]

class PnPHandler:
    def __init__(self, stochastic_plan, ths_uncertainty=0.0, p_norm=1, certain_percentage=0.999, channel_dim: int = 1):
        self.stochastic_step_map = self._build_stochastic_step_map(stochastic_plan)
        self.ths_uncertainty = ths_uncertainty
        self.p_norm = p_norm
        self.certain_percentage = certain_percentage
        self.channel_dim = channel_dim
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
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += latents.ndim
            # dim=channel_dim is channels/features
            uncertainty = torch.norm(diff, p=self.p_norm, dim=channel_dim) / latents.shape[channel_dim]
            
            certain_mask = uncertainty < self.ths_uncertainty
            if self.buffer[-1][0] is not None:
                certain_mask = certain_mask | self.buffer[-1][0]
            
            if certain_mask.sum() / certain_mask.numel() > self.certain_percentage:
                self.certain_flag = True
            
            certain_mask_float = certain_mask.to(latents.dtype).unsqueeze(channel_dim) # Broadcast channels
            
            # Blend
            latents_next = certain_mask_float * self.buffer[-1][2] + (1.0 - certain_mask_float) * latents_next
            pred_original_sample = certain_mask_float * self.buffer[-1][1] + (1.0 - certain_mask_float) * pred_original_sample
            
            certain_mask_stored = certain_mask # keep bool
        else:
            certain_mask_stored = None
        self.buffer.append([certain_mask_stored, pred_original_sample, latents_next])
        return latents_next

    def perturb_latents(self, latents, buffer_latent, sigma, generator=None, device=None, noise_mask=None):
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)

        if noise_mask is None:
            return (1.0 - sigma) * buffer_latent + sigma * noise

        sigma_t = (noise_mask.to(latents.dtype) * sigma)
        return (1.0 - sigma_t) * buffer_latent + sigma_t * noise

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
                            device=None,
                            noise_mask=None):
        
        if noise_pred is None:
            return None
        # Save initial state if needed
        scheduler_state = None
        if clone_func:
            scheduler_state = clone_func()

        # Step 0 (Initial)
        latents_next_0, pred_original_sample_0 = step_func(noise_pred, latents)
        if latents_next_0 is None or pred_original_sample_0 is None:
            return None
        
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
                latents,
                self.buffer[-1][1],
                current_sigma,
                generator=generator,
                device=device,
                noise_mask=noise_mask,
            )
            
            # Denoise
            n_pred = denoise_func(latents_perturbed)
            if n_pred is None:
                return None
            
            # Step
            latents_next_loop, pred_original_sample_loop = step_func(n_pred, latents_perturbed)
            if latents_next_loop is None or pred_original_sample_loop is None:
                return None
            
            # Refine
            latents_next = self.process_step(
                latents_perturbed, n_pred, current_sigma, next_sigma,
                latents_next=latents_next_loop, pred_original_sample=pred_original_sample_loop
            )
            
            if self.certain_flag:
                break
                
        return latents_next

    def step(self, step_index, latents, noise_pred, t, timesteps, target_shape, seed_g, sample_scheduler, scheduler_kwargs, denoise_func):
        if noise_pred is None:
            return None, sample_scheduler
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
            if latents is None:
                return None, sample_scheduler
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

def create_self_refiner_handler(pnp_plan, pnp_f_uncertainty, pnp_p_norm, pnp_certain_percentage, channel_dim: int = 1):
    plans, _ = normalize_self_refiner_plan(pnp_plan, max_plans=1)
    stochastic_plan = plans[0]

    return PnPHandler(
        stochastic_plan,
        ths_uncertainty=pnp_f_uncertainty,
        p_norm=pnp_p_norm,
        certain_percentage=pnp_certain_percentage,
        channel_dim=channel_dim,
    )


def run_refinement_loop_multi(
    handlers,
    latents_list,
    noise_pred_list,
    current_sigma,
    next_sigma,
    m_steps,
    denoise_func,
    step_func,
    generators=None,
    devices=None,
    noise_masks=None,
    stop_when: str = "all",
):
    if m_steps <= 1:
        return latents_list
    if noise_pred_list is None:
        return None
    if not isinstance(noise_pred_list, (list, tuple)) or any(pred is None for pred in noise_pred_list):
        return None

    def _should_stop():
        if stop_when == "any":
            return any(handler.certain_flag for handler in handlers)
        return all(handler.certain_flag for handler in handlers)

    latents_next_list, pred_original_list = step_func(noise_pred_list, latents_list)
    if latents_next_list is None or pred_original_list is None:
        return None
    if not isinstance(latents_next_list, (list, tuple)) or not isinstance(pred_original_list, (list, tuple)):
        return None
    if len(latents_next_list) != len(handlers) or len(pred_original_list) != len(handlers):
        return None
    if any(latent is None for latent in latents_next_list) or any(pred is None for pred in pred_original_list):
        return None

    refined_latents_list = []
    for handler, latents, latents_next, pred_original in zip(
        handlers, latents_list, latents_next_list, pred_original_list
    ):
        refined_latents_list.append(
            handler.process_step(
                latents,
                None,
                current_sigma,
                next_sigma,
                latents_next=latents_next,
                pred_original_sample=pred_original,
            )
        )
    if _should_stop():
        return refined_latents_list

    for _ in range(1, m_steps):
        perturbed_list = []
        for idx, (handler, latents) in enumerate(zip(handlers, latents_list)):
            generator = generators[idx] if generators is not None else None
            device = devices[idx] if devices is not None else latents.device
            noise_mask = noise_masks[idx] if noise_masks is not None else None
            perturbed_list.append(
                handler.perturb_latents(
                    latents,
                    handler.buffer[-1][1],
                    current_sigma,
                    generator=generator,
                    device=device,
                    noise_mask=noise_mask,
                )
            )

        noise_pred_list = denoise_func(perturbed_list)
        if noise_pred_list is None:
            return None
        if not isinstance(noise_pred_list, (list, tuple)) or any(pred is None for pred in noise_pred_list):
            return None
        latents_next_list, pred_original_list = step_func(noise_pred_list, perturbed_list)
        if latents_next_list is None or pred_original_list is None:
            return None
        if not isinstance(latents_next_list, (list, tuple)) or not isinstance(pred_original_list, (list, tuple)):
            return None
        if len(latents_next_list) != len(handlers) or len(pred_original_list) != len(handlers):
            return None
        if any(latent is None for latent in latents_next_list) or any(pred is None for pred in pred_original_list):
            return None
        refined_latents_list = []
        for handler, latents, latents_next, pred_original in zip(
            handlers, perturbed_list, latents_next_list, pred_original_list
        ):
            refined_latents_list.append(
                handler.process_step(
                    latents,
                    None,
                    current_sigma,
                    next_sigma,
                    latents_next=latents_next,
                    pred_original_sample=pred_original,
                )
            )
        if _should_stop():
            break

    return refined_latents_list
