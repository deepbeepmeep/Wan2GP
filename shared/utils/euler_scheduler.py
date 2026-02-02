import numpy as np
import torch


def _timestep_transform(t, shift=5.0, num_timesteps=1000):
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    return new_t * num_timesteps


class EulerSchedulerOutput:
    def __init__(self, prev_sample, pred_original_sample=None):
        self.prev_sample = prev_sample
        if pred_original_sample is not None:
            self.pred_original_sample = pred_original_sample

    def __getitem__(self, index):
        if index == 0:
            return self.prev_sample
        raise IndexError("EulerSchedulerOutput only supports index 0.")

    def __iter__(self):
        yield self.prev_sample


class EulerScheduler:
    is_stateful = False

    def __init__(self, num_train_timesteps=1000, use_timestep_transform=True):
        self.num_train_timesteps = num_train_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps, device=None, shift=5.0):
        self.num_inference_steps = num_inference_steps
        timesteps = list(
            np.linspace(self.num_train_timesteps, 1, num_inference_steps, dtype=np.float32)
        )
        timesteps.append(0.0)
        if device is None:
            timesteps = [torch.tensor([t]) for t in timesteps]
        else:
            timesteps = [torch.tensor([t], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                _timestep_transform(t, shift=shift, num_timesteps=self.num_train_timesteps)
                for t in timesteps
            ][:-1]
        self.timesteps = torch.tensor(timesteps)
        return self.timesteps

    def _timestep_to_index(self, timestep):
        if self.timesteps is None:
            raise ValueError("Timesteps are not set. Call set_timesteps first.")
        if torch.is_tensor(timestep):
            if timestep.numel() != 1:
                t_val = timestep.flatten()[0].item()
            else:
                t_val = timestep.item()
        else:
            t_val = float(timestep)
        diff = (self.timesteps - t_val).abs()
        idx = int(torch.argmin(diff).item())
        return idx, t_val

    def step(self, model_output, timestep, sample, return_dict=True, **kwargs):
        if self.timesteps is None:
            raise ValueError("Timesteps are not set. Call set_timesteps first.")
        idx, t_val = self._timestep_to_index(timestep)
        if idx + 1 < len(self.timesteps):
            dt_raw = self.timesteps[idx] - self.timesteps[idx + 1]
        else:
            dt_raw = self.timesteps[idx]
        dt = dt_raw.item() / self.num_train_timesteps
        prev_sample = sample - model_output * dt
        pred_original_sample = sample - (t_val / self.num_train_timesteps) * model_output
        if not return_dict:
            return (prev_sample,)
        return EulerSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )

    def scale_model_input(self, sample, *args, **kwargs):
        return sample
