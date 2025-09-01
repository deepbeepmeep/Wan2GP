from typing import Callable

import torch
import torch.nn as nn
import math

class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""
    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition_type=None, token_replace_vec=None) -> torch.Tensor:
        x_out = self.linear(self.act(x))

        if condition_type == "token_replace":
            x_token_replace_out = self.linear(self.act(token_replace_vec))
            return x_out, x_token_replace_out
        else:
            return x_out

def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_(x, shift=None, scale=None):

    if scale is None and shift is None:
        return x
    elif shift is None:
        scale = scale + 1
        scale = scale.unsqueeze(1)
        return x.mul_(scale) 
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        scale = scale + 1
        scale = scale.unsqueeze(1)
        # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        torch.addcmul(shift.unsqueeze(1), x,  scale, out =x )
        return x 
    
def modulate(x, shift=None, scale=None, condition_type=None,
             tr_shift=None, tr_scale=None,
             frist_frame_token_num=None):
    if condition_type == "token_replace":
        x_zero = x[:, :frist_frame_token_num] * (1 + tr_scale.unsqueeze(1)) + tr_shift.unsqueeze(1)
        x_orig = x[:, frist_frame_token_num:] * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = torch.concat((x_zero, x_orig), dim=1)
        return x
    else:
        if scale is None and shift is None:
            return x
        elif shift is None:
            return x * (1 + scale.unsqueeze(1))
        elif scale is None:
            return x + shift.unsqueeze(1)
        else:
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            
def apply_gate(x, gate=None, tanh=False, condition_type=None, tr_gate=None, frist_frame_token_num=None):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if condition_type == "token_replace":
        if gate is None:
            return x
        if tanh:
            x_zero = x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1).tanh()
            x_orig = x[:, frist_frame_token_num:] * gate.unsqueeze(1).tanh()
            x = torch.concat((x_zero, x_orig), dim=1)
            return x
        else:
            x_zero = x[:, :frist_frame_token_num] * tr_gate.unsqueeze(1)
            x_orig = x[:, frist_frame_token_num:] * gate.unsqueeze(1)
            x = torch.concat((x_zero, x_orig), dim=1)
            return x
    else:
        if gate is None:
            return x
        if tanh:
            return x * gate.unsqueeze(1).tanh()
        else:
            return x * gate.unsqueeze(1)
        
def apply_gate_and_accumulate_(accumulator, x, gate=None, tanh=False):
    if gate is None:
        return accumulator
    if tanh:
        return accumulator.addcmul_(x, gate.unsqueeze(1).tanh())   
    else:
        return accumulator.addcmul_(x, gate.unsqueeze(1))
    
def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
