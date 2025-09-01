import math
import torch
from typing import Optional, Union, Tuple


# @torch.jit.script
def get_similarity(mk: torch.Tensor,
                   ms: torch.Tensor,
                   qk: torch.Tensor,
                   qe: torch.Tensor,
                   add_batch_dim: bool = False,
                   uncert_mask = None) -> torch.Tensor:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    # Return: B*N*HW
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)

    CK = mk.shape[1]

    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    
    # query token selection based on temporal sparsity
    if uncert_mask is not None:
        uncert_mask = uncert_mask.flatten(start_dim=2)
        uncert_mask = uncert_mask.expand(-1, 64, -1)
        qk = qk * uncert_mask
        qe = qe * uncert_mask
    # Behold the work of DeeBeepMeep the Code Butcher !
    if qe is not None:
        # See XMem's appendix for derivation
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab =  mk @ (qk * qe)
        two_ab *= 2
        two_ab.sub_(a_sq)
        del a_sq
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        two_ab.sub_(b_sq)
        similarity = two_ab
        del b_sq, two_ab 
        # similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = mk.transpose(1, 2) @ qk
        two_ab *= 2
        two_ab.sub_(a_sq)
        del a_sq
        similarity = two_ab
        del two_ab 
        # similarity = (-a_sq + two_ab)

    if ms is not None:
        similarity *= ms
        similarity /=  math.sqrt(CK)
        # similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
    else:
        similarity /=  math.sqrt(CK)
        # similarity = similarity / math.sqrt(CK)  # B*N*HW

    return similarity


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


def get_affinity(mk: torch.Tensor, ms: torch.Tensor, qk: torch.Tensor,
                 qe: torch.Tensor, uncert_mask = None) -> torch.Tensor:
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe, uncert_mask=uncert_mask)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity: torch.Tensor, mv: torch.Tensor, uncert_mask: torch.Tensor=None) -> torch.Tensor:
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    if uncert_mask is not None:
        uncert_mask = uncert_mask.flatten(start_dim=2).expand(-1, CV, -1)
        mem = mem * uncert_mask
    mem = mem.view(B, CV, H, W)

    return mem
