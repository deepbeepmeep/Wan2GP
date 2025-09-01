import torch


def distribute_into_histogram(loss: torch.Tensor,
                              t: torch.Tensor,
                              num_bins: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    loss = loss.detach().flatten()
    t = t.detach().flatten()
    t = (t * num_bins).long()
    hist = torch.zeros(num_bins, device=loss.device)
    count = torch.zeros(num_bins, device=loss.device)
    hist.scatter_add_(0, t, loss)
    count.scatter_add_(0, t, torch.ones_like(loss))
    return hist, count
