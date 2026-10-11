import torch


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        rows = x.view(-1, x.shape[-1])
        for part in torch.split(rows, max(1, (64 << 20) // (rows.shape[-1] * rows.element_size()))):
            part.copy_(torch.nn.functional.gelu(part, approximate="tanh"))  # in place by rows: the same values without a second copy
        return x
