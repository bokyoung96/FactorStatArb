from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


def turnover_cost(
    w: torch.Tensor,
    w_prev: Optional[torch.Tensor],
    turn_penalty: float = 5e-4,
    short_penalty: float = 1e-4,
) -> torch.Tensor:
    if w_prev is None:
        w_prev = torch.zeros_like(w)
    turn = torch.abs(w - w_prev).sum(-1)
    short_cost = torch.clamp(-w, min=0.0).sum(-1)
    return turn_penalty * turn + short_penalty * short_cost


def explained_variance(residual: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = mask > 0
    if not valid.any():
        return torch.tensor(0.0, device=returns.device, dtype=returns.dtype)
    res_var = torch.var(residual[valid])
    ret_var = torch.var(returns[valid])
    return 1.0 - res_var / torch.clamp(ret_var, min=eps)


def rolling_sharpe(net_returns: Iterable[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    buf = [r for r in net_returns]
    stacked = torch.stack(buf)
    mean = stacked.mean()
    std = stacked.std(unbiased=False)
    return mean / torch.clamp(std, min=eps)


def detach_tail(buffer: Iterable[torch.Tensor], keep: int) -> list:
    """
    Keep last `keep-1` items detached to avoid exploding graphs; caller
    typically appends a fresh tensor for gradient flow.
    """
    tail = list(buffer)[-keep:]
    if not tail:
        return []
    detached = [t.detach() for t in tail[:-1]] + [tail[-1]]
    return detached
