from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        lookback: int,
        dropout: float = 0.1,
        squash: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.squash = squash
        self.eps = eps
        self.kernel = nn.Parameter(torch.randn(hidden_dim, lookback))
        self.skip = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _squash_kernel(self) -> torch.Tensor:
        if self.squash <= 0:
            return self.kernel
        sign = torch.sign(self.kernel)
        mag = torch.clamp(torch.abs(self.kernel) - self.squash, min=0.0)
        return sign * mag

    def forward(self, residuals: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, n_assets, _ = residuals.shape
        x = residuals.reshape(bsz * n_assets, 1, self.lookback)
        kernel = self._squash_kernel().view(self.hidden_dim, 1, self.lookback)
        conv = F.conv1d(x, kernel, padding=0)  # (bsz*n_assets, hidden, 1)
        conv = conv.view(bsz, n_assets, self.hidden_dim)
        skip = self.skip.view(1, 1, self.hidden_dim) * residuals[..., -1:].expand(-1, -1, self.hidden_dim)
        feats = self.dropout(conv + skip)
        scores = self.out(feats).squeeze(-1)  # (bsz, assets)
        if mask is not None:
            scores = scores * mask
        weights = scores
        if mask is not None:
            weights = weights * mask
        denom = weights.abs().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = weights / denom
        return weights

    def forward_with_feats(
        self, residuals: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_assets, _ = residuals.shape
        x = residuals.reshape(bsz * n_assets, 1, self.lookback)
        kernel = self._squash_kernel().view(self.hidden_dim, 1, self.lookback)
        conv = F.conv1d(x, kernel, padding=0)
        conv = conv.view(bsz, n_assets, self.hidden_dim)
        skip = self.skip.view(1, 1, self.hidden_dim) * residuals[..., -1:].expand(-1, -1, self.hidden_dim)
        feats = self.dropout(conv + skip)
        scores = self.out(feats).squeeze(-1)
        if mask is not None:
            scores = scores * mask
        weights = scores
        if mask is not None:
            weights = weights * mask
        denom = weights.abs().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = weights / denom
        return weights, feats
