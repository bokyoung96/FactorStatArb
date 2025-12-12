from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention.data import FactorBatch, FactorResult


class AttentionFactorLayer(nn.Module):
    def __init__(self, num_features: int, num_factors: int, emb_dim: int) -> None:
        super().__init__()
        self.W_K = nn.Linear(num_features, emb_dim, bias=False)
        self.Q = nn.Parameter(torch.randn(num_factors, emb_dim))

    def forward(
        self,
        features: torch.Tensor,
        returns: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_emb = self.W_K(features)
        scores = (self.Q @ x_emb.T) / (self.Q.size(1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0) <= 0, -1e9)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = F.softmax(scores, dim=-1)
        factor_returns = weights @ returns
        predicted = weights.T @ factor_returns
        residual = returns - predicted
        return weights, factor_returns, predicted, residual


def run_layer(layer: AttentionFactorLayer, batch: FactorBatch) -> FactorResult:
    weights, fr, pred, res = layer(batch.features, batch.returns, batch.mask)
    return FactorResult(weights=weights, factor_returns=fr, predicted=pred, residuals=res)
