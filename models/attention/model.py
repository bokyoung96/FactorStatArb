from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.data import FactorBatch, FactorResult


@dataclass(frozen=True)
class AttentionOutputs:
    weights: torch.Tensor
    factor_returns: torch.Tensor
    predicted: torch.Tensor
    residuals: torch.Tensor
    omega_eps: torch.Tensor
    beta: torch.Tensor


class AttentionFactorLayer(nn.Module):
    def __init__(self, num_features: int, num_factors: int, emb_dim: int, ridge: float = 1e-3) -> None:
        super().__init__()
        self.W_K = nn.Linear(num_features, emb_dim, bias=False)
        self.Q = nn.Parameter(torch.randn(num_factors, emb_dim))
        self.ridge = ridge

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
        weights = F.softmax(scores, dim=-1)  # (factors, assets)
        factor_returns = weights @ returns
        predicted = weights.T @ factor_returns  # (assets,)
        n_assets = returns.size(0)
        reg = self.ridge if self.ridge is not None else 0.0
        eye = torch.eye(weights.size(0), device=weights.device, dtype=weights.dtype)
        gram = weights @ weights.T + reg * eye
        inv = torch.linalg.inv(gram)
        beta_t = weights.T @ inv  # (assets, factors)
        omega_eps = torch.eye(n_assets, device=weights.device, dtype=weights.dtype) - beta_t @ weights
        residual = omega_eps @ returns
        return AttentionOutputs(
            weights=weights,
            factor_returns=factor_returns,
            predicted=predicted,
            residuals=residual,
            omega_eps=omega_eps,
            beta=beta_t.T,
        )


def run_layer(layer: AttentionFactorLayer, batch: FactorBatch) -> FactorResult:
    out = layer(batch.features, batch.returns, batch.mask)
    return FactorResult(
        weights=out.weights,
        factor_returns=out.factor_returns,
        predicted=out.predicted,
        residuals=out.residuals,
        omega_eps=out.omega_eps,
        beta=out.beta,
    )
