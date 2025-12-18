import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .groups import (
    FEATURE_ORDER,
    LIQUIDITY_FEATURES,
    MOMENTUM_FEATURES,
    PRICE_FEATURES,
    TECHNICAL_FEATURES,
    VOLATILITY_FEATURES,
    group_dims_for,
)


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GRN(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hid, d_out)
        self.drop = nn.Dropout(drop)
        self.glu = GLU(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = self.skip(x)
        x = self.fc1(x)
        if ctx is not None:
            x = x + ctx
        x = self.elu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.glu(x)
        return self.norm(x + res)


class VSN(nn.Module):
    def __init__(self, n_vars: int, d_in: int, d_hid: int, drop: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.grns = nn.ModuleList([GRN(d_in, d_hid, d_hid, drop) for _ in range(n_vars)])
        self.w_grn = GRN(n_vars * d_in, d_hid, n_vars, drop)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_grn(x)
        w = self.sm(w).unsqueeze(-1)
        processed = []
        for i in range(self.n_vars):
            feat = x[..., i : i + 1]
            processed.append(self.grns[i](feat))
        stack = torch.stack(processed, dim=2)
        return (stack * w).sum(dim=2)


class PosEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        drop: float = 0.1,
        n_class: int = 2,
        max_len: int = 5000,
    ):
        super().__init__()

        expected = len(FEATURE_ORDER)
        if int(n_feat) != expected:
            raise ValueError(f"Expected n_feat={expected} based on FEATURE_ORDER, got n_feat={n_feat}.")

        d_price, d_mom, d_vol, d_liq, d_tech = group_dims_for(int(d_model))
        if d_price + d_mom + d_vol + d_liq + d_tech != int(d_model):
            raise ValueError("Group dims must sum to d_model.")

        """ 
        NOTE:
        Feature groups are embedded separately to provide semantic structure
        to the Transformer. This stabilizes attention by allowing the model
        to reason over information types (price, momentum, risk, liquidity,
        technical) rather than raw features
        """
        self.price_proj = nn.Linear(len(PRICE_FEATURES), d_price)
        self.mom_proj = nn.Linear(len(MOMENTUM_FEATURES), d_mom)
        self.vol_proj = nn.Linear(len(VOLATILITY_FEATURES), d_vol)
        self.liq_proj = nn.Linear(len(LIQUIDITY_FEATURES), d_liq)
        self.tech_proj = nn.Linear(len(TECHNICAL_FEATURES), d_tech)
        self.act = nn.GELU()
        self.in_norm = nn.LayerNorm(int(d_model))

        self.pe = PosEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        except TypeError:
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_grn = GRN(d_model, d_ff, d_model, drop)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_class)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq, n_features) with features ordered as FEATURE_ORDER.
        p_end = len(PRICE_FEATURES)
        m_end = p_end + len(MOMENTUM_FEATURES)
        v_end = m_end + len(VOLATILITY_FEATURES)
        l_end = v_end + len(LIQUIDITY_FEATURES)
        t_end = l_end + len(TECHNICAL_FEATURES)
        if x.size(-1) != t_end:
            raise ValueError("Unexpected feature dimension; ensure config features match FEATURE_ORDER.")

        e_price = self.act(self.price_proj(x[..., :p_end]))
        e_mom = self.act(self.mom_proj(x[..., p_end:m_end]))
        e_vol = self.act(self.vol_proj(x[..., m_end:v_end]))
        e_liq = self.act(self.liq_proj(x[..., v_end:l_end]))
        e_tech = self.act(self.tech_proj(x[..., l_end:t_end]))

        x = torch.cat([e_price, e_mom, e_vol, e_liq, e_tech], dim=-1)
        x = self.in_norm(x)
        x = self.pe(x)
        x = self.enc(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.out_grn(x)
        x = self.norm(x)
        return self.head(x)
