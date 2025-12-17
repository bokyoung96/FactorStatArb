from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from attention.cache import ResidualCache
from attention.data import FeatureRepository, FactorBatch
from attention.losses import explained_variance, rolling_sharpe, detach_tail
from attention.longconv import LongConv
from attention.model import AttentionFactorLayer
from attention.writer import FactorResultWriter


def _freq(freq: str) -> str:
    f = str(freq or "D").upper()
    return {"D": "D", "B": "B", "W": "W-FRI", "M": "ME"}.get(f, f)


def _reb_dates(dates: Sequence, freq: str) -> set[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).sort_values().unique()
    if idx.empty:
        return set()
    f = _freq(freq)
    if f in {"D", "B"}:
        return set(pd.Timestamp(x) for x in idx)
    s = pd.Series(1, index=idx)
    out: set[pd.Timestamp] = set()
    for _, g in s.groupby(pd.Grouper(freq=f)):
        if not g.empty:
            out.add(pd.Timestamp(g.index[-1]))
    out.add(pd.Timestamp(idx[-1]))
    return out


def _keep(cfg: "AttentionConfig") -> int:
    f = _freq(cfg.rebalance)
    if f in {"D", "B"}:
        return int(cfg.sharpe_window)
    if f.startswith("W"):
        return max(4, int(cfg.sharpe_window) // 5)
    if f in {"M", "ME"}:
        return max(3, int(cfg.sharpe_window) // 21)
    return int(cfg.sharpe_window)


def _short(w: torch.Tensor) -> torch.Tensor:
    return torch.clamp(-w, min=0.0).sum(-1)


def _turn(w: torch.Tensor, w_prev: Optional[torch.Tensor]) -> torch.Tensor:
    if w_prev is None:
        w_prev = torch.zeros_like(w)
    return torch.abs(w - w_prev).sum(-1)


def _normalize_ls(w: torch.Tensor, mask: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(w)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    w = w * mask

    valid = mask.sum().clamp_min(eps)
    mean = (w * mask).sum() / valid
    w = (w - mean) * mask

    gross = w.abs().sum().clamp_min(eps)
    w = w / gross

    pos = torch.clamp(w, min=0.0)
    neg = torch.clamp(w, max=0.0)
    pos_sum = pos.sum()
    neg_sum = (-neg).sum()
    if pos_sum > eps and neg_sum > eps:
        w = pos * (0.5 / pos_sum) + neg * (0.5 / neg_sum)
    return w


@dataclass(frozen=True)
class AttentionConfig:
    num_factors: int = 32
    emb_dim: int = 32
    return_col: str = "ret1"
    normalize: bool = True
    train_epochs: int = 0
    train_batch_size: int = 32
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-4
    longconv_dim: int = 32
    lookback: int = 30
    sharpe_window: int = 60
    lambda_sr: float = 1.0
    lambda_var: float = 1.0
    ridge: float = 1e-3
    dropout: float = 0.1
    squash: float = 1e-3
    cache_dir: Optional[str] = None
    sharpe_eps: float = 1e-6
    turn_penalty: float = 5e-4
    short_penalty: float = 1e-4
    short_alpha: float = 0.0
    run_train: bool = True
    run_test: bool = True
    best_model_path: Optional[str] = None
    window_years: int = 8
    step_years: int = 1
    rebalance: str = "D"

    @classmethod
    def from_file(cls, path: Path) -> "AttentionConfig":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        fields = cls().__dataclass_fields__.keys()
        kwargs = {k: raw[k] for k in raw if k in fields}
        return cls(**kwargs)


class StatArbModel(nn.Module):
    def __init__(self, cfg: AttentionConfig, num_features: int, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.attention = AttentionFactorLayer(
            num_features=num_features,
            num_factors=cfg.num_factors,
            emb_dim=cfg.emb_dim,
            ridge=cfg.ridge,
        )
        self.sequence = LongConv(
            hidden_dim=cfg.longconv_dim,
            lookback=cfg.lookback,
            dropout=cfg.dropout,
            squash=cfg.squash,
            eps=cfg.sharpe_eps,
        )
        self.device = device

    def forward(self, window_batches: Sequence[FactorBatch], target_batch: FactorBatch) -> Dict[str, torch.Tensor]:
        residuals = []
        masks = []
        for b in window_batches:
            out = self.attention(b.features, b.returns, b.mask)
            residuals.append(out.residuals)
            masks.append(b.mask)
        res_stack = torch.stack(residuals, dim=0).transpose(0, 1).unsqueeze(0)  # (1, assets, lookback)
        mask_stack = torch.stack(masks, dim=0)
        joint_mask = mask_stack.min(dim=0).values.unsqueeze(0)
        port_weights = self.sequence(res_stack, mask=joint_mask).squeeze(0)  # (assets,)

        target_out = self.attention(target_batch.features, target_batch.returns, target_batch.mask)
        w_asset = target_out.omega_eps.T @ port_weights
        w_asset = _normalize_ls(w_asset, target_batch.mask, eps=self.cfg.sharpe_eps)

        return {
            "w_asset": w_asset,
            "w_port": port_weights,
            "target": target_out,
            "res_window": res_stack.squeeze(0),
            "mask_window": joint_mask.squeeze(0),
        }


class StatArbTrainer:
    def __init__(
        self,
        cfg: AttentionConfig,
        repository: FeatureRepository,
        device: torch.device,
        writer: FactorResultWriter,
        cache: ResidualCache,
        best_path: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.repository = repository
        self.device = device
        sample = repository.load_snapshot(repository.latest_date())
        self.model = StatArbModel(cfg, num_features=sample.features.size(1), device=device).to(device)
        self.writer = writer
        self.cache = cache
        self.best_path = Path(best_path) if best_path is not None else None
        self.best_sharpe: Optional[float] = None
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train_lr,
            weight_decay=cfg.train_weight_decay,
        )

    def _window(self, dates: Sequence, pos: int) -> Tuple[List[FactorBatch], FactorBatch, str]:
        window_dates = dates[pos - self.cfg.lookback : pos]
        target_date = dates[pos]
        batches = [self.repository.load_snapshot(d) for d in window_dates]
        target = self.repository.load_snapshot(target_date)
        return batches, target, str(target_date)

    def train(self, dates: Sequence) -> None:
        if not dates or self.cfg.train_epochs <= 0:
            return
        lookback = self.cfg.lookback
        for ep in range(self.cfg.train_epochs):
            reb = _reb_dates(dates[lookback:], self.cfg.rebalance)
            pos = [i for i in range(lookback, len(dates)) if pd.Timestamp(dates[i]) in reb]
            if not pos:
                return
            bar = tqdm(pos, desc=f"joint train {ep+1}/{self.cfg.train_epochs}", leave=False)
            sharpe_buf: List[torch.Tensor] = []
            short_buf: List[torch.Tensor] = []
            w_prev: Optional[torch.Tensor] = None
            k = _keep(self.cfg)
            for pi, idx in enumerate(bar):
                window_batches, target_batch, date_key = self._window(dates, idx)
                out = self.model(window_batches, target_batch)
                w_asset = out["w_asset"]
                nxt = pos[pi + 1] if pi + 1 < len(pos) else len(dates)
                end = max(idx + 1, nxt)
                tcost = self.cfg.turn_penalty * _turn(w_asset, w_prev)
                log_sum = torch.tensor(0.0, device=self.device)
                short_log_sum = torch.tensor(0.0, device=self.device)
                short_daily = self.cfg.short_penalty * _short(w_asset)
                w_short = -torch.clamp(w_asset, max=0.0)  # positive exposure for shorts
                for j in range(idx, end):
                    rj = self.repository.load_ret(dates[j])
                    gross_j = (rj * w_asset).sum()
                    cost_j = short_daily + (tcost if j == idx else 0.0)
                    net_j = gross_j - cost_j
                    net_j = torch.clamp(net_j, min=-0.999999)
                    log_sum = log_sum + torch.log1p(net_j)

                    # Short-leg net: realized PnL from the short book (positive if underlying falls),
                    # net of short borrow penalty; turnover cost is accounted for in total net.
                    short_gross_j = -(rj * w_short).sum()
                    short_net_j = short_gross_j - short_daily
                    short_net_j = torch.clamp(short_net_j, min=-0.999999)
                    short_log_sum = short_log_sum + torch.log1p(short_net_j)
                net = torch.expm1(log_sum)
                short_net = torch.expm1(short_log_sum)

                sharpe_buf = detach_tail(sharpe_buf + [net], k)
                sharpe = rolling_sharpe(sharpe_buf, eps=self.cfg.sharpe_eps)
                short_buf = detach_tail(short_buf + [short_net], k)
                sharpe_short = rolling_sharpe(short_buf, eps=self.cfg.sharpe_eps)
                ev = explained_variance(out["target"].residuals, target_batch.returns, target_batch.mask)
                loss = -(self.cfg.lambda_sr * (sharpe + self.cfg.short_alpha * sharpe_short) + self.cfg.lambda_var * ev)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.opt.step()

                w_prev = w_asset.detach()
                self.cache.set(date_key, out["target"].residuals.detach().cpu())
                cur_sharpe = float(sharpe.detach().cpu().item())
                if self.best_path is not None and (self.best_sharpe is None or cur_sharpe > self.best_sharpe):
                    self.best_sharpe = cur_sharpe
                    torch.save(self.model.state_dict(), self.best_path)
                bar.set_postfix({"date": str(date_key), "loss": f"{loss.item():.6f}", "sharpe": f"{cur_sharpe:.3f}", "ev": f"{ev.item():.3f}"})


class StatArbRunner:
    def __init__(
        self,
        cfg: AttentionConfig,
        repository: FeatureRepository,
        device: torch.device,
        writer: FactorResultWriter,
        cache: ResidualCache,
        model: StatArbModel,
    ) -> None:
        self.cfg = cfg
        self.repository = repository
        self.device = device
        self.writer = writer
        self.cache = cache
        self.model = model

    def _residual_from_cache(self, date_key: str, batch: FactorBatch) -> torch.Tensor:
        cached = self.cache.get(date_key)
        if cached is not None:
            return cached.to(self.device)
        out = self.model.attention(batch.features, batch.returns, batch.mask)
        self.cache.set(date_key, out.residuals.detach().cpu())
        return out.residuals

    def run(self, dates: Sequence, split: str = "full") -> None:
        lookback = self.cfg.lookback
        w_prev: Optional[torch.Tensor] = None
        w_hold: Optional[torch.Tensor] = None
        reb = _reb_dates(dates[lookback:], self.cfg.rebalance)
        pnl_records = []
        for idx in tqdm(range(lookback, len(dates)), desc=f"infer[{split}]", leave=False):
            window_dates = dates[idx - lookback : idx]
            target_date = dates[idx]
            target_batch = self.repository.load_snapshot(target_date)
            do_reb = pd.Timestamp(target_date) in reb or w_hold is None
            if do_reb:
                window_batches = [self.repository.load_snapshot(d) for d in window_dates]
                for d, b in zip(window_dates, window_batches):
                    self._residual_from_cache(str(d), b)
                out = self.model(window_batches, target_batch)
                w_hold = out["w_asset"].detach()
                self.cache.set(str(target_date), out["target"].residuals.detach().cpu())
            w_asset = w_hold
            gross = float((target_batch.returns * w_asset).sum().item())
            turn = float((_turn(w_asset, w_prev) if do_reb else torch.tensor(0.0, device=self.device)).detach().cpu().item())
            short = float(_short(w_asset).detach().cpu().item())
            cost = self.cfg.turn_penalty * turn + self.cfg.short_penalty * short
            if do_reb:
                w_prev = w_asset
            net = gross - cost
            pnl_records.append({"date": target_date, "gross": gross, "net": net, "cost": cost})
            self.writer.save_portfolio(target_date, target_batch.assets, w_asset, split=split)
        self.writer.save_pnl(pnl_records, split=split)


def _train_test_split(dates: Sequence, split_year: int) -> Tuple[List, List]:
    ordered = pd.to_datetime(pd.Index(dates)).sort_values()
    train = [d for d in ordered if d.year <= split_year]
    test = [d for d in ordered if d.year > split_year]
    return train, test
