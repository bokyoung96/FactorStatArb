from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loader import Loader
from models.classify import classify
from models.util import DeviceSelector
from root import (
    FACTOR_DIR,
    FACTOR_PRED_PARQUET,
    FACTOR_RES_PARQUET,
    FACTOR_RETURNS_PARQUET,
    FACTOR_WEIGHTS_PARQUET,
)


@dataclass(frozen=True)
class AttentionConfig:
    num_factors: int = 32
    emb_dim: int = 32
    return_col: str = "ret1"
    split_year: int = 2020
    normalize: bool = True
    train_epochs: int = 20
    train_batch_size: int = 32
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-4

    @classmethod
    def from_file(cls, path: Path) -> "AttentionConfig":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        kwargs = {}
        for key in (
            "num_factors",
            "emb_dim",
            "return_col",
            "split_year",
            "normalize",
            "train_epochs",
            "train_batch_size",
            "train_lr",
            "train_weight_decay",
        ):
            if key in raw:
                kwargs[key] = raw[key]
        return cls(**kwargs)


@dataclass(frozen=True)
class FactorBatch:
    features: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor
    assets: pd.Index
    normalize_idx: Sequence[int]


@dataclass(frozen=True)
class FactorResult:
    weights: torch.Tensor
    factor_returns: torch.Tensor
    predicted: torch.Tensor
    residuals: torch.Tensor


class FeatureNormalizer:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._normalize_idx: Optional[torch.Tensor] = None

    @property
    def fitted(self) -> bool:
        return self._mean is not None and self._std is not None

    def fit(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> "FeatureNormalizer":
        normalize_idx = self._normalize_idx
        feats = self._select_valid(features, mask)
        if normalize_idx is not None:
            feats = feats[:, normalize_idx]
        self._mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True)
        self._std = torch.clamp(std, min=self.eps)
        return self

    def transform(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        normalize_idx: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform.")
        idx = (
            torch.as_tensor(normalize_idx, dtype=torch.long, device=features.device)
            if normalize_idx is not None
            else self._normalize_idx.to(features.device) if self._normalize_idx is not None else None
        )
        if (idx is None or idx.numel() == 0) and self._normalize_idx is not None and self._normalize_idx.numel() > 0:
            idx = self._normalize_idx.to(features.device)
        feats = features
        if idx is not None and idx.numel() > 0:
            feats = feats.clone()
            feats[:, idx] = (feats[:, idx] - self._mean.to(features.device)) / self._std.to(features.device)
        if mask is not None:
            feats = feats * mask.unsqueeze(-1)
        return feats

    def fit_transform(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fit(features, mask).transform(features, mask)

    def set_normalize_idx(self, normalize_idx: Sequence[int]) -> None:
        self._normalize_idx = torch.as_tensor(list(normalize_idx), dtype=torch.long)

    def _select_valid(self, features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return features
        selector = mask > 0
        return features[selector] if selector.any() else features


class FeatureRepository:
    def __init__(
        self,
        loader: Loader,
        device: torch.device,
        return_col: str = "ret1",
        master_assets: Optional[Sequence[str]] = None,
    ) -> None:
        self.loader = loader
        self.device = device
        self.return_col = return_col
        self.master_assets = pd.Index(master_assets) if master_assets is not None else pd.Index(self.loader.universe().columns)

    def available_dates(self) -> pd.Index:
        return pd.Index(self.loader.features_price().index)

    def latest_date(self):
        return self.available_dates()[-1]

    def load_snapshot(self, date) -> FactorBatch:
        price_feats = self._to_matrix(self.loader.features_price(), date)
        fundamentals = self._to_matrix(self.loader.features_fundamentals(), date)
        consensus = self._to_matrix(self.loader.features_consensus(), date)
        sector = self._to_matrix(self.loader.features_sector(), date)
        mask_series = self.loader.universe().loc[date]

        assets = self.master_assets.intersection(price_feats.index)
        if assets.empty:
            raise ValueError("No overlapping assets between features and universe mask.")

        features_df = (
            pd.concat([price_feats, fundamentals, consensus, sector], axis=1)
            .reindex(assets)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        if self.return_col not in price_feats.columns:
            raise KeyError(f"Return column '{self.return_col}' not found in price features.")
        returns_series = price_feats.reindex(assets)[self.return_col].astype(float).fillna(0.0)
        mask_vals = mask_series.reindex(assets).fillna(0).astype(float)
        to_norm, _ = classify(features_df.columns)
        normalize_idx = [i for i, c in enumerate(features_df.columns) if c in to_norm]

        return FactorBatch(
            features=torch.tensor(features_df.values, dtype=torch.float32, device=self.device),
            returns=torch.tensor(returns_series.values, dtype=torch.float32, device=self.device),
            mask=torch.tensor(mask_vals.values, dtype=torch.float32, device=self.device),
            assets=assets,
            normalize_idx=normalize_idx,
        )

    def _to_matrix(self, df: pd.DataFrame, date) -> pd.DataFrame:
        snap = df.loc[date]
        if isinstance(snap, pd.Series):
            if isinstance(df.columns, pd.MultiIndex):
                return snap.unstack(0)
            return snap.to_frame().T
        if isinstance(df.columns, pd.MultiIndex):
            return snap.unstack(0)
        return snap


class FactorResultWriter:
    def __init__(self, output_dir: Path = FACTOR_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _append_parquet(self, df: pd.DataFrame, path: Path) -> None:
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(path)

    def save(self, date, assets: Sequence[str], result: FactorResult, split: str = "full") -> None:
        factor_cols = [f"f{i}" for i in range(result.weights.size(0))]

        weights_df = pd.DataFrame(
            result.weights.detach().cpu().numpy().T, index=pd.Index(assets, name="asset"), columns=factor_cols
        )
        weights_df["date"] = date
        weights_df["split"] = split
        weights_df = weights_df.reset_index().set_index(["date", "split", "asset"])
        self._append_parquet(weights_df, FACTOR_WEIGHTS_PARQUET)

        fr_df = pd.DataFrame(
            [result.factor_returns.detach().cpu().numpy()], index=pd.Index([date], name="date"), columns=factor_cols
        )
        fr_df["split"] = split
        fr_df = fr_df.set_index("split", append=True)
        self._append_parquet(fr_df, FACTOR_RETURNS_PARQUET)

        pred_df = pd.DataFrame(
            {"predicted": result.predicted.detach().cpu().numpy(), "residual": result.residuals.detach().cpu().numpy()},
            index=pd.Index(assets, name="asset"),
        )
        pred_df["date"] = date
        pred_df["split"] = split
        pred_df = pred_df.reset_index().set_index(["date", "split", "asset"])
        self._append_parquet(pred_df[["predicted"]], FACTOR_PRED_PARQUET)
        self._append_parquet(pred_df[["residual"]], FACTOR_RES_PARQUET)


class AttentionFactorLayer(nn.Module):
    def __init__(self, num_features: int, num_factors: int = 8, emb_dim: int = 32) -> None:
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
        scores = (self.Q @ x_emb.T) / math.sqrt(self.Q.size(1))
        if mask is not None:
            scores = scores + (mask.to(features.device).unsqueeze(0) - 1) * 1e9
        scores = scores - scores.max(dim=-1, keepdim=True).values
        weights = F.softmax(scores, dim=-1)
        factor_returns = weights @ returns
        predicted = weights.T @ factor_returns
        residual = returns - predicted
        return weights, factor_returns, predicted, residual


class AttentionFactorModel(nn.Module):
    def __init__(self, num_features: int, num_factors: int = 32, emb_dim: int = 32) -> None:
        super().__init__()
        self.layer = AttentionFactorLayer(num_features, num_factors, emb_dim)

    def forward(
        self,
        features: torch.Tensor,
        returns: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights_list = []
        fr_list = []
        pred_list = []
        res_list = []
        for t in range(features.size(0)):
            m_t = None if mask is None else mask[t]
            w, fr, pred, res = self.layer(features[t], returns[t], m_t)
            weights_list.append(w)
            fr_list.append(fr)
            pred_list.append(pred)
            res_list.append(res)
        factor_weights_all = torch.stack(weights_list, dim=0)
        factor_returns_all = torch.stack(fr_list, dim=0)
        predicted_all = torch.stack(pred_list, dim=0)
        residuals_all = torch.stack(res_list, dim=0)
        return factor_weights_all, factor_returns_all, predicted_all, residuals_all


class AttentionPipeline:
    def __init__(
        self,
        model: AttentionFactorLayer,
        repository: FeatureRepository,
        normalizer: FeatureNormalizer,
        writer: Optional[FactorResultWriter] = None,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.repository = repository
        self.normalizer = normalizer
        self.writer = writer
        self.normalize = normalize

    def _apply_normalization(self, batch: FactorBatch) -> torch.Tensor:
        if not self.normalize:
            return batch.features
        if not self.normalizer.fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        return self.normalizer.transform(batch.features, batch.mask, batch.normalize_idx)

    @classmethod
    def from_loader(
        cls,
        loader: Loader,
        device: Optional[torch.device] = None,
        num_factors: int = 32,
        emb_dim: int = 32,
        return_col: str = "ret1",
        normalize: bool = True,
    ) -> "AttentionPipeline":
        device = device or DeviceSelector().resolve()
        repository = FeatureRepository(loader=loader, device=device, return_col=return_col)
        seed_batch = repository.load_snapshot(repository.latest_date())
        model = AttentionFactorLayer(num_features=seed_batch.features.size(1), num_factors=num_factors, emb_dim=emb_dim)
        model = model.to(device)
        writer = FactorResultWriter()
        normalizer = FeatureNormalizer()
        return cls(model=model, repository=repository, normalizer=normalizer, writer=writer, normalize=normalize)

    def fit_normalizer(self, dates: Sequence) -> None:
        loader = self.repository.loader
        price = loader.features_price()
        fundamentals = loader.features_fundamentals()
        consensus = loader.features_consensus()
        sector = loader.features_sector()
        combined = pd.concat([price, fundamentals, consensus, sector], axis=1)
        if not isinstance(combined.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns with asset level for stacking.")
        X = combined.stack(level=0, future_stack=True)  # (date, asset) index, feature columns
        univ = loader.universe()
        valid_idx = (univ.stack() > 0)
        valid_mask = valid_idx.reindex(X.index, fill_value=False)
        if not valid_mask.any():
            logging.warning("Universe mask empty after alignment; using all rows for normalization.")
            X_valid = X
        else:
            X_valid = X[valid_mask]
        to_norm, _ = classify(X.columns)
        normalize_idx = [i for i, c in enumerate(X.columns) if c in to_norm]
        self.normalizer.set_normalize_idx(normalize_idx)
        if not normalize_idx:
            raise ValueError("No columns selected for normalization.")
        target = X_valid.iloc[:, normalize_idx].apply(pd.to_numeric, errors="coerce")
        if target.empty:
            logging.warning("Normalization columns empty after filtering; using unfiltered data.")
            target = X.iloc[:, normalize_idx].apply(pd.to_numeric, errors="coerce")
        mean = target.mean(axis=0)
        std = target.std(axis=0).clip(lower=self.normalizer.eps)
        self.normalizer._mean = torch.tensor(mean.values, dtype=torch.float32).unsqueeze(0)
        self.normalizer._std = torch.tensor(std.values, dtype=torch.float32).unsqueeze(0)

    def run_on_date(self, date, save: bool = True, split: str = "full") -> FactorResult:
        batch = self.repository.load_snapshot(date)
        feats = self._apply_normalization(batch)
        weights, fr, pred, res = self.model(feats, batch.returns, batch.mask)
        result = FactorResult(weights=weights, factor_returns=fr, predicted=pred, residuals=res)
        if save and self.writer:
            self.writer.save(date, batch.assets, result, split=split)
        return result

    def run_dates(self, dates: Sequence, save: bool = True, split: str = "full") -> None:
        for date in tqdm(dates, desc=f"run_dates[{split}]"):
            self.run_on_date(date, save=save, split=split)

    def run_latest(self, save: bool = True, split: str = "full") -> FactorResult:
        return self.run_on_date(self.repository.latest_date(), save=save, split=split)

    def train_factors(
        self,
        dates: Sequence,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
    ) -> None:
        if self.normalize and not self.normalizer.fitted:
            raise RuntimeError("Normalizer must be fitted before training when normalize=True.")
        if not dates:
            logging.info("No training dates provided; skipping training.")
            return
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        device = next(self.model.parameters()).device
        for epoch in range(epochs):
            perm = torch.randperm(len(dates)).tolist()
            epoch_loss = 0.0
            epoch_count = 0
            for start in range(0, len(perm), batch_size):
                idxs = perm[start : start + batch_size]
                loss_sum = torch.tensor(0.0, device=device)
                count = 0
                optimizer.zero_grad()
                for idx in idxs:
                    date = dates[idx]
                    batch = self.repository.load_snapshot(date)
                    feats = self._apply_normalization(batch)
                    _, _, predicted, _ = self.model(feats, batch.returns, batch.mask)
                    valid = batch.mask > 0
                    if not valid.any():
                        continue
                    loss_sum = loss_sum + F.mse_loss(
                        predicted[valid], batch.returns[valid], reduction="sum"
                    )
                    count += int(valid.sum().item())
                if count == 0:
                    continue
                loss = loss_sum / count
                loss.backward()
                optimizer.step()
                epoch_loss += loss_sum.detach().item()
                epoch_count += count
            if epoch_count > 0:
                logging.info("Epoch %d average loss: %.6f", epoch + 1, epoch_loss / epoch_count)


def _train_test_split(dates: Sequence, split_year: int) -> Tuple[Sequence, Sequence]:
    if not dates:
        return [], []
    ordered = pd.to_datetime(pd.Index(dates)).sort_values()
    train = [d for d in ordered if d.year <= split_year]
    test = [d for d in ordered if d.year > split_year]
    return train, test


def main(config_path: Optional[Path] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg_path = config_path or Path(__file__).with_name("config.json")
    cfg = AttentionConfig.from_file(cfg_path)
    device = DeviceSelector().resolve()
    print(DeviceSelector().summary("attention"))

    pipeline = AttentionPipeline.from_loader(
        loader=Loader(),
        device=device,
        num_factors=cfg.num_factors,
        emb_dim=cfg.emb_dim,
        return_col=cfg.return_col,
        normalize=cfg.normalize,
    )

    dates = list(pipeline.repository.available_dates())
    train_dates, test_dates = _train_test_split(dates, cfg.split_year)

    if cfg.normalize and train_dates:
        pipeline.fit_normalizer(train_dates)

    if train_dates:
        pipeline.train_factors(
            train_dates,
            epochs=cfg.train_epochs,
            batch_size=cfg.train_batch_size,
            lr=cfg.train_lr,
            weight_decay=cfg.train_weight_decay,
        )
        pipeline.run_dates(train_dates, save=True, split="train")
    if test_dates:
        pipeline.run_dates(test_dates, save=True, split="test")


if __name__ == "__main__":
    main()
