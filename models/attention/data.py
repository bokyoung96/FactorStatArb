from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Iterable

import pandas as pd
import torch

from loader import Loader
from models.classify import classify
from root import FEATURES_PRICE_PARQUET, FEATURES_SECTOR_PARQUET


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    path: Path
    allowed: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class FactorBatch:
    features: torch.Tensor  # (assets, feats)
    returns: torch.Tensor  # (assets,)
    mask: torch.Tensor  # (assets,)
    assets: pd.Index
    norm_idx: Sequence[int]
    log_idx: Sequence[int]
    mean: torch.Tensor  # (feats,)
    std: torch.Tensor  # (feats,)


@dataclass(frozen=True)
class FactorResult:
    weights: torch.Tensor
    factor_returns: torch.Tensor
    predicted: torch.Tensor
    residuals: torch.Tensor


class FeatureRepository:
    def __init__(
        self,
        loader: Loader,
        device: torch.device,
        return_col: str,
        groups: Optional[Sequence[FeatureGroup]] = None,
    ) -> None:
        self.device = device
        self.loader = loader
        self.return_col = return_col
        self.groups = list(groups) if groups else self._default_groups()
        self._load_and_cache()

    def _default_groups(self) -> List[FeatureGroup]:
        return [
            FeatureGroup("price", FEATURES_PRICE_PARQUET),
            FeatureGroup("sector", FEATURES_SECTOR_PARQUET),
        ]

    def _load_group(self, group: FeatureGroup) -> pd.DataFrame:
        df = pd.read_parquet(group.path)
        df = df.apply(pd.to_numeric, errors="coerce")
        if group.allowed:
            keep = [c for c in df.columns if (isinstance(c, tuple) and c[0] in group.allowed) or c in group.allowed]
            df = df.loc[:, keep]
        return df

    def _collect_schema(self, frames: List[pd.DataFrame]) -> Tuple[List[str], List[str]]:
        features = set()
        assets = set()
        for df in frames:
            for c in df.columns:
                if isinstance(c, tuple):
                    features.add(str(c[0]))
                    assets.add(str(c[1]))
        return sorted(features), sorted(assets)

    def _load_and_cache(self) -> None:
        frames = [self._load_group(g) for g in self.groups]
        combined = pd.concat(frames, axis=1).fillna(0.0)
        feature_names, assets = self._collect_schema(frames)
        cols = [(f, a) for f in feature_names for a in assets if (f, a) in combined.columns]
        if self.return_col not in feature_names:
            raise KeyError(f"Return column {self.return_col} not in features.")

        mat = combined.loc[:, cols].values.astype("float32")
        n_dates = len(combined.index)
        n_feats = len(feature_names)
        n_assets = len(assets)
        self.feature_tensor = torch.tensor(mat.reshape(n_dates, n_feats, n_assets))
        self.dates = pd.Index(combined.index)
        self.assets_order = assets
        self.feature_names = feature_names
        self.date_to_pos = {d: i for i, d in enumerate(self.dates)}
        self.return_idx = feature_names.index(self.return_col)
        self.return_tensor = self.feature_tensor[:, self.return_idx, :]

        univ = self.loader.universe().reindex(index=self.dates, columns=self.assets_order).fillna(0.0)
        self.mask_tensor = torch.tensor(univ.values.astype("float32"))

        to_norm, _, to_log = classify(feature_names, include_log1p=True)
        self.norm_idx = [feature_names.index(f) for f in to_norm]
        self.log_idx = [feature_names.index(f) for f in to_log if f in feature_names]

    def available_dates(self) -> pd.Index:
        return self.dates

    def latest_date(self):
        return self.dates[-1]

    def _compute_cs_stats(self, feats: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = mask.unsqueeze(1)
        count = valid.sum(dim=0).clamp_min(1.0)
        mean = (feats * valid).sum(dim=0) / count
        var = ((feats - mean) ** 2 * valid).sum(dim=0) / count
        std = torch.sqrt(var).clamp_min(1e-6)
        return mean, std

    def load_snapshot(self, date) -> FactorBatch:
        if date not in self.date_to_pos:
            raise KeyError(f"Date {date} not found.")
        pos = self.date_to_pos[date]
        feats = self.feature_tensor[pos].permute(1, 0)
        mask = torch.nan_to_num(self.mask_tensor[pos], nan=0.0, posinf=0.0, neginf=0.0)
        returns = self.return_tensor[pos]

        feats = feats.clone()
        if self.log_idx:
            feats[:, self.log_idx] = torch.log1p(torch.clamp(feats[:, self.log_idx], min=0))
        mean, std = self._compute_cs_stats(feats, mask)

        return FactorBatch(
            features=feats.to(self.device),
            returns=returns.to(self.device),
            mask=mask.to(self.device),
            assets=pd.Index(self.assets_order, name="asset"),
            norm_idx=self.norm_idx,
            log_idx=self.log_idx,
            mean=mean.to(self.device),
            std=std.to(self.device),
        )
