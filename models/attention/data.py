from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import ast
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loader import Loader
from models.classify import classify


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    loader_method: str
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
    omega_eps: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None


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
            FeatureGroup("price", "features_price"),
            FeatureGroup("sector", "features_sector"),
        ]

    def _load_group(self, group: FeatureGroup) -> pd.DataFrame:
        loader_fn = getattr(self.loader, group.loader_method)
        df = self._get_multiindex(loader_fn())
        if group.allowed:
            allowed = set(group.allowed)
            keep = [c for c in df.columns if (isinstance(c, tuple) and c[0] in allowed) or c in allowed]
            df = df.loc[:, keep]
        return df.apply(pd.to_numeric, errors="coerce")

    def _get_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            return df
        parsed = []
        tuple_found = False
        for c in df.columns:
            if isinstance(c, str) and c.startswith("(") and c.endswith(")"):
                try:
                    val = ast.literal_eval(c)
                    parsed.append(val)
                    tuple_found = tuple_found or isinstance(val, tuple)
                    continue
                except Exception:
                    pass
            parsed.append(c)
        if tuple_found and all(isinstance(p, tuple) for p in parsed):
            out = df.copy()
            out.columns = pd.MultiIndex.from_tuples(parsed)
            return out
        raise ValueError("Expected 2-level MultiIndex columns for features.")

    def _collect_schema(self, frames: List[pd.DataFrame]) -> Tuple[List[str], List[str]]:
        features = set()
        assets = set()
        for df in frames:
            features.update(str(f) for f in df.columns.get_level_values(0).unique())
            assets.update(str(a) for a in df.columns.get_level_values(1).unique())
        return sorted(features), sorted(assets)

    def _load_and_cache(self) -> None:
        frames = [self._load_group(g) for g in self.groups]
        if not frames:
            raise ValueError("No feature frames loaded.")
        common_dates = frames[0].index
        for df in frames[1:]:
            common_dates = common_dates.intersection(df.index)
        common_dates = common_dates.sort_values()
        frames = [df.reindex(common_dates) for df in frames]

        feature_names, assets = self._collect_schema(frames)
        grid = pd.MultiIndex.from_product([feature_names, assets])
        combined = pd.concat(frames, axis=1).reindex(index=common_dates, columns=grid)
        combined = combined.dropna(how="all")
        combined = combined.fillna(0.0)
        common_dates = combined.index
        if self.return_col not in feature_names:
            raise KeyError(f"Return column {self.return_col} not in features.")

        mat = combined.values.astype("float32")
        n_dates = len(common_dates)
        n_feats = len(feature_names)
        n_assets = len(assets)
        self.feature_tensor = torch.tensor(mat.reshape(n_dates, n_feats, n_assets))
        self.dates = pd.Index(common_dates)
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
