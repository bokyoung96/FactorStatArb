from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from root import FEATURES_PRICE_PARQUET, PRICE_PARQUET
from models.classify import classify as _classify_features


def _parse_tuple_col(name: str):
    if isinstance(name, str) and name.startswith("(") and name.endswith(")"):
        try:
            return ast.literal_eval(name)
        except Exception:
            return name
    return name


def _schema_columns_by_feature(path: Path, features: Sequence[str]) -> dict[str, list[str]]:
    pf = pq.ParquetFile(path)
    wanted = set(features)
    out: dict[str, list[str]] = {f: [] for f in features}
    for raw in pf.schema.names:
        parsed = _parse_tuple_col(raw)
        if not (isinstance(parsed, tuple) and len(parsed) == 2):
            continue
        feat, asset = parsed
        if feat in wanted:
            out[str(feat)].append(raw)
    return out


def _assets_from_schema(cols: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for raw in cols:
        parsed = _parse_tuple_col(raw)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            out.add(str(parsed[1]))
    return out


def read_close(*, assets: Sequence[str] | np.ndarray | None = None) -> pd.DataFrame:
    path = Path(PRICE_PARQUET)
    cols_map = _schema_columns_by_feature(path, ["close"])
    cols = cols_map.get("close", [])
    if assets is not None:
        want = set(map(str, assets))
        cols = [c for c in cols if str(_parse_tuple_col(c)[1]) in want]
    df = pd.read_parquet(path, columns=cols)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    parsed = [_parse_tuple_col(c) for c in df.columns]
    if all(isinstance(c, tuple) and len(c) == 2 for c in parsed):
        df.columns = pd.MultiIndex.from_tuples(parsed, names=["feature", "asset"])
    return df


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex
    assets: np.ndarray
    values: np.ndarray  # (steps, assets, features)
    mask: np.ndarray  # (steps, assets)


def make_panel(*, features: Sequence[str], norm: str, zero_invalid: bool = False) -> Panel:
    derived = {
        "ret1_cs_rank": "ret1",
        "m1_cs_z": "m1",
        "vol20_cs_z": "vol20",
    }
    requested = list(features)
    base_features = [f for f in requested if f not in derived]
    # Ensure required bases for derived features are available.
    for d, base in derived.items():
        if d in requested and base not in base_features:
            base_features.append(base)

    path = Path(FEATURES_PRICE_PARQUET)
    cols_by_feat = _schema_columns_by_feature(path, base_features)
    if not any(cols_by_feat.values()):
        raise ValueError("features_price parquet has no matching feature columns.")

    assets_sets = []
    for f in base_features:
        cols = cols_by_feat.get(f, [])
        if not cols:
            raise KeyError(f"Feature {f!r} not found in features_price parquet.")
        assets_sets.append(_assets_from_schema(cols))
    common_assets = sorted(set.intersection(*assets_sets)) if assets_sets else []
    if not common_assets:
        raise ValueError("No common assets across selected features.")

    # Use the first feature as the date index anchor.
    anchor_cols = [c for c in cols_by_feat[base_features[0]] if str(_parse_tuple_col(c)[1]) in set(common_assets)]
    anchor_df = pd.read_parquet(path, columns=anchor_cols)
    dates = pd.DatetimeIndex(pd.to_datetime(anchor_df.index)).sort_values()
    steps = len(dates)
    n_assets = len(common_assets)
    n_features = len(requested)
    vals = np.empty((steps, n_assets, n_features), dtype=np.float32)

    cache: dict[str, pd.DataFrame] = {}

    def load_feat(feat: str) -> pd.DataFrame:
        if feat in cache:
            return cache[feat]
        cols = [c for c in cols_by_feat[feat] if str(_parse_tuple_col(c)[1]) in set(common_assets)]
        df = pd.read_parquet(path, columns=cols).reindex(index=dates)
        df.index = pd.to_datetime(df.index)
        df.columns = [str(_parse_tuple_col(c)[1]) for c in df.columns]
        df = df.reindex(columns=common_assets)
        cache[feat] = df
        return df

    def cs_rank01(df: pd.DataFrame) -> pd.DataFrame:
        # Rank within each date across assets, scaled to [0, 1].
        return df.rank(axis=1, pct=True, method="average")

    def cs_z(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        x = df.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(x)
        cnt = finite.sum(axis=1, keepdims=True).astype(np.float64)
        s = np.where(finite, x, 0.0).sum(axis=1, keepdims=True)
        mean = np.zeros_like(s, dtype=np.float64)
        np.divide(s, cnt, out=mean, where=cnt > 0)
        centered = np.where(finite, x - mean, 0.0)
        ss = (centered**2).sum(axis=1, keepdims=True)
        var = np.zeros_like(ss, dtype=np.float64)
        np.divide(ss, cnt, out=var, where=cnt > 0)
        std = np.sqrt(var)
        denom = np.where((std < eps) | (cnt <= 0), 1.0, std)
        z = centered / denom
        return pd.DataFrame(z, index=df.index, columns=df.columns)

    for fi, f in enumerate(requested):
        if f in derived:
            base = derived[f]
            base_df = load_feat(base)
            if f == "ret1_cs_rank":
                out_df = cs_rank01(base_df)
            elif f == "m1_cs_z":
                out_df = cs_z(base_df)
            elif f == "vol20_cs_z":
                out_df = cs_z(base_df)
            else:
                raise KeyError(f"Unknown derived feature: {f}")
        else:
            out_df = load_feat(f)
        vals[:, :, fi] = out_df.to_numpy(dtype=np.float32, copy=True)

    # Feature-specific transforms aligned with FactorStatArb normalization rules.
    to_norm, _, to_log = _classify_features(requested, include_log1p=True)
    idx_by_feat = {str(f).lower(): i for i, f in enumerate(requested)}
    cs_exempt = {"ret1_cs_rank", "m1_cs_z", "vol20_cs_z"}
    norm_idx = [idx_by_feat[str(f).lower()] for f in to_norm if str(f).lower() in idx_by_feat and str(f).lower() not in cs_exempt]
    log_idx = [idx_by_feat[str(f).lower()] for f in to_log if str(f).lower() in idx_by_feat and str(f).lower() not in cs_exempt]

    vals_f64 = vals.astype(np.float64, copy=False)
    if log_idx:
        x = vals_f64[:, :, log_idx]
        vals_f64[:, :, log_idx] = np.log1p(np.clip(x, a_min=0.0, a_max=None))

    if norm == "asset":
        if norm_idx:
            sub = vals_f64[:, :, norm_idx]
            finite = np.isfinite(sub)
            cnt = finite.sum(axis=0, keepdims=True).astype(np.float64)  # (1, assets, feats)
            s = np.where(finite, sub, 0.0).sum(axis=0, keepdims=True)
            mean = np.zeros_like(s, dtype=np.float64)
            np.divide(s, cnt, out=mean, where=cnt > 0)
            centered = np.where(finite, sub - mean, 0.0)
            ss = (centered**2).sum(axis=0, keepdims=True)
            var = np.zeros_like(ss, dtype=np.float64)
            np.divide(ss, cnt, out=var, where=cnt > 0)
            std = np.sqrt(var)
            denom = np.where((std == 0.0) | (cnt <= 0), 1.0, std)
            vals_f64[:, :, norm_idx] = centered / denom
    elif norm == "cross":
        if norm_idx:
            sub = vals_f64[:, :, norm_idx]
            finite = np.isfinite(sub)
            cnt = finite.sum(axis=1, keepdims=True).astype(np.float64)  # (steps, 1, feats)
            s = np.where(finite, sub, 0.0).sum(axis=1, keepdims=True)
            mean = np.zeros_like(s, dtype=np.float64)
            np.divide(s, cnt, out=mean, where=cnt > 0)
            centered = np.where(finite, sub - mean, 0.0)
            ss = (centered**2).sum(axis=1, keepdims=True)
            var = np.zeros_like(ss, dtype=np.float64)
            np.divide(ss, cnt, out=var, where=cnt > 0)
            std = np.sqrt(var)
            denom = np.where((std == 0.0) | (cnt <= 0), 1.0, std)
            vals_f64[:, :, norm_idx] = centered / denom
    elif norm != "none":
        raise ValueError(f"Unknown norm: {norm!r}")

    mask = np.isfinite(vals_f64).all(axis=2)
    if zero_invalid:
        mask &= vals_f64.any(axis=2)
    vals_out = np.nan_to_num(vals_f64, nan=0.0).astype(np.float32)
    return Panel(dates=dates, assets=np.array(common_assets, dtype=object), values=vals_out, mask=mask)
