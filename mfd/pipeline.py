from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import gc
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from tqdm import tqdm

from .io import make_panel, read_close
from .params import TransformerConfig


@dataclass(frozen=True)
class WindowIndex:
    start: np.ndarray  # (n,) int32 start positions in panel.dates
    asset_idx: np.ndarray  # (n,) int32 asset positions in panel.assets
    targets: np.ndarray  # (n,) float32
    dates: np.ndarray  # (n,) datetime64[ns] of target date (t_idx)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {"start": self.start, "asset_idx": self.asset_idx, "targets": self.targets, "dates": self.dates},
            tmp,
            pickle_protocol=4,
        )
        tmp.replace(path)
        return path

    @staticmethod
    def load(path: Path) -> "WindowIndex":
        d = torch.load(path, weights_only=False)
        return WindowIndex(d["start"], d["asset_idx"], d["targets"], d["dates"])


def _win_path(cfg: TransformerConfig) -> Path:
    key = json.dumps({"features": list(cfg.features), "norm": cfg.norm}, sort_keys=True).encode("utf-8")
    h = hashlib.md5(key).hexdigest()[:10]
    return Path(cfg.cache_dir) / f"win_{cfg.mode}_lb{cfg.lookback}_hz{cfg.horizon}_s{cfg.stride}_{h}.pt"


def build_index(cfg: TransformerConfig, *, panel) -> WindowIndex:
    close_df = read_close(assets=panel.assets)
    close = close_df.xs("close", level=0, axis=1) if isinstance(close_df.columns, pd.MultiIndex) else close_df
    close = close.reindex(index=panel.dates, columns=panel.assets.astype(str))
    close_vals = close.to_numpy(dtype=np.float64, copy=False)

    steps, n_assets, n_features = panel.values.shape
    max_start = steps - cfg.lookback - cfg.horizon
    if max_start <= 0:
        raise ValueError("Not enough history for lookback+horizon.")

    min_valid = int(np.ceil(cfg.lookback * float(cfg.min_valid)))
    estimated_windows = ((max_start + 1) // cfg.stride) * n_assets

    starts = np.empty(estimated_windows, dtype=np.int32)
    asset_idx = np.empty(estimated_windows, dtype=np.int32)
    targets = np.empty(estimated_windows, dtype=np.float32)
    dates = np.empty(estimated_windows, dtype="datetime64[ns]")
    counter = 0

    for s in tqdm(range(0, max_start + 1, cfg.stride), desc="mfd windows", leave=False):
        e = s + cfg.lookback
        t_idx = e - 1

        mask = panel.mask[s:e]
        counts = mask.sum(axis=0)

        base = close_vals[t_idx]
        fut = close_vals[t_idx + cfg.horizon]
        fut_ok = np.isfinite(base) & np.isfinite(fut) & (base != 0.0)
        ok = (counts >= min_valid) & fut_ok
        idx = np.where(ok)[0]
        if idx.size < cfg.min_assets:
            continue
        n_valid = idx.size

        ret = (fut[idx] / base[idx]) - 1.0
        if cfg.label_type == "classification":
            y = (ret > cfg.threshold).astype(np.float32)
        else:
            y = ret.astype(np.float32)
        starts[counter : counter + n_valid] = np.int32(s)
        asset_idx[counter : counter + n_valid] = idx.astype(np.int32, copy=False)
        targets[counter : counter + n_valid] = y
        dates[counter : counter + n_valid] = panel.dates[t_idx].to_datetime64()
        counter += n_valid

    if counter == 0:
        raise RuntimeError("No windows created; relax min_assets/min_valid or check data coverage.")

    return WindowIndex(
        start=starts[:counter].copy(),
        asset_idx=asset_idx[:counter].copy(),
        targets=targets[:counter].copy(),
        dates=dates[:counter].copy(),
    )


class StockDataset(TorchDataset):
    def __init__(self, panel, idx: WindowIndex, lookback: int):
        self.panel = panel
        self.idx = idx
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return int(self.idx.targets.shape[0])

    def __getitem__(self, i: int) -> Dict:
        s = int(self.idx.start[i])
        a = int(self.idx.asset_idx[i])
        x = self.panel.values[s : s + self.lookback, a, :]  # (lookback, features)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(int(self.idx.targets[i]), dtype=torch.long),
            "date": str(self.idx.dates[i]),
            "asset": str(self.panel.assets[a]),
        }


@dataclass(frozen=True)
class Bundle:
    panel: any
    idx: WindowIndex


def load_bundle(cfg: TransformerConfig) -> Bundle:
    panel = make_panel(features=cfg.features, norm=cfg.norm)
    path = _win_path(cfg)
    if path.exists():
        try:
            idx = WindowIndex.load(path)
        except Exception:
            path.unlink(missing_ok=True)
            idx = build_index(cfg, panel=panel)
            idx.save(path)
    else:
        idx = build_index(cfg, panel=panel)
        idx.save(path)
    return Bundle(panel=panel, idx=idx)


def get_loaders(
    cfg: TransformerConfig,
    *,
    batch: int,
    workers: int = 0,
) -> Dict[str, DataLoader]:
    bundle = load_bundle(cfg)
    ds = StockDataset(bundle.panel, bundle.idx, lookback=cfg.lookback)
    return {"all": DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)}
