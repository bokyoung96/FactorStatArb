from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from attention.data import FactorResult
from root import FACTOR_DIR


class FactorResultWriter:
    def __init__(self, output_dir: Path = FACTOR_DIR) -> None:
        base = Path(output_dir)
        self.weights_dir = base / "weights"
        self.returns_dir = base / "returns"
        self.pred_dir = base / "predicted"
        self.res_dir = base / "residuals"
        self.portfolio_dir = base / "portfolio_weights"
        self.pnl_dir = base / "pnl"
        for d in (self.weights_dir, self.returns_dir, self.pred_dir, self.res_dir, self.portfolio_dir, self.pnl_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _append(self, df: pd.DataFrame, path: Path) -> None:
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(path)

    def save(self, date, assets: Sequence[str], result: FactorResult, split: str) -> None:
        self.save_batch([(date, assets, result)], split=split)

    def save_batch(self, items: Sequence[tuple], split: str) -> None:
        buckets: Dict[int, List[Tuple]] = {}
        for date, assets, result in items:
            ts = pd.to_datetime(date)
            year = ts.year
            buckets.setdefault(year, []).append((ts, assets, result))

        for year, group in buckets.items():
            self._save_year(group, split=split, year=year)

    def _save_year(self, group: List[Tuple], split: str, year: int) -> None:
        dates = [g[0] for g in group]
        assets = group[0][1]
        assets_list = list(assets)
        n_items = len(group)
        n_assets = len(assets_list)
        factor_cols = [f"f{i}" for i in range(group[0][2].weights.size(0))]

        weights_t = torch.stack([g[2].weights for g in group])  # (n, factors, assets)
        fr_t = torch.stack([g[2].factor_returns for g in group])  # (n, factors)
        pred_t = torch.stack([g[2].predicted for g in group])  # (n, assets)
        res_t = torch.stack([g[2].residuals for g in group])  # (n, assets)

        weights_np = weights_t.detach().cpu().numpy().reshape(n_items * n_assets, -1)
        fr_np = fr_t.detach().cpu().numpy()
        pred_np = pred_t.detach().cpu().numpy().reshape(n_items * n_assets)
        res_np = res_t.detach().cpu().numpy().reshape(n_items * n_assets)

        date_rep = pd.Index(dates).repeat(n_assets)
        asset_rep = pd.Index(assets_list * n_items, name="asset")

        weights_df = pd.DataFrame(weights_np, index=asset_rep, columns=factor_cols)
        weights_df["date"] = date_rep.values
        weights_df["split"] = split
        weights_df = weights_df.reset_index().set_index(["date", "split", "asset"])

        fr_df = pd.DataFrame(fr_np, index=pd.Index(dates, name="date"), columns=factor_cols)
        fr_df["split"] = split
        fr_df = fr_df.set_index("split", append=True)

        pred_res_df = pd.DataFrame({"predicted": pred_np, "residual": res_np}, index=asset_rep)
        pred_res_df["date"] = date_rep.values
        pred_res_df["split"] = split
        pred_res_df = pred_res_df.reset_index().set_index(["date", "split", "asset"])

        self._append(weights_df, self.weights_dir / f"{year}.parquet")
        self._append(fr_df, self.returns_dir / f"{year}.parquet")
        self._append(pred_res_df, self.pred_dir / f"{year}.parquet")
        self._append(pred_res_df[["residual"]], self.res_dir / f"{year}.parquet")

    def save_portfolio(self, date, assets: Sequence[str], weights: torch.Tensor, split: str) -> None:
        ts = pd.to_datetime(date)
        year = ts.year
        weights_np = weights.detach().cpu().numpy()
        df = pd.DataFrame({"weight": weights_np}, index=pd.Index(list(assets), name="asset"))
        df["date"] = ts
        df["split"] = split
        df = df.reset_index().set_index(["date", "split", "asset"])
        self._append(df, self.portfolio_dir / f"{year}.parquet")

    def save_pnl(self, records: Sequence[Dict], split: str) -> None:
        if not records:
            return
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["split"] = split
        df = df.set_index(["date", "split"])
        years = df.index.get_level_values(0).year.unique()
        for y in years:
            chunk = df[df.index.get_level_values(0).year == y]
            self._append(chunk, self.pnl_dir / f"{y}.parquet")
