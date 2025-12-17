from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable
import sys

import pandas as pd

# Ensure repo root is on sys.path for root.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from root import factor_dir_freq


def _load_parquet_dir(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df


@dataclass(frozen=True)
class PortfolioWeightsStore:
    """
    Loader for portfolio_weights parquet outputs.
    """

    base_dir: Path = field(default_factory=lambda: factor_dir_freq("W") / "portfolio_weights")

    @classmethod
    def for_rebalance(cls, rebalance: str) -> "PortfolioWeightsStore":
        return cls(base_dir=factor_dir_freq(rebalance) / "portfolio_weights")

    def load(self) -> pd.DataFrame:
        """
        Returns DataFrame indexed by (date, split, asset) with column 'weight'.
        """
        return _load_parquet_dir(self.base_dir)

    def pivot_by_asset(self, splits: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Returns weights: index=date, columns=asset, values=weight.
        If splits is None, uses all splits (drop split level).
        """
        df = self.load()
        if df.empty:
            return df
        if splits is not None:
            df = df[df.index.get_level_values("split").isin(splits)]
        s = df["weight"]
        if "split" in s.index.names:
            s = s.reset_index("split", drop=True)
        if s.index.duplicated().any():
            raise ValueError("Duplicate (date, asset) rows found.")
        out = s.unstack("asset").sort_index()
        out.index.name = "date"
        return out

    @property
    def weight(self) -> pd.DataFrame:
        """
        Returns weights: index=date, columns=asset, values=weight.
        Uses all splits (drop split level). Raises if (date, asset) duplicates exist.
        """
        return self.pivot_by_asset(splits=None)


@dataclass(frozen=True)
class PnLStore:
    """
    Loader for pnl parquet outputs.
    """

    base_dir: Path = field(default_factory=lambda: factor_dir_freq("D") / "pnl")

    @classmethod
    def for_rebalance(cls, rebalance: str) -> "PnLStore":
        return cls(base_dir=factor_dir_freq(rebalance) / "pnl")

    def load(self) -> pd.DataFrame:
        """
        Returns DataFrame indexed by (date, split) with columns ['gross','net','cost'].
        """
        return _load_parquet_dir(self.base_dir)

    def get_split(self, split: str) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return df
        return df.xs(split, level="split", drop_level=False)


if __name__ == "__main__":
    w_store = PortfolioWeightsStore()
    w = w_store.load()
    print("weights shape:", w.shape)
    weight = w_store.weight
    print("weight shape:", weight.shape)
    pnl_store = PnLStore()
    pnl = pnl_store.load()
    print("pnl shape:", pnl.shape)
