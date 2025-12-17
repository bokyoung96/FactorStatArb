from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from root import RAW_DIR, FACTOR_DIR


BT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PxStore:
    csv_path: Path = RAW_DIR / "qw_adj_c.csv"
    pq_path: Path = BT_DIR / "px_adj_c.parquet"

    def ensure(self) -> Path:
        return self.build(force=False)

    def build(self, *, force: bool = False, cols: Optional[Iterable[str]] = None) -> Path:
        if self.pq_path.exists() and not force:
            return self.pq_path
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing price csv: {self.csv_path}")
        df = pd.read_csv(self.csv_path, index_col=0, low_memory=False)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "date"
        df = df[~df.index.isna()].sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        if cols is not None:
            df = df.reindex(columns=list(cols))
        self.pq_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.pq_path)
        return self.pq_path

    def load(self) -> pd.DataFrame:
        self.ensure()
        return pd.read_parquet(self.pq_path)


@dataclass(frozen=True)
class WStore:
    base_dir: Path = FACTOR_DIR / "portfolio_weights"

    def load(self) -> pd.DataFrame:
        files = sorted(self.base_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No weights found in {self.base_dir}")
        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames).sort_index()
        return df

    def weight(self) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return pd.DataFrame()
        if "weight" not in df.columns:
            raise ValueError("Weights parquet must have 'weight' column.")
        if set(df.index.names) != {"date", "split", "asset"}:
            raise ValueError(f"Unexpected index levels: {df.index.names}")
        s = df["weight"].reset_index("split", drop=True)
        if s.index.duplicated().any():
            raise ValueError("Duplicate (date, asset) rows found after dropping split.")
        w = s.unstack("asset").sort_index()
        w.index.name = "date"
        return w
