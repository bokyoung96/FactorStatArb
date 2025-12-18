from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from root import UNIVERSE_PARQUET

from mfd.read import ScoreReader


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


@dataclass(frozen=True)
class ScorePaths:
    root_dir: Path
    run_name: str
    out_root: Path
    mfd_dir: Path

    @staticmethod
    def from_reader(reader: ScoreReader) -> "ScorePaths":
        out_root = Path(reader.paths.out_root).resolve()
        return ScorePaths(
            root_dir=Path(reader.paths.root_dir).resolve(),
            run_name=str(reader.paths.run_name),
            out_root=out_root,
            mfd_dir=out_root / "mfd",
        )


class TradeScore:
    def __init__(
        self,
        *,
        scores: ScoreReader,
        bottom_pct: float = 10.0,
        mfd_metric: str = "dispersion",
        universe_path: Path = UNIVERSE_PARQUET,
    ):
        self.scores = scores
        self.paths = ScorePaths.from_reader(scores)
        self.bottom_pct = float(bottom_pct)
        self.mfd_metric = str(mfd_metric).strip().lower()
        self.universe_path = Path(universe_path)

        if not (0.0 < self.bottom_pct <= 100.0):
            raise ValueError("bottom_pct must be in (0, 100].")
        if self.mfd_metric not in {"dispersion", "confidence", "score"}:
            raise ValueError("mfd_metric must be one of: dispersion, confidence, score.")

    @staticmethod
    def from_config(
        *,
        mode: str = "TEST",
        timeframe: str = "MEDIUM",
        config_path: Optional[Path] = None,
        bottom_pct: float = 10.0,
        mfd_metric: str = "dispersion",
        universe_path: Path = UNIVERSE_PARQUET,
    ) -> "TradeScore":
        sr = ScoreReader.from_config(mode=mode, timeframe=timeframe, config_path=config_path)
        return TradeScore(scores=sr, bottom_pct=bottom_pct, mfd_metric=mfd_metric, universe_path=universe_path)

    def _load_universe(self) -> pd.DataFrame:
        uni = pd.read_parquet(self.universe_path)
        uni = _ensure_datetime_index(uni)
        if uni.dtypes.nunique() == 1 and uni.dtypes.iloc[0] == bool:
            pass
        else:
            uni = uni.astype(bool)
        return uni

    @cached_property
    def universe_mask(self) -> pd.DataFrame:
        base = _ensure_datetime_index(self.scores.full)
        uni = self._load_universe()
        mask = uni.reindex(index=base.index, columns=base.columns).fillna(False)
        return mask.astype(bool)

    @cached_property
    def score_full(self) -> pd.DataFrame:
        df = _ensure_datetime_index(self.scores.full)
        return df.where(self.universe_mask)

    @cached_property
    def mfd(self) -> pd.DataFrame:
        if self.mfd_metric == "dispersion":
            path = self.paths.mfd_dir / "mfd_dispersion.parquet"
        elif self.mfd_metric == "confidence":
            path = self.paths.mfd_dir / "mfd_confidence.parquet"
        else:
            path = self.paths.mfd_dir / "mfd_score.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing MFD artifact: {path}")
        df = pd.read_parquet(path)
        df = _ensure_datetime_index(df)
        df = df.reindex(index=self.score_full.index, columns=self.score_full.columns)
        return df.where(self.universe_mask)

    @cached_property
    def trade_mask(self) -> pd.DataFrame:
        m = self.mfd
        p = self.bottom_pct / 100.0
        ascending = True if self.mfd_metric in {"dispersion"} else False
        pct_rank = m.rank(axis=1, pct=True, ascending=ascending, na_option="keep")
        sel = pct_rank <= p
        sel = sel & self.universe_mask
        return sel.astype(bool)

    @cached_property
    def final_score(self) -> pd.DataFrame:
        return self.score_full.where(self.trade_mask)

    @cached_property
    def counts(self) -> pd.DataFrame:
        n_univ = self.universe_mask.sum(axis=1).astype(int)
        n_sel = self.trade_mask.sum(axis=1).astype(int)
        df = pd.DataFrame({"n_universe": n_univ, "n_selected": n_sel})
        df.index.name = "date"
        return df

    def plot_counts(self, *, figsize: tuple[int, int] = (12, 4)):
        c = self.counts
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        ax.plot(c.index, c["n_selected"], label="n_selected")
        ax.plot(c.index, c["n_universe"], label="n_universe", alpha=0.5)
        ax.set_title(f"Selected tickers per date (bottom_pct={self.bottom_pct:g}%, metric={self.mfd_metric})")
        ax.legend()
        return fig, ax

    def save(self, *, out_path: Optional[Path] = None) -> Path:
        out = Path(out_path) if out_path is not None else (self.paths.out_root / "final_score.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.final_score.to_parquet(out)
        return out


if __name__ == "__main__":
    ts = TradeScore.from_config(mode="TEST", timeframe="MEDIUM", bottom_pct=50.0, mfd_metric="dispersion")
    path = ts.save()
    ts.plot_counts()
    print("saved", path)
