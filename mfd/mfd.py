from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

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


def _to_bool_mask(df: pd.DataFrame) -> pd.DataFrame:
    if df.dtypes.nunique() == 1 and df.dtypes.iloc[0] == bool:
        return df
    if df.dtypes.nunique() == 1 and str(df.dtypes.iloc[0]).startswith("int"):
        return df != 0
    if df.dtypes.nunique() == 1 and str(df.dtypes.iloc[0]).startswith("float"):
        return df.fillna(0.0) != 0.0
    return df.astype(bool)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=True, na_option="keep")


@dataclass(frozen=True)
class MFDResult:
    ranks: Dict[str, pd.DataFrame]
    mean_rank: pd.DataFrame
    dispersion: pd.DataFrame
    confidence: pd.DataFrame
    score: pd.DataFrame

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, df in self.ranks.items():
            df.to_parquet(out_dir / f"rank_{k}.parquet")
        self.mean_rank.to_parquet(out_dir / "mfd_mean_rank.parquet")
        self.dispersion.to_parquet(out_dir / "mfd_dispersion.parquet")
        self.confidence.to_parquet(out_dir / "mfd_confidence.parquet")
        self.score.to_parquet(out_dir / "mfd_score.parquet")
        return out_dir


class MFDProcessor:
    def __init__(self, *, scores: ScoreReader, universe_path: Path = UNIVERSE_PARQUET):
        self.scores = scores
        self.universe_path = Path(universe_path)

    @staticmethod
    def from_config(
        *,
        mode: str = "TEST",
        timeframe: str = "MEDIUM",
        config_path: Optional[Path] = None,
        universe_path: Path = UNIVERSE_PARQUET,
    ) -> "MFDProcessor":
        sr = ScoreReader.from_config(mode=mode, timeframe=timeframe, config_path=config_path)
        return MFDProcessor(scores=sr, universe_path=universe_path)

    def load_universe(self) -> pd.DataFrame:
        uni = pd.read_parquet(self.universe_path)
        uni = _ensure_datetime_index(uni)
        return _to_bool_mask(uni)

    def universe_mask_like(self, like: pd.DataFrame) -> pd.DataFrame:
        uni = self.load_universe()
        mask = uni.reindex(index=like.index, columns=like.columns)
        mask = mask.fillna(False)
        return mask.astype(bool)

    def _filtered_scores(self) -> Dict[str, pd.DataFrame]:
        full = _ensure_datetime_index(self.scores.full)
        pm = _ensure_datetime_index(self.scores.pm)
        liq = _ensure_datetime_index(self.scores.liq)
        tech = _ensure_datetime_index(self.scores.tech)
        mask = self.universe_mask_like(full)
        return {
            "full": full.where(mask),
            "pm": pm.where(mask),
            "liq": liq.where(mask),
            "tech": tech.where(mask),
        }

    def compute(
        self,
        *,
        views: Iterable[str] = ("full", "pm", "liq", "tech"),
        max_dispersion: float = 0.5,
    ) -> MFDResult:
        scores = self._filtered_scores()
        use = [v for v in views]
        missing = [v for v in use if v not in scores]
        if missing:
            raise KeyError(f"Unknown views: {missing}. Available: {sorted(scores)}")

        ranks = {v: _cs_rank(scores[v]) for v in use}

        stack = np.stack([ranks[v].to_numpy(dtype=np.float32, copy=False) for v in use], axis=2)
        valid = np.isfinite(stack)
        count = valid.sum(axis=2).astype(np.float32, copy=False)
        s = np.nansum(stack, axis=2)
        mean_rank = np.divide(s, count, where=count > 0)
        mean_rank[count == 0] = np.nan

        demeaned = np.where(valid, stack - mean_rank[..., None], 0.0)
        var = np.divide((demeaned**2).sum(axis=2), count, where=count > 0)
        dispersion = np.sqrt(var, dtype=np.float32)
        dispersion[count == 0] = np.nan

        conf = 1.0 - (dispersion / float(max_dispersion))
        conf = np.clip(conf, 0.0, 1.0)
        score = (mean_rank - 0.5) * conf + 0.5

        idx = next(iter(ranks.values())).index
        cols = next(iter(ranks.values())).columns
        to_df = lambda x: pd.DataFrame(x, index=idx, columns=cols)

        return MFDResult(
            ranks=ranks,
            mean_rank=to_df(mean_rank),
            dispersion=to_df(dispersion),
            confidence=to_df(conf),
            score=to_df(score),
        )

    def default_out_dir(self) -> Path:
        return Path(self.scores.paths.out_root) / "mfd"

    def compute_and_save(
        self,
        *,
        out_dir: Optional[Path] = None,
        views: Iterable[str] = ("full", "pm", "liq", "tech"),
        max_dispersion: float = 0.5,
    ) -> MFDResult:
        res = self.compute(views=views, max_dispersion=max_dispersion)
        res.save(out_dir or self.default_out_dir())
        return res


if __name__ == "__main__":
    m = MFDProcessor.from_config(mode="TEST", timeframe="MEDIUM")
    res = m.compute_and_save()
    print("saved", m.default_out_dir(), "shape", res.score.shape)
