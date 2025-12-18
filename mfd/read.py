from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from mfd.params import TransformerParams, build_name


@dataclass(frozen=True)
class ScorePaths:
    run_name: str
    root_dir: Path
    out_root: Path

    def path(self, view: str) -> Path:
        return self.out_root / f"{view}.parquet"


class ScoreReader:
    def __init__(self, *, run_name: str, out_root: Path):
        self.paths = ScorePaths(run_name=run_name, root_dir=Path(ROOT_DIR).resolve(), out_root=Path(out_root).resolve())
        self._scores: Dict[str, pd.DataFrame] = {}
        self._scores["full"] = self._read("score_full")
        self._scores["pm"] = self._read("score_price_mom")
        self._scores["liq"] = self._read("score_vol_liq")
        self._scores["tech"] = self._read("score_tech")

    @staticmethod
    def from_config(*, mode: str = "TEST", timeframe: str = "MEDIUM", config_path: Optional[Path] = None) -> "ScoreReader":
        params = TransformerParams(config_path=config_path)
        cfg = params.get_config(mode=mode, timeframe=timeframe)
        run_name = build_name(cfg.mode, cfg.rolling_train_years, cfg.rolling_test_years)
        return ScoreReader(run_name=run_name, out_root=Path(cfg.output_dir) / run_name)

    def _read(self, view: str) -> pd.DataFrame:
        path = self.paths.path(view)
        if not path.exists():
            raise FileNotFoundError(f"Missing score file: {path}")
        return pd.read_parquet(path)

    @property
    def full(self) -> pd.DataFrame:
        return self._scores["full"]

    @property
    def pm(self) -> pd.DataFrame:
        return self._scores["pm"]

    @property
    def liq(self) -> pd.DataFrame:
        return self._scores["liq"]

    @property
    def tech(self) -> pd.DataFrame:
        return self._scores["tech"]

    @property
    def scores(self) -> Dict[str, pd.DataFrame]:
        return dict(self._scores)


if __name__ == "__main__":
    r = ScoreReader.from_config(mode="TEST", timeframe="MEDIUM")
    print("run", r.paths.run_name)
    print("full", r.full.shape, "pm", r.pm.shape, "liq", r.liq.shape, "tech", r.tech.shape)
