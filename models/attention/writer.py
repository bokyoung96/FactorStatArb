from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from models.attention.data import FactorResult
from root import FACTOR_DIR, FACTOR_PRED_PARQUET, FACTOR_RES_PARQUET, FACTOR_RETURNS_PARQUET, FACTOR_WEIGHTS_PARQUET


class FactorResultWriter:
    def __init__(self, output_dir: Path = FACTOR_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _append(self, df: pd.DataFrame, path: Path) -> None:
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(path)

    def save(self, date, assets: Sequence[str], result: FactorResult, split: str) -> None:
        factor_cols = [f"f{i}" for i in range(result.weights.size(0))]
        weights_df = pd.DataFrame(result.weights.detach().cpu().numpy().T, index=pd.Index(assets, name="asset"), columns=factor_cols)
        weights_df["date"] = date
        weights_df["split"] = split
        weights_df = weights_df.reset_index().set_index(["date", "split", "asset"])
        self._append(weights_df, FACTOR_WEIGHTS_PARQUET)

        fr_df = pd.DataFrame([result.factor_returns.detach().cpu().numpy()], index=pd.Index([date], name="date"), columns=factor_cols)
        fr_df["split"] = split
        fr_df = fr_df.set_index("split", append=True)
        self._append(fr_df, FACTOR_RETURNS_PARQUET)

        pred_df = pd.DataFrame(
            {"predicted": result.predicted.detach().cpu().numpy(), "residual": result.residuals.detach().cpu().numpy()},
            index=pd.Index(assets, name="asset"),
        )
        pred_df["date"] = date
        pred_df["split"] = split
        pred_df = pred_df.reset_index().set_index(["date", "split", "asset"])
        self._append(pred_df[["predicted"]], FACTOR_PRED_PARQUET)
        self._append(pred_df[["residual"]], FACTOR_RES_PARQUET)
