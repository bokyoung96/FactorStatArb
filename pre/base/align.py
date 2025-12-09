from __future__ import annotations

import pandas as pd


class Align:
    def daily(self, df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        return df.reindex(idx).ffill()

    def merge(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        out = dfs[0]
        for d in dfs[1:]:
            out = out.join(d, how="outer")
        return out.ffill()
