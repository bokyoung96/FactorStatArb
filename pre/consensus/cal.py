from __future__ import annotations

import pandas as pd


def yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(252)
