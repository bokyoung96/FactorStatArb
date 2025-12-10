from __future__ import annotations

import pandas as pd


def yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(252)


def rev_ffill_pct(series: pd.Series) -> pd.Series:
    s = series.ffill()
    return s.pct_change(fill_method=None)


def turn_flag(series: pd.Series) -> pd.Series:
    s = series.ffill()
    return (s * s.shift(1) < 0).astype(float)
