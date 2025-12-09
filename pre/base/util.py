from __future__ import annotations

import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    return df


def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return safe_div(numer, denom)


def safe_div(num: pd.Series, denom: pd.Series, *, allow_negative_denom: bool = False) -> pd.Series:
    valid = denom.notna()
    if allow_negative_denom:
        valid &= denom != 0
    else:
        valid &= denom > 0
    return (num / denom).where(valid)
