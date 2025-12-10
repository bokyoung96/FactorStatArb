from __future__ import annotations

import pandas as pd


def master(df: pd.DataFrame) -> list[str]:
    mask = df.fillna(0) == 1
    return mask.any(axis=0)[lambda s: s].index.tolist()


def active(df: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    sub = df.loc[:, [c for c in df.columns if c in universe]]
    return (sub == 1).astype(bool)
