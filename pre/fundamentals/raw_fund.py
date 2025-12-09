from __future__ import annotations

import pandas as pd


class FundPrep:
    @staticmethod
    def sort_and_fill(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index().ffill()

    @staticmethod
    def drop_sparse(df: pd.DataFrame, min_obs: int = 5) -> pd.DataFrame:
        """Drop columns with fewer than min_obs non-null observations."""
        return df.dropna(thresh=min_obs, axis=1)
