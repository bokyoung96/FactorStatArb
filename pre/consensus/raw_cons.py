from __future__ import annotations

import pandas as pd


class ConsPrep:
    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index().ffill()
