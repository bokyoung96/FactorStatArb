from __future__ import annotations

import pandas as pd


class F:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def add_feature(df: pd.DataFrame, name: str, values: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    values.columns = pd.MultiIndex.from_product([[name], values.columns])
    return pd.concat([df, values], axis=1)
