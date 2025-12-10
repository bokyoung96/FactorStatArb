import pandas as pd

from pre.base.feature import F, add_feature
from pre.base.reg import FE, register


def _sector_series(df: pd.DataFrame):
    if "sector" in df.columns:
        return df["sector"]
    if "sec" in df.columns:
        return df["sec"]
    return None


@register(FE.SECTOR_ID)
class SECTOR_ID(F):
    def run(self, df):
        sector = _sector_series(df)
        if sector is None:
            return df
        return add_feature(df, FE.SECTOR_ID.value, sector)


@register(FE.SECTOR_OH)
class SECTOR_OH(F):
    def run(self, df):
        sector = _sector_series(df)
        if sector is None:
            return df
        frames = []
        if isinstance(sector, pd.Series):
            frames.append(pd.get_dummies(sector, prefix="sector"))
        else:
            for col in sector.columns:
                frames.append(pd.get_dummies(sector[col], prefix=f"sector_{col}"))
        dummies = pd.concat(frames, axis=1).reindex(df.index)
        dummies.columns = pd.MultiIndex.from_product([[FE.SECTOR_OH.value], dummies.columns])
        return pd.concat([df, dummies], axis=1)
