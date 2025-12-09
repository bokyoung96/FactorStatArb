from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    shifted = series.shift(periods)
    return np.log(series / shifted)


def forward_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(-periods).div(series).sub(1.0)
