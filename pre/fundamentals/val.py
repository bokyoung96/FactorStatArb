from __future__ import annotations

import pandas as pd

from pre.base.util import safe_divide


def price_to_book(price: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    return safe_divide(price, equity)


def enterprise_value(market_cap: pd.DataFrame, debt: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    return market_cap.add(debt, fill_value=0).sub(cash, fill_value=0)


def ev_to_sales(ev: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    return safe_divide(ev, revenue)
