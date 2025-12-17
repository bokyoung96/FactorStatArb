from __future__ import annotations

import math

import pandas as pd


def _dd(eq: pd.Series) -> pd.Series:
    eq = eq.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    peak = eq.cummax()
    return (eq / peak) - 1.0


def stats(eq: pd.Series) -> dict:
    eq = pd.Series(eq).dropna().sort_index()
    if eq.empty:
        return {"tot": 0.0, "cagr": 0.0, "vol": 0.0, "sh": 0.0, "mdd": 0.0, "n": 0}

    r = eq.pct_change().dropna()
    s0 = float(eq.iloc[0])
    s1 = float(eq.iloc[-1])
    tot = 0.0 if s0 == 0 else (s1 / s0) - 1.0

    n = int(r.shape[0])
    if n <= 1:
        return {"tot": tot, "cagr": 0.0, "vol": 0.0, "sh": 0.0, "mdd": float(_dd(eq).min()), "n": n}

    days = (eq.index[-1] - eq.index[0]).days
    yrs = days / 365.25 if days > 0 else 0.0
    cagr = 0.0 if yrs <= 0 or s0 <= 0 else (s1 / s0) ** (1.0 / yrs) - 1.0

    vol = float(r.std(ddof=0) * math.sqrt(252))
    sh = 0.0 if vol == 0 else float(r.mean() / r.std(ddof=0) * math.sqrt(252))
    mdd = float(_dd(eq).min())
    return {"tot": float(tot), "cagr": float(cagr), "vol": vol, "sh": sh, "mdd": mdd, "n": n}

