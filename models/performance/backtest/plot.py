from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _cum(eq: pd.Series) -> pd.Series:
    eq = pd.Series(eq).dropna().sort_index()
    if eq.empty:
        return eq
    s0 = float(eq.iloc[0])
    return pd.Series(0.0, index=eq.index) if s0 == 0 else (eq / s0) - 1.0


def _dd(eq: pd.Series) -> pd.Series:
    eq = pd.Series(eq).dropna().sort_index()
    if eq.empty:
        return eq
    peak = eq.cummax()
    return (eq / peak) - 1.0


def save(*, eq: pd.Series, out: Path, title: str = "bt", eq_l: pd.Series | None = None, eq_s: pd.Series | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    eq = pd.Series(eq).dropna().sort_index()
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(eq.index, eq, label="tot", lw=1.2)
    if eq_l is not None:
        eq_l = pd.Series(eq_l).reindex(eq.index).ffill()
        ax[0].plot(eq_l.index, eq_l, label="L", lw=0.9, alpha=0.9)
    if eq_s is not None:
        eq_s = pd.Series(eq_s).reindex(eq.index).ffill()
        ax[0].plot(eq_s.index, eq_s, label="S", lw=0.9, alpha=0.9)
    ax[0].set_title(title)
    ax[0].set_ylabel("eq")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)

    c = _cum(eq)
    ax[1].plot(c.index, c, label="tot", lw=1.2)
    if eq_l is not None:
        ax[1].plot(c.index, _cum(eq_l), label="L", lw=0.9, alpha=0.9)
    if eq_s is not None:
        ax[1].plot(c.index, _cum(eq_s), label="S", lw=0.9, alpha=0.9)
    ax[1].set_ylabel("cum")
    ax[1].legend(loc="best")
    ax[1].grid(True, alpha=0.3)

    dd = _dd(eq)
    ax[2].plot(dd.index, dd, label="dd", lw=1.0, color="tab:red")
    ax[2].set_ylabel("dd")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "bt.png", dpi=150)
    plt.close(fig)
