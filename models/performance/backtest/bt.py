from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .cost import Cost
from .perf import stats as _stats
from .port import Port
from .plot import save as _plot
from tqdm import tqdm


@dataclass(frozen=True)
class Cfg:
    aum: float = 100_000_000.0
    freq: str = "D"
    lag: int = 0
    cost: Cost = Cost.off()
    lw: float = 0.5
    sw: float = 0.5
    beta_neutral: bool = False
    beta_lookback: int = 252


@dataclass(frozen=True)
class Win:
    ent: pd.Timestamp
    ex: pd.Timestamp
    w: pd.Timestamp


@dataclass(frozen=True)
class Rep:
    eq: pd.Series
    eq_l: pd.Series
    eq_s: pd.Series
    ret: pd.Series
    ret_l: pd.Series
    ret_s: pd.Series
    trades: list
    trades_l: list
    trades_s: list

    @property
    def stat(self) -> dict:
        return _stats(self.eq)

    def save(self, dir: Path) -> None:
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        self.eq.to_frame("eq").to_parquet(dir / "eq.parquet")
        self.eq_l.to_frame("eq_l").to_parquet(dir / "eq_l.parquet")
        self.eq_s.to_frame("eq_s").to_parquet(dir / "eq_s.parquet")
        self.ret.to_frame("ret").to_parquet(dir / "ret.parquet")
        pnl = pd.DataFrame({"gross": self.ret, "net": self.ret, "cost": 0.0}).dropna()
        pnl.to_parquet(dir / "pnl.parquet")
        pd.DataFrame({"gross_l": self.ret_l, "gross_s": self.ret_s}).dropna().to_parquet(dir / "pnl_books.parquet")
        pd.DataFrame([self.stat]).to_parquet(dir / "stat.parquet")
        _trades(self.trades).to_parquet(dir / "trades.parquet")
        _trades(self.trades_l).to_parquet(dir / "trades_l.parquet")
        _trades(self.trades_s).to_parquet(dir / "trades_s.parquet")
        _plot(eq=self.eq, eq_l=self.eq_l, eq_s=self.eq_s, out=dir, title="bt")


def _trades(xs: list) -> pd.DataFrame:
    if not xs:
        return pd.DataFrame()
    rows = []
    for t in xs:
        rows.append(
            {
                "gid": t.gid,
                "d0": t.d0,
                "d1": t.d1,
                "cin": t.cin,
                "cout": t.cout,
                "r": t.r,
                "cost": t.cost,
                "n": len(t.legs),
                "note": t.note,
            }
        )
    return pd.DataFrame(rows)


class Sched:
    def __init__(self, px_dates: pd.DatetimeIndex, w_dates: pd.DatetimeIndex, freq: str, lag: int = 0) -> None:
        self.px_dates = pd.DatetimeIndex(px_dates).sort_values().unique()
        self.w_dates = pd.DatetimeIndex(w_dates).sort_values().unique()
        self.freq = str(freq)
        self.lag = int(lag)
        if self.px_dates.empty:
            raise ValueError("empty px dates")
        if self.w_dates.empty:
            raise ValueError("empty weight dates")
        if self.lag < 0:
            raise ValueError("lag must be >= 0")
        self._sig = self._sig_dates()
        self._wins = self._wins_build()

    @property
    def sig(self) -> list[pd.Timestamp]:
        return list(self._sig)

    def wins(self) -> list[Win]:
        return list(self._wins)

    def _sig_dates(self) -> list[pd.Timestamp]:
        end = min(self.w_dates.max(), self.px_dates.max())
        w = self.w_dates[self.w_dates <= end]
        if w.empty:
            raise ValueError("no overlap in dates")
        if self.freq.upper() in {"D", "B"}:
            sig = list(pd.DatetimeIndex(w).unique())
        else:
            s = pd.Series(1, index=pd.DatetimeIndex(w))
            sig = [pd.Timestamp(x) for x in s.resample(self.freq).last().dropna().index]
        if pd.Timestamp(end) not in sig:
            sig.append(pd.Timestamp(end))
        sig = sorted(set(sig))
        if len(sig) < 2:
            raise ValueError("need >=2 signal dates")
        return sig

    def _px_at_or_before(self, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        d = pd.Timestamp(d)
        loc = int(self.px_dates.searchsorted(d, side="right")) - 1
        if loc < 0:
            return None
        return pd.Timestamp(self.px_dates[loc])

    def _px_at_lag(self, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        base = self._px_at_or_before(d)
        if base is None:
            return None
        loc = int(self.px_dates.get_indexer([base])[0])
        idx = loc + self.lag
        if idx >= len(self.px_dates):
            return None
        return pd.Timestamp(self.px_dates[idx])

    def _wins_build(self) -> list[Win]:
        out: list[Win] = []
        if self.freq.upper() in {"D", "B"}:
            for cur in self._sig:
                ex = self._px_at_or_before(cur)
                if ex is None:
                    continue
                pos = int(self.px_dates.searchsorted(ex, side="left"))
                if pos <= 0:
                    continue
                ent = pd.Timestamp(self.px_dates[pos - 1])
                if ent >= ex:
                    continue
                out.append(Win(ent=ent, ex=ex, w=pd.Timestamp(cur)))
        else:
            for i in range(1, len(self._sig)):
                prev = pd.Timestamp(self._sig[i - 1])
                cur = pd.Timestamp(self._sig[i])
                ent = self._px_at_or_before(prev)
                ex = self._px_at_or_before(cur)
                if ent is None or ex is None or ent >= ex:
                    continue
                out.append(Win(ent=ent, ex=ex, w=prev))
        if not out:
            raise ValueError("no windows built")
        return out


class Bt:
    def __init__(self, cfg: Cfg, px: pd.DataFrame, w: pd.DataFrame) -> None:
        self.cfg = cfg
        px = px.sort_index()
        w = w.sort_index()
        cols = px.columns.intersection(w.columns)
        self.px = px.reindex(columns=cols)
        self.w = w.reindex(columns=cols)
        if self.px.empty or self.w.empty:
            raise ValueError("px/w empty")
        self._w_idx = pd.DatetimeIndex(self.w.index)
        self._ret = self.px.pct_change(fill_method=None)

    def _beta_neutralize(self, w: pd.Series, asof: pd.Timestamp) -> pd.Series:
        w = pd.Series(w, dtype=float)
        if w.empty:
            return w
        lookback = int(self.cfg.beta_lookback)
        if lookback <= 1:
            return w
        hist = self._ret.loc[:pd.Timestamp(asof), w.index].tail(lookback)
        if hist.empty:
            return w
        mkt = hist.mean(axis=1, skipna=True)
        mv = float(mkt.var())
        if not np.isfinite(mv) or mv <= 0:
            return w

        mkt_c = mkt - float(mkt.mean())
        betas = {}
        for col in hist.columns:
            ri = hist[col].dropna()
            if ri.size < max(20, lookback // 4):
                continue
            aligned = ri.align(mkt_c, join="inner")[0]
            m_al = mkt_c.reindex(aligned.index)
            rc = aligned - float(aligned.mean())
            cov = float((rc * m_al).mean())
            if np.isfinite(cov):
                betas[col] = cov / mv
        if not betas:
            return w
        beta = pd.Series(betas, dtype=float).reindex(w.index).fillna(0.0)
        denom = float((beta * beta).sum())
        if denom <= 1e-12:
            return w
        port_beta = float((w * beta).sum())
        return w - (port_beta / denom) * beta

    def run(self) -> Rep:
        sched = Sched(pd.DatetimeIndex(self.px.index), self._w_idx, self.cfg.freq, lag=self.cfg.lag)
        wins = sched.wins()
        p = Port("LS", self.cfg.aum, self.cfg.cost)
        p.mark0(wins[0].ent)
        eq_l: dict[pd.Timestamp, float] = {pd.Timestamp(wins[0].ent): float(self.cfg.aum * self.cfg.lw)}
        eq_s: dict[pd.Timestamp, float] = {pd.Timestamp(wins[0].ent): float(self.cfg.aum * self.cfg.sw)}
        ret_l: dict[pd.Timestamp, float] = {}
        ret_s: dict[pd.Timestamp, float] = {}

        for win in tqdm(wins, desc=f"bt[{self.cfg.freq}]", leave=False):
            w0 = self._w_asof(win.w)
            if w0 is None or w0.empty:
                pxs = self.px.loc[win.ent : win.ex]
                p.rebalance(d0=win.ent, d1=win.ex, tickers=tuple(), px=pxs, note="no w")
                eq_l[pd.Timestamp(win.ex)] = float(p.cap * self.cfg.lw)
                eq_s[pd.Timestamp(win.ex)] = float(p.cap * self.cfg.sw)
                ret_l[pd.Timestamp(win.ex)] = 0.0
                ret_s[pd.Timestamp(win.ex)] = 0.0
                continue

            w0 = pd.Series(w0, dtype=float).dropna()
            w0 = w0[w0 != 0]

            pxs = self.px.loc[win.ent : win.ex]
            if self.cfg.beta_neutral:
                # Neutralize to market beta using info available at win.w (rebalance date).
                # Uses trailing daily returns up to win.w.
                w0 = self._beta_neutralize(w0, asof=win.w)
                w0 = w0 - float(w0.mean())
                gross = float(w0.abs().sum())
                if gross > 0:
                    w0 = w0 / gross
                w0 = w0[w0 != 0]
            cap0 = float(p.cap)
            p0 = pxs.iloc[0].reindex(w0.index)
            p1 = pxs.iloc[-1].reindex(w0.index)
            r = (p1 / p0 - 1.0).replace([pd.NA, float("inf"), float("-inf")], 0.0).fillna(0.0)
            gl = float((w0[w0 > 0] * r.reindex(w0[w0 > 0].index)).sum())
            gs = float((w0[w0 < 0] * r.reindex(w0[w0 < 0].index)).sum())

            p.rebalance(d0=win.ent, d1=win.ex, tickers=tuple(w0.index), px=pxs, w=w0, note=None)
            eq_l[pd.Timestamp(win.ex)] = float(eq_l[pd.Timestamp(win.ent)] + cap0 * gl) if pd.Timestamp(win.ent) in eq_l else float(self.cfg.aum * self.cfg.lw + cap0 * gl)
            eq_s[pd.Timestamp(win.ex)] = float(eq_s[pd.Timestamp(win.ent)] + cap0 * gs) if pd.Timestamp(win.ent) in eq_s else float(self.cfg.aum * self.cfg.sw + cap0 * gs)
            ret_l[pd.Timestamp(win.ex)] = float(gl)
            ret_s[pd.Timestamp(win.ex)] = float(gs)

        idx = pd.DatetimeIndex(self.px.index)
        eq = p.eq().reindex(idx).ffill().dropna()
        eq_l_s = pd.Series(eq_l).sort_index().reindex(idx).ffill().dropna()
        eq_s_s = pd.Series(eq_s).sort_index().reindex(idx).ffill().dropna()
        ret = eq.pct_change().dropna()
        ret_l_s = pd.Series(ret_l).sort_index()
        ret_s_s = pd.Series(ret_s).sort_index()
        return Rep(eq=eq, eq_l=eq_l_s, eq_s=eq_s_s, ret=ret, ret_l=ret_l_s, ret_s=ret_s_s, trades=p.trades, trades_l=[], trades_s=[])

    def _w_asof(self, d: pd.Timestamp) -> Optional[pd.Series]:
        d = pd.Timestamp(d)
        pos = int(self._w_idx.searchsorted(d, side="right")) - 1
        if pos < 0:
            return None
        return self.w.iloc[pos]
