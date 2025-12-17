from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .cost import Cost


@dataclass(frozen=True)
class Leg:
    t: str
    q: float
    p0: float
    p1: float
    v0: float
    v1: float


@dataclass(frozen=True)
class Trade:
    gid: str
    d0: pd.Timestamp
    d1: pd.Timestamp
    legs: Tuple[Leg, ...]
    cin: float
    cout: float
    r: float
    cost: float
    note: str | None = None


class Port:
    def __init__(
        self,
        gid: str,
        cap0: float,
        cost: Cost | None = None,
    ) -> None:
        self.gid = str(gid)
        self.cap = float(cap0)
        self.cost = cost or Cost.off()
        self._eq: Dict[pd.Timestamp, float] = {}
        self._ret: Dict[pd.Timestamp, float] = {}
        self.trades: List[Trade] = []
        self._init = False

    def mark0(self, d: pd.Timestamp) -> None:
        if not self._init:
            self._eq[pd.Timestamp(d)] = float(self.cap)
            self._init = True

    def eq(self) -> pd.Series:
        return pd.Series(self._eq).sort_index()

    def rets(self) -> pd.Series:
        return pd.Series(self._ret).sort_index()

    def rebalance(
        self,
        *,
        d0: pd.Timestamp,
        d1: pd.Timestamp,
        tickers: Sequence[str],
        px: pd.DataFrame,
        w: pd.Series | None = None,
        note: str | None = None,
    ) -> None:
        cin = float(self.cap)
        tickers = tuple(tickers)
        if px.empty:
            raise ValueError("Empty px slice.")
        if cin <= 0 or not tickers:
            self._carry(d0=d0, d1=d1, cin=cin, note=note or "no tickers", px=px)
            return

        p0 = px.iloc[0]
        p1 = px.iloc[-1]
        ent, ex, halt = self._px_ok(p0, p1, tickers)
        if ent.empty:
            self._carry(d0=d0, d1=d1, cin=cin, note=note or "no px", px=px)
            return

        invest = self.cost.in_cap(cin)
        if invest <= 0:
            self._carry(d0=d0, d1=d1, cin=cin, note=note or "cap <= 0 after cost", px=px)
            return

        w = self._w_ok(ent.index, w)
        q, legs = self._legs(ent, ex, invest, w)
        if q.empty:
            self._carry(d0=d0, d1=d1, cin=cin, note=note or "no size", px=px)
            return

        pnl = float(((ex.reindex(q.index) - ent.reindex(q.index)) * q).sum())
        gross = cin + pnl
        cout = self.cost.out_cap(gross, frac=1.0)
        cost = (cin - invest) + (gross - cout)

        px2 = px.reindex(columns=q.index)
        deq = self._deq(px2, q, invest, cout, ent)
        msg = self._note(note, self._halt_msg(halt))
        self._rec(d0=d0, d1=d1, legs=legs, cin=cin, cout=cout, cost=cost, note=msg, deq=deq)

    def _carry(self, *, d0: pd.Timestamp, d1: pd.Timestamp, cin: float, note: str | None, px: pd.DataFrame) -> None:
        deq = pd.Series(cin, index=pd.Index(px.index)) if len(px.index) else None
        self._rec(d0=d0, d1=d1, legs=tuple(), cin=cin, cout=cin, cost=0.0, note=note, deq=deq)

    def _px_ok(
        self,
        p0: pd.Series,
        p1: pd.Series,
        tickers: Sequence[str],
    ) -> tuple[pd.Series, pd.Series, tuple[str, ...]]:
        ent = p0.reindex(tickers)
        ok = ent.notna() & (ent > 0)
        ent = ent[ok]
        ex = p1.reindex(ent.index)
        bad = ex.isna() | (ex <= 0)
        halt = list(ex.index[bad])
        if bad.any():
            ex = ex.copy()
            ex[bad] = ent[bad]
        return ent, ex, tuple(halt)

    def _halt_msg(self, halt: tuple[str, ...]) -> str | None:
        if not halt:
            return None
        return "halt: " + ", ".join(halt)

    def _note(self, *notes: str | None) -> str | None:
        xs = [n for n in notes if n]
        return None if not xs else " | ".join(xs)

    def _w_ok(self, tickers: Sequence[str], w: pd.Series | None) -> pd.Series | None:
        if w is None:
            return None
        w = pd.Series(w, dtype=float).reindex(tickers).fillna(0.0)
        if w.empty:
            return w

        # Enforce dollar-neutral + gross normalization on the tradable subset.
        w = w - float(w.mean())
        gross = float(w.abs().sum())
        if gross <= 0:
            return pd.Series(0.0, index=pd.Index(tickers), dtype=float)
        w = w / gross

        pos = w.clip(lower=0.0)
        neg = w.clip(upper=0.0)
        pos_sum = float(pos.sum())
        neg_sum = float((-neg).sum())
        if pos_sum > 0 and neg_sum > 0:
            w = pos * (0.5 / pos_sum) + neg * (0.5 / neg_sum)

        return w

    def _legs(
        self,
        p0: pd.Series,
        p1: pd.Series,
        invest: float,
        w: pd.Series | None = None,
    ) -> tuple[pd.Series, Tuple[Leg, ...]]:
        if p0.empty or invest <= 0:
            return pd.Series(dtype=float), tuple()

        qs: Dict[str, float] = {}
        legs: list[Leg] = []
        for t, px0 in p0.items():
            if px0 <= 0:
                continue
            wt = 0.0 if w is None else float(w.get(t, 0.0))
            if wt == 0.0:
                continue
            px1 = float(p1.get(t, px0))
            q = invest * wt / float(px0)
            tid = str(t)
            qs[tid] = float(q)
            v0 = float(px0 * q)
            v1 = float(px1 * q)
            legs.append(Leg(t=tid, q=float(q), p0=float(px0), p1=px1, v0=v0, v1=v1))
        return pd.Series(qs, dtype=float), tuple(legs)

    def _deq(
        self,
        px: pd.DataFrame,
        q: pd.Series,
        invest: float,
        cap1: float,
        ent: pd.Series | None = None,
    ) -> pd.Series | None:
        if invest <= 0 or q.empty or px.empty:
            return None
        px = px.ffill().dropna(how="all")
        if px.empty:
            return None
        if ent is None:
            return None
        ent = pd.Series(ent).reindex(q.index).ffill().bfill()
        pnl = (px - ent).mul(q, axis=1).sum(axis=1)
        eq = invest + pnl
        if eq.empty:
            return None
        eq.iloc[0] = invest
        eq.iloc[-1] = cap1
        return eq

    def _rec(
        self,
        *,
        d0: pd.Timestamp,
        d1: pd.Timestamp,
        legs: Tuple[Leg, ...],
        cin: float,
        cout: float,
        cost: float,
        note: str | None,
        deq: pd.Series | None = None,
    ) -> None:
        self.cap = float(cout)
        if deq is not None and not deq.empty:
            for ts, v in deq.sort_index().items():
                ts = pd.Timestamp(ts)
                if ts not in self._eq:
                    self._eq[ts] = float(v)
        self._eq[pd.Timestamp(d1)] = float(cout)
        r = 0.0 if cin == 0 else (cout / cin) - 1.0
        self._ret[pd.Timestamp(d1)] = float(r)
        self.trades.append(Trade(gid=self.gid, d0=d0, d1=d1, legs=legs, cin=cin, cout=cout, r=float(r), cost=float(cost), note=note))
