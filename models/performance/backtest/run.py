from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.performance.backtest.bt import Bt, Cfg, Rep
from models.performance.backtest.cost import Cost
from models.performance.backtest.io import BT_DIR, PxStore, WStore
from root import factor_dir_freq


def prep_px(*, force: bool = False) -> None:
    PxStore().build(force=force)


def load_w():
    return WStore().weight()


def run(
    *,
    freq: str = "D",
    lag: int = 0,
    aum: float = 100_000_000.0,
    cost: Optional[Cost] = None,
    beta_neutral: bool = False,
    beta_lookback: int = 252,
    force_px: bool = False,
    out: Optional[Path] = BT_DIR / "out",
) -> Rep:
    freq_map = {"d": "D", "daily": "D", "w": "W-FRI", "weekly": "W-FRI", "m": "ME", "monthly": "ME"}
    freq = freq_map.get(str(freq).lower(), str(freq))
    store = PxStore()
    if force_px:
        store.build(force=True)
    else:
        store.ensure()
    px = store.load()
    factor_dir = factor_dir_freq(freq)
    w = WStore(base_dir=factor_dir / "portfolio_weights").weight()
    cfg = Cfg(
        aum=float(aum),
        freq=str(freq),
        lag=int(lag),
        cost=cost or Cost.off(),
        beta_neutral=bool(beta_neutral),
        beta_lookback=int(beta_lookback),
    )
    rep = Bt(cfg, px, w).run()
    if out is not None:
        rep.save(Path(out))
    return rep


if __name__ == "__main__":
    rep = run(freq="weekly", lag=0)
    print(rep.stat)
