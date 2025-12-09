## Features Overview

This project produces daily-aligned factor outputs saved under `DATA/processed/features/`. Source data are loaded at lower frequency (monthly/quarterly), transformed (TTM, averages, growth), forward-filled to the price index, and then factorized. Below is a concise catalog of the current features by group.

### Price Factors
- `ret1`, `ret5`, `ret10`, `m1`, `m3`, `m6`, `m12` — Horizon returns.
- `m1_va`, `m12_va` — Return over volatility.
- `rev5` — Short-term reversal.
- `vol5`, `vol10`, `vol20`, `vol60`, `vol120` — Rolling return volatility.
- `hlr`, `idv`, `trange` — High-low range based intraday ranges.
- `volz` — Volume z-score (20-day).
- `volma20`, `volma_r` — Volume moving average and ratio.
- `turnover`, `amihud`, `sprd`, `pimpact`, `volshock` — Liquidity/impact measures.
- `ma5`, `ma20`, `ma60`, `ma120`, `macd`, `macds` — Moving averages and MACD signals.
- `rsi14`, `sto_k`, `sto_d` — Momentum oscillators.
- `boll_up`, `boll_low`, `boll_w` — Bollinger band metrics.
- `high52`, `low52`, `price_z` — Relative price level vs. 52-week and z-score.
- `dist_ma20` — Distance to 20-day MA.
- `breakout` — 20-day breakout indicator.

### Fundamentals Factors (TTM / averages precomputed on quarterly snapshots, then ffilled to daily)
- `bm` — Book-to-market: equity / mcap (denom > 0).
- `ep` — Earnings-to-price: `ni_ttm` / mcap (denom > 0, earnings can be negative).
- `roe` — Return on equity: `ni_ttm` / `equity_avg` (denom > 0).
- `gp_a` — Gross profit to assets: `gp_ttm` / `assets_avg` (denom > 0).
- `acc` — Accruals: `(ni_ttm - ocf_ttm)` / `assets_avg` (denom > 0).
- `opm` — Operating margin: `op_ttm` / `rev_ttm` when revenue exists (denom > 0).
- `sg` — Sales growth: `rev_g` when revenue growth exists.
- `ag` — Asset growth: `assets_g`.
- `lev` — Leverage: liab / assets (denom > 0).
- `turn` — Asset turnover (revenue unavailable): `op_ttm` / `assets_avg` (denom > 0).

### Consensus Factors
- `op_fq1`, `op_fq2`, `op_fy1` — Operating profit forecasts (next quarters/year).
- `eps_fq1`, `eps_fq2`, `eps_fy1` — EPS forecasts (next quarters/year).
- `rev_op_fq1`, `rev_op_fq2`, `rev_op_fy1` — Revisions of operating profit forecasts.
- `rev_eps_fq1`, `rev_eps_fq2` — Revisions of EPS forecasts.

### Sector Factors
- `sector_oh`, `sector_id` — One-hot and numeric sector identifiers.

### Notes
- All denominators are guarded via `safe_div` to avoid inf/-inf when denom ≤ 0 or NaN (unless explicitly allowed).
- TTM sums and growth rates are computed on quarterly snapshots (`resample("QE")`), then forward-filled to trading days; no rolling/pct_change is done on daily data.
