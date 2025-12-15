# FactorStatArb — Attention + LongConv Statistical Arbitrage

## Overview
- Builds a feature tensor from the intersection of available features (price, sector by default), estimates residuals with an attention factor layer, and learns residual time-series patterns with LongConv to produce asset weights.
- Uses rolling windows: train for `window_years` (default 8) and test for `step_years` (default 1). For each window, the best-Sharpe checkpoint is saved and used to score the test period.
- Outputs are written under `DATA/factor` (parquet). Residuals are cached per date under `DATA/factor/cache/<year>/YYYYMMDD.pt` to avoid recomputation; deleting the cache is safe.

## How to Run
1) Activate env: `conda activate myenv` (CUDA GPU is used automatically if available).
2) Configure: edit `models/config.json` (`window_years`, `step_years`, `lookback`, `train_epochs`, cost/Sharpe/EV weights, etc.).
3) Execute from repo root: `python -m models.main`.

## Workflow
- Rolling schedule: iterate years in blocks of `window_years` (train) then `step_years` (test).
- Train (per day, per epoch): Attention → residuals → LongConv → portfolio weights → loss = -(λ_SR * Sharpe + λ_Var * explained_variance) with turnover/short costs; best-Sharpe checkpoint is saved.
- Infer: load the best checkpoint for that window, run through test dates, and write weights/PNL.
- Cache: residuals saved per date (overwritten if recalculated). Stored by year in `cache/<year>/`.

## Outputs
- `DATA/factor/portfolio_weights/*.parquet`: asset-level weights by date/split.
- `DATA/factor/pnl/*.parquet`: gross/net/cost by date/split.
- Residual cache (temporary): `DATA/factor/cache/<year>/YYYYMMDD.pt`.
- Factor-level parquet (weights/returns/predicted/residuals) is not written by default; add writer calls in the runner if needed.

## Notes
- Sharpe is computed on daily net returns (not annualized). Loss uses a negative sign to maximize Sharpe + explained variance.
- Split names follow `test_<train_end>_<test_end>`.
- GPU: `DeviceSelector` automatically prefers CUDA (then MPS, else CPU).

## Code Structure (key files)
- `models/main.py`: entry point, rolling schedule, checkpoint load/run.
- `models/attention/pipeline.py`: config, trainer, runner, end-to-end loop.
- `models/attention/model.py`: attention factor layer (weights, β, ω_ε, residuals).
- `models/attention/longconv.py`: LongConv sequence model on residual windows.
- `models/attention/losses.py`: turnover/short costs, rolling Sharpe, explained variance.
- `models/attention/cache.py`: residual cache (date → safe filename, year folders).
- `models/attention/writer.py`: parquet writers for weights/PNL (extensible).
