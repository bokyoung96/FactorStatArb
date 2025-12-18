from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from mfd.model import Transformer
from mfd.params import TransformerConfig, TransformerParams, build_name
from mfd.pipeline import load_bundle, StockDataset
from mfd.utils import DeviceSelector


logger = logging.getLogger(__name__)


def _rolling_splits(cfg: TransformerConfig, years: np.ndarray) -> list[tuple[list[int], list[int]]]:
    uniq = sorted({int(y) for y in years})
    tr = int(cfg.rolling_train_years)
    te = int(cfg.rolling_test_years)
    st = int(cfg.rolling_step_years)
    out = []
    for start in range(0, len(uniq) - tr - te + 1, st):
        train_years = uniq[start : start + tr]
        test_years = uniq[start + tr : start + tr + te]
        out.append((train_years, test_years))
    if not out:
        raise ValueError("Not enough years for rolling split.")
    return out


def evaluate(*, mode: str = "TEST", timeframe: str = "MEDIUM", out: Optional[Path] = None) -> Path:
    params = TransformerParams()
    cfg = params.get_config(mode=mode, timeframe=timeframe)
    params.validate_features(cfg.features)
    device = DeviceSelector().resolve()
    logger.info(DeviceSelector().summary("mfd.eval"))

    bundle = load_bundle(cfg)
    ds = StockDataset(bundle.panel, bundle.idx, lookback=cfg.lookback)
    years = pd.to_datetime(bundle.idx.dates).year.astype(int)
    splits = _rolling_splits(cfg, years)

    from mfd.model.groups import (
        LIQUIDITY_FEATURES,
        MOMENTUM_FEATURES,
        PRICE_FEATURES,
        TECHNICAL_FEATURES,
        VOLATILITY_FEATURES,
        FEATURE_ORDER,
    )

    if tuple(cfg.features) != tuple(FEATURE_ORDER):
        raise ValueError("Config features must exactly match FEATURE_ORDER for view masking.")

    p_end = len(PRICE_FEATURES)
    m_end = p_end + len(MOMENTUM_FEATURES)
    v_end = m_end + len(VOLATILITY_FEATURES)
    l_end = v_end + len(LIQUIDITY_FEATURES)
    t_end = l_end + len(TECHNICAL_FEATURES)
    n_feat = t_end

    keep_full = np.ones(n_feat, dtype=np.float32)
    keep_price_mom = np.zeros(n_feat, dtype=np.float32)
    keep_price_mom[:m_end] = 1.0
    keep_vol_liq = np.zeros(n_feat, dtype=np.float32)
    keep_vol_liq[m_end:l_end] = 1.0
    keep_tech = np.zeros(n_feat, dtype=np.float32)
    keep_tech[l_end:t_end] = 1.0

    view_masks = {
        "score_full": keep_full,
        "score_price_mom": keep_price_mom,
        "score_vol_liq": keep_vol_liq,
        "score_tech": keep_tech,
    }

    run_name = build_name(cfg.mode, cfg.rolling_train_years, cfg.rolling_test_years)
    out_root = Path(out) if out is not None else Path(cfg.output_dir)
    out_root = out_root / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, list[pd.DataFrame]] = {k: [] for k in view_masks}
    tickers = pd.Index(bundle.panel.assets.astype(str), name="ticker")

    for tr_years, te_years in splits:
        split_tag = f"{te_years[0]}" if len(te_years) == 1 else f"{te_years[0]}_{te_years[-1]}"
        ckpt = Path(cfg.checkpoint_dir) / run_name / split_tag / "checkpoint0.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for split {split_tag}: {ckpt}")

        idx_all = np.arange(len(ds))
        test_mask = np.isin(years, np.array(te_years, dtype=int))
        test_idx = idx_all[test_mask]

        year_dates = pd.DatetimeIndex(bundle.panel.dates)
        year_dates = year_dates[year_dates.year.isin(te_years)]
        date_pos = {d: i for i, d in enumerate(year_dates)}

        grids = {name: np.full((len(year_dates), len(tickers)), np.nan, dtype=np.float32) for name in view_masks}

        test_ds = torch.utils.data.Subset(ds, test_idx)
        test_loader: DataLoader = torch.utils.data.DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        sample = next(iter(test_loader))["input"]
        model = Transformer(
            n_feat=int(sample.shape[-1]),
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            drop=cfg.drop,
            n_class=2,
            max_len=int(cfg.lookback + 100),
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        model = model.to(device)

        ticker_pos = {t: i for i, t in enumerate(tickers)}
        masks_t = {k: torch.tensor(v, device=device, dtype=torch.float32).view(1, 1, -1) for k, v in view_masks.items()}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"eval[{split_tag}]", leave=False):
                x = batch["input"].to(device=device, dtype=torch.float32)
                bs = int(x.size(0))
                dates_b = pd.to_datetime(list(batch["date"]))
                assets_b = list(batch["asset"])

                for name, m in masks_t.items():
                    x_view = x * m
                    logits = model(x_view)
                    p_up = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    g = grids[name]
                    for i in range(bs):
                        d = pd.Timestamp(dates_b[i])
                        r = date_pos.get(d)
                        c = ticker_pos.get(str(assets_b[i]))
                        if r is None or c is None:
                            continue
                        g[r, c] = float(p_up[i])

        year_dir = out_root / split_tag
        year_dir.mkdir(parents=True, exist_ok=True)
        for name, grid in grids.items():
            df = pd.DataFrame(grid, index=year_dates, columns=tickers)
            df.to_parquet(year_dir / f"{name}.parquet")
            all_scores[name].append(df)
        logger.info("Saved scores for test=%s to %s", split_tag, year_dir)

    out_paths = []
    for name, frames in all_scores.items():
        full = pd.concat(frames).sort_index() if frames else pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)
        path = out_root / f"{name}.parquet"
        full.to_parquet(path)
        out_paths.append(path)
        logger.info("Saved merged %s to %s", name, path)

    return out_paths[0] if out_paths else (out_root / "score_full.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(mode="TEST", timeframe="MEDIUM")
