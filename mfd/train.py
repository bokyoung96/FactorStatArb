from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
    if tr <= 0 or te <= 0 or st <= 0:
        raise ValueError("rolling years must be positive.")
    out = []
    for start in range(0, len(uniq) - tr - te + 1, st):
        train_years = uniq[start : start + tr]
        test_years = uniq[start + tr : start + tr + te]
        out.append((train_years, test_years))
    if not out:
        raise ValueError("Not enough years for rolling split.")
    return out


def _train_one(
    cfg: TransformerConfig,
    *,
    train_years: list[int],
    test_years: list[int],
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> Path:
    device = DeviceSelector().resolve()
    logger.info(DeviceSelector().summary("mfd.train"))
    bundle = load_bundle(cfg)
    ds = StockDataset(bundle.panel, bundle.idx, lookback=cfg.lookback)
    years = pd.to_datetime(bundle.idx.dates).year.astype(int)
    idx = np.arange(len(ds))
    train_mask = np.isin(years, np.array(train_years, dtype=int))
    test_mask = np.isin(years, np.array(test_years, dtype=int))
    train_idx = idx[train_mask]
    test_idx = idx[test_mask]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Empty train/test split; check rolling years.")

    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    y_all = bundle.idx.targets
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    train_pos = float((y_train > 0.5).mean()) if y_train.size else float("nan")
    test_pos = float((y_test > 0.5).mean()) if y_test.size else float("nan")
    logger.info(
        "split sizes train=%d test=%d pos_rate train=%.3f test=%.3f",
        int(train_idx.size),
        int(test_idx.size),
        train_pos,
        test_pos,
    )

    sample = next(iter(train_loader))["input"]
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
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(lr or cfg.lr))
    max_epoch = int(epochs or cfg.max_epoch)

    split_tag = f"{test_years[0]}" if len(test_years) == 1 else f"{test_years[0]}_{test_years[-1]}"
    run_name = build_name(cfg.mode, cfg.rolling_train_years, cfg.rolling_test_years)
    out_dir = Path(cfg.checkpoint_dir) / run_name / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "checkpoint0.pth"
    best_loss = float("inf")

    for ep in range(max_epoch):
        model.train()
        run_loss = 0.0
        run_correct = 0
        n = 0
        pbar = tqdm(train_loader, desc=f"train {ep+1}/{max_epoch}", leave=False)
        for batch in pbar:
            x = batch["input"].to(device=device, dtype=torch.float32)
            y = batch["label"].to(device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            preds = torch.argmax(logits, dim=1)
            bs = int(x.size(0))
            run_loss += float(loss.item()) * bs
            run_correct += int((preds == y).sum().item())
            n += bs
            pbar.set_postfix({"loss": float(loss.item())})
        train_loss = run_loss / max(n, 1)
        train_acc = run_correct / max(n, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        vn = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"val {ep+1}/{max_epoch}", leave=False):
                x = batch["input"].to(device=device, dtype=torch.float32)
                y = batch["label"].to(device=device, dtype=torch.long)
                logits = model(x)
                loss = crit(logits, y)
                preds = torch.argmax(logits, dim=1)
                bs = int(x.size(0))
                val_loss += float(loss.item()) * bs
                val_correct += int((preds == y).sum().item())
                vn += bs
        val_loss = val_loss / max(vn, 1)
        val_acc = val_correct / max(vn, 1)
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f",
            ep + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("saved best checkpoint to %s (val_loss=%.6f val_acc=%.4f)", best_path, best_loss, val_acc)

    return best_path


def train(
    *,
    mode: str = "TEST",
    timeframe: str = "MEDIUM",
    config_path: Optional[Path] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> list[Path]:
    params = TransformerParams(config_path=config_path)
    cfg = params.get_config(mode=mode, timeframe=timeframe)
    params.validate_features(cfg.features)
    bundle = load_bundle(cfg)
    years = pd.to_datetime(bundle.idx.dates).year.astype(int)
    splits = _rolling_splits(cfg, years)
    out = []
    for tr_years, te_years in splits:
        logger.info("rolling train=%s test=%s", tr_years, te_years)
        out.append(_train_one(cfg, train_years=tr_years, test_years=te_years, epochs=epochs, lr=lr))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(mode="TEST", timeframe="MEDIUM")
