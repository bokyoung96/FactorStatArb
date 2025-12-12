from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loader import Loader
from models.attention.data import FeatureGroup, FeatureRepository, FactorBatch, FactorResult
from models.attention.model import AttentionFactorLayer, run_layer
from models.attention.writer import FactorResultWriter
from models.util import DeviceSelector
from root import FACTOR_BEST_MODEL_PATH


@dataclass(frozen=True)
class AttentionConfig:
    num_factors: int = 32
    emb_dim: int = 32
    return_col: str = "ret1"
    split_year: int = 2020
    normalize: bool = True
    train_epochs: int = 0
    train_batch_size: int = 32
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-4
    max_train_dates: Optional[int] = None
    max_test_dates: Optional[int] = None
    run_train: bool = True
    run_test: bool = True
    best_model_path: Optional[str] = None

    @classmethod
    def from_file(cls, path: Path) -> "AttentionConfig":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        fields = cls().__dataclass_fields__.keys()
        kwargs = {k: raw[k] for k in raw if k in fields}
        return cls(**kwargs)


class AttentionPipeline:
    def __init__(self, cfg: AttentionConfig, repository: FeatureRepository, device: torch.device) -> None:
        self.cfg = cfg
        self.repository = repository
        self.device = device
        if self.cfg.best_model_path is None:
            object.__setattr__(self.cfg, "best_model_path", str(FACTOR_BEST_MODEL_PATH))
        sample = repository.load_snapshot(repository.latest_date())
        self.model = AttentionFactorLayer(
            num_features=sample.features.size(1),
            num_factors=cfg.num_factors,
            emb_dim=cfg.emb_dim,
        ).to(device)
        self.writer = FactorResultWriter()

    def _normalize(self, batch: FactorBatch) -> torch.Tensor:
        if not batch.norm_idx:
            feats = batch.features
        else:
            feats = batch.features.clone()
            feats[:, batch.norm_idx] = (feats[:, batch.norm_idx] - batch.mean[batch.norm_idx]) / batch.std[batch.norm_idx]
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats * batch.mask.unsqueeze(-1)

    def _run_single(self, date, save: bool, split: str) -> FactorResult:
        batch = self.repository.load_snapshot(date)
        feats = self._normalize(batch) if self.cfg.normalize else batch.features
        result = run_layer(self.model, FactorBatch(**{**batch.__dict__, "features": feats}))
        if save:
            self.writer.save(date, batch.assets, result, split=split)
        return result

    def run_dates(self, dates: Sequence, save: bool, split: str, buffer_size: int = 50) -> None:
        norm_bar = tqdm(dates, desc=f"normalize+forward[{split}]")
        if not save:
            for d in norm_bar:
                self._run_single(d, save=False, split=split)
            return

        buffer = []
        save_bar = tqdm(total=len(dates), desc=f"save[{split}]", leave=False)
        for d in norm_bar:
            res = self._run_single(d, save=False, split=split)
            batch = self.repository.load_snapshot(d)
            buffer.append((d, batch.assets, res))
            if len(buffer) >= buffer_size:
                self.writer.save_batch(buffer, split=split)
                save_bar.update(len(buffer))
                buffer.clear()
        if buffer:
            self.writer.save_batch(buffer, split=split)
            save_bar.update(len(buffer))
        save_bar.close()

    def run_latest(self, save: bool = True, split: str = "full") -> FactorResult:
        return self._run_single(self.repository.latest_date(), save=save, split=split)

    def train(self, dates: Sequence, epochs: int, lr: float, weight_decay: float) -> None:
        if not dates or epochs <= 0:
            return
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss: Optional[float] = None
        best_path = Path(self.cfg.best_model_path) if self.cfg.best_model_path else None
        for ep in range(epochs):
            bar = tqdm(dates, desc=f"train epoch {ep+1}/{epochs}")
            total_loss = 0.0
            total_count = 0
            epoch_grad_means = []
            for d in bar:
                batch = self.repository.load_snapshot(d)
                feats = self._normalize(batch) if self.cfg.normalize else batch.features
                pred = self.model(feats, batch.returns, batch.mask)[2]
                valid = batch.mask > 0
                if not valid.any():
                    continue
                loss = torch.nn.functional.mse_loss(pred[valid], batch.returns[valid])
                opt.zero_grad()
                loss.backward()
                grads = [p.grad.abs().mean().item() for p in self.model.parameters() if p.grad is not None]
                if grads:
                    epoch_grad_means.append(sum(grads) / len(grads))
                opt.step()
                total_loss += loss.item() * int(valid.sum().item())
                total_count += int(valid.sum().item())
                bar.set_postfix({"loss": f"{loss.item():.8f}"})
            if total_count > 0:
                avg_loss = total_loss / total_count
                grad_mean = sum(epoch_grad_means) / len(epoch_grad_means) if epoch_grad_means else 0.0
                logging.info("Epoch %d avg loss %.8f avg|grad| %.8e", ep + 1, avg_loss, grad_mean)
                avg_loss = total_loss / total_count
                if best_path is not None and (best_loss is None or avg_loss < best_loss):
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), best_path)


def _train_test_split(dates: Sequence, split_year: int) -> Tuple[List, List]:
    ordered = pd.to_datetime(pd.Index(dates)).sort_values()
    train = [d for d in ordered if d.year <= split_year]
    test = [d for d in ordered if d.year > split_year]
    return train, test


def main(config_path: Optional[Path] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg_path = config_path or Path(__file__).with_name("config.json")
    cfg = AttentionConfig.from_file(cfg_path)
    device = DeviceSelector().resolve()

    repo = FeatureRepository(loader=Loader(), device=device, return_col=cfg.return_col, groups=None)
    pipeline = AttentionPipeline(cfg=cfg, repository=repo, device=device)

    dates = list(repo.available_dates())
    train_dates, test_dates = _train_test_split(dates, cfg.split_year)
    if cfg.max_train_dates is not None:
        train_dates = train_dates[: cfg.max_train_dates]
    if cfg.max_test_dates is not None:
        test_dates = test_dates[: cfg.max_test_dates]

    if train_dates:
        if cfg.run_train:
            pipeline.train(train_dates, epochs=cfg.train_epochs, lr=cfg.train_lr, weight_decay=cfg.train_weight_decay)
        pipeline.run_dates(train_dates, save=True, split="train")
    if cfg.run_test and test_dates:
        pipeline.run_dates(test_dates, save=True, split="test")


if __name__ == "__main__":
    main()
