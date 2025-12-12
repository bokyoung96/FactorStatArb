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

    def run_dates(self, dates: Sequence, save: bool, split: str) -> None:
        norm_bar = tqdm(dates, desc=f"normalize+forward[{split}]")
        save_bar = tqdm(total=len(dates), desc=f"save[{split}]", leave=False) if save else None
        for d in norm_bar:
            self._run_single(d, save=save, split=split)
            if save_bar is not None:
                save_bar.update(1)
        if save_bar is not None:
            save_bar.close()

    def run_latest(self, save: bool = True, split: str = "full") -> FactorResult:
        return self._run_single(self.repository.latest_date(), save=save, split=split)


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
        pipeline.run_dates(train_dates, save=True, split="train")
    if test_dates:
        pipeline.run_dates(test_dates, save=True, split="test")


if __name__ == "__main__":
    main()
