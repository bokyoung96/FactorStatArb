from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT_DIR = REPO_ROOT / "models"

from attention import (
    AttentionConfig,
    StatArbRunner,
    StatArbTrainer,
    ResidualCache,
    FeatureRepository,
    FactorResultWriter,
)
from loader import Loader
from models.util import DeviceSelector
from root import FACTOR_DIR


def main(config_path: Optional[Path] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg_path = config_path or (ROOT_DIR / "config.json")
    cfg = AttentionConfig.from_file(cfg_path)
    device = DeviceSelector().resolve()

    repo = FeatureRepository(loader=Loader(), device=device, return_col=cfg.return_col, groups=None)
    dates = list(pd.to_datetime(repo.available_dates()).sort_values())

    window_years = cfg.window_years
    step_years = cfg.step_years
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else FACTOR_DIR / "cache"
    best_path = Path(cfg.best_model_path) if cfg.best_model_path else FACTOR_DIR / "best_longconv.pt"
    writer = FactorResultWriter()
    cache = ResidualCache(cache_dir=cache_dir)

    years = sorted({d.year for d in dates})
    for start in range(0, len(years) - window_years):
        train_years = years[start : start + window_years]
        test_years = years[start + window_years : start + window_years + step_years]
        train_dates = [d for d in dates if d.year in train_years]
        test_dates = [d for d in dates if d.year in test_years]
        if not train_dates or not test_dates:
            continue
        logging.info("Rolling window train years %s test years %s (train days=%d, test days=%d)", train_years, test_years, len(train_dates), len(test_dates))
        trainer = StatArbTrainer(cfg=cfg, repository=repo, device=device, writer=writer, cache=cache, best_path=best_path)
        if cfg.run_train:
            trainer.train(train_dates)
        if best_path.exists():
            trainer.model.load_state_dict(torch.load(best_path, map_location=device))
        runner = StatArbRunner(
            cfg=cfg,
            repository=repo,
            device=device,
            writer=writer,
            cache=cache,
            model=trainer.model,
        )
        split_name = f"test_{train_years[-1]}_{test_years[-1]}"
        logging.info("Start inference split=%s", split_name)
        runner.run(test_dates, split=split_name)
        logging.info("Finished split=%s", split_name)


if __name__ == "__main__":
    main()
