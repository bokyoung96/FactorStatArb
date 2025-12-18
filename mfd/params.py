import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TransformerConfig:
    mode: str
    batch_size: int
    max_epoch: int
    lr: float
    lookback: int
    stride: int
    horizon: int
    min_assets: int
    min_valid: float
    rolling_train_years: int
    rolling_test_years: int
    rolling_step_years: int
    d_model: int
    nhead: int
    n_layers: int
    d_ff: int
    drop: float
    features: Tuple[str, ...]
    norm: str
    label_type: str
    threshold: float
    cache_dir: Path
    output_dir: Path
    checkpoint_dir: Path


class TransformerParams:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path is not None else Path(__file__).parent / "config.json"
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

    def get_config(self, mode: str = "TEST", timeframe: str = "MEDIUM") -> TransformerConfig:
        mode_cfg = self.config["mode_configs"][mode]
        tf_cfg = self.config["timeframe_configs"][timeframe]
        model_cfg = tf_cfg["model"]
        rolling_cfg = self.config.get("rolling", {})
        cache_dir = (ROOT_DIR / Path(self.config.get("cache_dir", "mfd/DATA"))).resolve()
        output_dir = (ROOT_DIR / Path(self.config.get("output_dir", "mfd/out"))).resolve()
        checkpoint_dir = (ROOT_DIR / Path(self.config.get("checkpoint_dir", "mfd/models"))).resolve()
        return TransformerConfig(
            mode=f"{mode_cfg['mode']}_{timeframe.lower()}",
            batch_size=int(mode_cfg["batch_size"]),
            max_epoch=int(mode_cfg["max_epoch"]),
            lr=float(mode_cfg["lr"]),
            lookback=int(tf_cfg["lookback"]),
            stride=int(tf_cfg["stride"]),
            horizon=int(tf_cfg["horizon"]),
            min_assets=int(tf_cfg["min_assets"]),
            min_valid=float(tf_cfg.get("min_valid", 0.95)),
            rolling_train_years=int(rolling_cfg.get("train_years", 5)),
            rolling_test_years=int(rolling_cfg.get("test_years", 1)),
            rolling_step_years=int(rolling_cfg.get("step_years", 1)),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            n_layers=int(model_cfg["n_layers"]),
            d_ff=int(model_cfg["d_ff"]),
            drop=float(model_cfg["drop"]),
            features=tuple(self.config["features"]),
            norm=str(self.config.get("norm", "none")),
            label_type=str(self.config.get("label_type", "classification")),
            threshold=float(self.config.get("threshold", 0.0)),
            cache_dir=cache_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

    def validate_features(self, features: Tuple[str, ...]) -> None:
        from mfd.model.groups import FEATURE_ORDER

        if tuple(features) != tuple(FEATURE_ORDER):
            raise ValueError(
                "mfd feature order mismatch: config features must exactly match FEATURE_ORDER "
                "(grouped hard-coded order)."
            )

    @property
    def modes(self) -> List[str]:
        return list(self.config["mode_configs"].keys())

    @property
    def timeframes(self) -> List[str]:
        return list(self.config["timeframe_configs"].keys())


def build_name(mode: str, train_years: int, test_years: int, model_type: str = "transformer", base: str = "mfd") -> str:
    return f"{base}_{model_type.lower()}_{mode.lower()}_{int(train_years)}_{int(test_years)}"
