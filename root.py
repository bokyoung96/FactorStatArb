from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DUCKDB_PATH = PROCESSED_DIR / "features.db"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_PRICE_PARQUET = FEATURES_DIR / "features_price.parquet"
FEATURES_FUND_PARQUET = FEATURES_DIR / "features_fundamentals.parquet"
FEATURES_CONS_PARQUET = FEATURES_DIR / "features_consensus.parquet"
FEATURES_SECTOR_PARQUET = FEATURES_DIR / "features_sector.parquet"
PRICE_PARQUET = PROCESSED_DIR / "price.parquet"
FUND_PARQUET = PROCESSED_DIR / "fundamentals.parquet"
CONS_PARQUET = PROCESSED_DIR / "consensus.parquet"
SECTOR_PARQUET = PROCESSED_DIR / "sector.parquet"
UNIVERSE_PARQUET = PROCESSED_DIR / "universe_k200.parquet"  # default K200, extendable
FACTOR_DIR = DATA_DIR / "factor"
FACTOR_WEIGHTS_PARQUET = FACTOR_DIR / "factor_weights.parquet"
FACTOR_RETURNS_PARQUET = FACTOR_DIR / "factor_returns.parquet"
FACTOR_PRED_PARQUET = FACTOR_DIR / "factor_predicted.parquet"
FACTOR_RES_PARQUET = FACTOR_DIR / "factor_residuals.parquet"
FACTOR_BEST_MODEL_PATH = FACTOR_DIR / "best_model.pt"


def _freq_bucket(freq: str) -> str:
    f = str(freq or "D").strip().upper()
    if f in {"D", "B", "DAILY"}:
        return "d"
    if f in {"W", "W-FRI", "W-MON", "WEEKLY"} or f.startswith("W"):
        return "w"
    if f in {"M", "ME", "MONTHLY"} or f.startswith("M"):
        return "m"
    return f.lower().replace("/", "_").replace("\\", "_").replace(" ", "_")


def factor_dir_freq(freq: str) -> Path:
    return DATA_DIR / f"factor_{_freq_bucket(freq)}"
