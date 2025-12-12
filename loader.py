from __future__ import annotations

import logging
import ast
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pre.base.duck import DB
from root import (
    CONS_PARQUET,
    FUND_PARQUET,
    PRICE_PARQUET,
    SECTOR_PARQUET,
    FEATURES_PRICE_PARQUET,
    FEATURES_FUND_PARQUET,
    FEATURES_CONS_PARQUET,
    FEATURES_SECTOR_PARQUET,
    UNIVERSE_PARQUET,
)


class Loader:
    def __init__(self):
        self.db = DB()

    def _to_multiindex(self, df):
        date_cols = [c for c in df.columns if str(c).lower() in {"date", "index", "level_0"} or str(c).startswith("Unnamed")]
        if date_cols:
            date_col = date_cols[0]
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = "date"
            drop_cols = [
                c for c in df.columns if str(c).startswith("Unnamed") or str(c).lower() in {"index", "level_0"}
            ]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        parsed = []
        tuple_found = False
        for c in df.columns:
            if isinstance(c, tuple):
                parsed.append(c)
                tuple_found = True
                continue
            if isinstance(c, str) and c.startswith("(") and c.endswith(")"):
                try:
                    val = ast.literal_eval(c)
                    parsed.append(val)
                    tuple_found = tuple_found or isinstance(val, tuple)
                    continue
                except Exception:
                    pass
            parsed.append(c)
        if tuple_found:
            df = df.copy()
            df.columns = pd.MultiIndex.from_tuples(parsed)
        return df

    def _first_level(self, col) -> str:
        if isinstance(col, tuple):
            return col[0]
        if isinstance(col, str) and col.startswith("(") and col.endswith(")"):
            try:
                val = ast.literal_eval(col)
                if isinstance(val, tuple):
                    return val[0]
            except Exception:
                pass
        return col

    def _quote(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    RAW_MAP = {
        "price": PRICE_PARQUET,
        "fundamentals": FUND_PARQUET,
        "consensus": CONS_PARQUET,
        "sector": SECTOR_PARQUET,
        "universe": UNIVERSE_PARQUET,
    }

    FEAT_MAP = {
        "features_price": FEATURES_PRICE_PARQUET,
        "features_fundamentals": FEATURES_FUND_PARQUET,
        "features_consensus": FEATURES_CONS_PARQUET,
        "features_sector": FEATURES_SECTOR_PARQUET,
    }

    def _select(self, path: Path, cols: Optional[Sequence[str]] = None):
        schema_cols = self.db.con.execute(f"SELECT * FROM parquet_scan('{path}') LIMIT 0").df().columns
        if cols:
            target = set(cols)
            keep = [c for c in schema_cols if self._first_level(c) in target]
        else:
            keep = list(schema_cols)
        if not keep:
            return pd.DataFrame()
        clause = ", ".join(self._quote(str(c)) for c in keep)
        df = self.db.q(f"SELECT {clause} FROM parquet_scan('{path}')")
        return self._to_multiindex(df)

    def raw(self, table: str = "price", cols: Optional[Sequence[str]] = None):
        if table not in self.RAW_MAP:
            raise ValueError(f"Unknown raw table: {table}")
        return self._select(Path(self.RAW_MAP[table]), cols)

    def price(self, cols: Optional[Sequence[str]] = None):
        return self.raw("price", cols)

    def fundamentals(self, cols: Optional[Sequence[str]] = None):
        return self.raw("fundamentals", cols)

    def consensus(self, cols: Optional[Sequence[str]] = None):
        return self.raw("consensus", cols)

    def sector(self, cols: Optional[Sequence[str]] = None):
        return self.raw("sector", cols)

    def universe(self, name: str = "k200", cols: Optional[Sequence[str]] = None):
        return self.raw("universe", cols)

    def features(self, table: str = "features_price", cols: Optional[Sequence[str]] = None):
        if table not in self.FEAT_MAP:
            raise ValueError(f"Unknown features table: {table}")
        return self._select(Path(self.FEAT_MAP[table]), cols)

    def features_price(self, cols: Optional[Sequence[str]] = None):
        return self.features("features_price", cols)

    def features_fundamentals(self, cols: Optional[Sequence[str]] = None):
        return self.features("features_fundamentals", cols)

    def features_consensus(self, cols: Optional[Sequence[str]] = None):
        return self.features("features_consensus", cols)

    def features_sector(self, cols: Optional[Sequence[str]] = None):
        return self.features("features_sector", cols)

    def feature_subset(self, table: str = "features_price", cols: Optional[Sequence[str]] = None):
        return self.features(table, cols)

    def raw_subset(self, table: str = "price", cols: Optional[Sequence[str]] = None):
        return self.raw(table, cols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ldr = Loader()

    raw = ldr.raw("price")
    feat = ldr.features("features_price")

