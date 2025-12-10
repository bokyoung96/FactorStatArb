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
)


class Loader:
    def __init__(self):
        self.db = DB()

    def _view(self, name: str, path) -> None:
        self.db.q(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{Path(path)}')")

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

    def price(self):
        self._view("price", PRICE_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM price"))

    def fundamentals(self):
        self._view("fundamentals", FUND_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM fundamentals"))

    def consensus(self):
        self._view("consensus", CONS_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM consensus"))

    def sector(self):
        self._view("sector", SECTOR_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM sector"))

    def features_price(self):
        self._view("features_price", FEATURES_PRICE_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM features_price"))

    def features_fundamentals(self):
        self._view("features_fundamentals", FEATURES_FUND_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM features_fundamentals"))

    def features_consensus(self):
        self._view("features_consensus", FEATURES_CONS_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM features_consensus"))

    def features_sector(self):
        self._view("features_sector", FEATURES_SECTOR_PARQUET)
        return self._to_multiindex(self.db.q("SELECT * FROM features_sector"))

    def feature_subset(self, table: str = "features_price", cols: Optional[Sequence[str]] = None):
        path_map = {
            "features_price": FEATURES_PRICE_PARQUET,
            "features_fundamentals": FEATURES_FUND_PARQUET,
            "features_consensus": FEATURES_CONS_PARQUET,
            "features_sector": FEATURES_SECTOR_PARQUET,
        }
        if table not in path_map:
            raise ValueError(f"Unknown table: {table}")

        path = Path(path_map[table])
        schema_cols = self.db.con.execute(f"SELECT * FROM parquet_scan('{path}') LIMIT 0").df().columns
        if cols:
            target = set(cols)
            keep = [c for c in schema_cols if self._first_level(c) in target]
        else:
            keep = list(schema_cols)

        if not keep:
            return pd.DataFrame()

        select_clause = ", ".join(self._quote(str(c)) for c in keep)
        df = self.db.q(f"SELECT {select_clause} FROM parquet_scan('{path}')")
        return self._to_multiindex(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loader = Loader()

    table = "features_consensus"

    df = getattr(loader, table)()

    logging.info("[%s] shape: %s", table, df.shape)
