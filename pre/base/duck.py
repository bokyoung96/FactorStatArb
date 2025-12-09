from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from root import DUCKDB_PATH

class DB:
    def __init__(self, path: str | Path = DUCKDB_PATH):
        self.con = duckdb.connect(str(path))

    def q(self, sql: str):
        return self.con.execute(sql).df()

    def save_parquet(self, df: Any, path: str | Path = DUCKDB_PATH):
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        idx_name = df.index.name or "date"
        df = df.copy()
        df.index.name = idx_name
        if not all(isinstance(c, str) for c in df.columns):
            df.columns = [str(c) for c in df.columns]
        df.to_parquet(dest, index=True)

    def save_table(self, df: Any, name: str):
        self.con.register("df", df)
        self.con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
