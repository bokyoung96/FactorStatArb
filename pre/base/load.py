from __future__ import annotations

from pathlib import Path

import pandas as pd

from pre.base.util import ensure_datetime_index
from root import RAW_DIR


class Load:
    def __init__(self, path: str | Path = None):
        self.p = Path(path) if path is not None else RAW_DIR

    def csv(self, name: str) -> pd.DataFrame:
        df = pd.read_csv(self.p / f"{name}.csv", low_memory=False)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
        return ensure_datetime_index(df)

    def price(self):
        return {
            "c": self.csv("qw_adj_c"),
            "o": self.csv("qw_adj_o"),
            "h": self.csv("qw_adj_h"),
            "l": self.csv("qw_adj_l"),
            "v": self.csv("qw_v"),
            "mc": self.csv("qw_mktcap"),
        }

    def fundamentals(self):
        return {
            "a": self.csv("qw_asset_lfq0"),
            "l": self.csv("qw_liability_lfq0"),
            "e": self.csv("qw_equity_lfq0"),
            "gp": self.csv("qw_gp_lfq0"),
            "op": self.csv("qw_op_lfq0"),
            "ni": self.csv("qw_ni_lfq0"),
            "ocf": self.csv("qw_ocf_lfq0"),
        }

    def consensus(self):
        return {
            "op_fq1_raw": self.csv("qw_op_nfq1"),
            "op_fq2_raw": self.csv("qw_op_nfq2"),
            "op_fy1_raw": self.csv("qw_op_nfy1"),
            "eps_fq1_raw": self.csv("qw_eps_nfq1"),
            "eps_fq2_raw": self.csv("qw_eps_nfq2"),
            "eps_fy1_raw": self.csv("qw_eps_nfy1"),
        }

    def sector(self):
        return self.csv("qw_wics_sec_big")
