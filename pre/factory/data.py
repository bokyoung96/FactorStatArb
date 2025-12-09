from __future__ import annotations

import pandas as pd
import logging

from pre.base.duck import DB
from root import CONS_PARQUET, FUND_PARQUET, PRICE_PARQUET, SECTOR_PARQUET
from root import FEATURES_PRICE_PARQUET, FEATURES_FUND_PARQUET, FEATURES_CONS_PARQUET, FEATURES_SECTOR_PARQUET, FEATURES_DIR


class Data:
    def __init__(self, load, align, build):
        self.load = load
        self.align = align
        self.build = build
        self.db = DB()
        self.log = logging.getLogger(__name__)

    def make(self, table: str = "features"):
        price = self.load.price()
        idx = price["c"].index

        fund_src = self.load.fundamentals()
        base_q = {
            "assets": fund_src["a"].resample("QE").last(),
            "liab": fund_src["l"].resample("QE").last(),
            "equity": fund_src["e"].resample("QE").last(),
            "gp": fund_src["gp"].resample("QE").last(),
            "op": fund_src["op"].resample("QE").last(),
            "ni": fund_src["ni"].resample("QE").last(),
            "ocf": fund_src["ocf"].resample("QE").last(),
        }
        flows = ("ni", "op", "gp", "ocf")
        ttm = {
            f"{name}_ttm": base_q[name].rolling(4, min_periods=4).sum()
            for name in flows
            if name in base_q
        }
        stock_avg = {
            "equity_avg": (base_q["equity"] + base_q["equity"].shift(1)) / 2,
            "assets_avg": (base_q["assets"] + base_q["assets"].shift(1)) / 2,
        }
        growth = {"assets_g": base_q["assets"].pct_change(4, fill_method=None)}

        fund_q = {**base_q, **ttm, **stock_avg, **growth}
        fundamentals = {k: self.align.daily(v, idx) for k, v in fund_q.items()}
        consensus = {k: self.align.daily(v, idx) for k, v in self.load.consensus().items()}
        sectors = self.align.daily(self.load.sector(), idx)

        price_df = pd.concat(
            [
                price["c"],
                price["o"],
                price["h"],
                price["l"],
                price["v"],
                price["mc"],
            ],
            axis=1,
            keys=["close", "open", "high", "low", "vol", "mcap"],
        )
        fund_keys = [
            "assets",
            "liab",
            "equity",
            "gp_ttm",
            "op_ttm",
            "ni_ttm",
            "ocf_ttm",
            "equity_avg",
            "assets_avg",
            "assets_g",
        ]
        existing = [k for k in fund_keys if k in fundamentals]
        fund_df = pd.concat([fundamentals[k] for k in existing], axis=1, keys=existing)
        cons_df = pd.concat(
            [
                consensus["op_fq1_raw"],
                consensus["op_fq2_raw"],
                consensus["op_fy1_raw"],
                consensus["eps_fq1_raw"],
                consensus["eps_fq2_raw"],
                consensus["eps_fy1_raw"],
            ],
            axis=1,
            keys=[
                "op_fq1_raw",
                "op_fq2_raw",
                "op_fy1_raw",
                "eps_fq1_raw",
                "eps_fq2_raw",
                "eps_fy1_raw",
            ],
        )
        sector_df = pd.concat([sectors], axis=1, keys=["sector"])

        self.log.info("Saving raw price to %s", PRICE_PARQUET)
        self.db.save_parquet(price_df, PRICE_PARQUET)
        self.log.info("Saving raw fundamentals to %s", FUND_PARQUET)
        self.db.save_parquet(fund_df, FUND_PARQUET)
        self.log.info("Saving raw consensus to %s", CONS_PARQUET)
        self.db.save_parquet(cons_df, CONS_PARQUET)
        self.log.info("Saving raw sector to %s", SECTOR_PARQUET)
        self.db.save_parquet(sector_df, SECTOR_PARQUET)

        df = pd.concat(
            [price_df, fund_df, cons_df, sector_df],
            axis=1,
        )

        feats = [f.value if hasattr(f, "value") else str(f) for f in self.build.feats]
        self.log.info("Applying %d features: %s", len(feats), feats)
        df = self.build.run(df)
        self.save_features(df)
        self.log.info("Saved features to %s", FEATURES_DIR)

    def save_features(self, df: pd.DataFrame):
        feature_groups = {
            "price": {
                "names": {
                    "ret1",
                    "ret5",
                    "ret10",
                    "m1",
                    "m3",
                    "m6",
                    "m12",
                    "m1_va",
                    "m12_va",
                    "rev5",
                    "vol5",
                    "vol10",
                    "vol20",
                    "vol60",
                    "vol120",
                    "hlr",
                    "idv",
                    "trange",
                    "volz",
                    "volma20",
                    "volma_r",
                    "turnover",
                    "amihud",
                    "sprd",
                    "pimpact",
                    "volshock",
                    "ma5",
                    "ma20",
                    "ma60",
                    "ma120",
                    "macd",
                    "macds",
                    "rsi14",
                    "sto_k",
                    "sto_d",
                    "boll_up",
                    "boll_low",
                    "boll_w",
                    "high52",
                    "low52",
                    "price_z",
                    "dist_ma20",
                    "breakout",
                },
                "path": FEATURES_PRICE_PARQUET,
            },
            "fundamentals": {
                "names": {
                    "bm",
                    "ep",
                    "roe",
                    "gp_a",
                    "acc",
                    "opm",
                    "sg",
                    "ag",
                    "lev",
                    "turn",
                },
                "path": FEATURES_FUND_PARQUET,
            },
            "consensus": {
                "names": {
                    "op_fq1",
                    "op_fq2",
                    "op_fy1",
                    "eps_fq1",
                    "eps_fq2",
                    "eps_fy1",
                    "rev_op_fq1",
                    "rev_op_fq2",
                    "rev_op_fy1",
                    "rev_eps_fq1",
                    "rev_eps_fq2",
                },
                "path": FEATURES_CONS_PARQUET,
            },
            "sector": {
                "names": {
                    "sector_oh",
                    "sector_id",
                },
                "path": FEATURES_SECTOR_PARQUET,
            },
        }

        for group_name, cfg in feature_groups.items():
            names = cfg["names"]
            cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    if c[0] in names:
                        cols.append(c)
                elif c in names:
                    cols.append(c)
            if not cols:
                continue
            subset = df.loc[:, cols]
            self.db.save_parquet(subset, cfg["path"])
