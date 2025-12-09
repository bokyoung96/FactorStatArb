from __future__ import annotations

from typing import Iterable

from tqdm import tqdm

from pre.base.reg import REG


class Build:
    def __init__(self, feats: Iterable):
        self.feats = feats

    def run(self, df):
        for f in tqdm(self.feats, desc="features"):
            df = REG[f]().run(df)
        return df
