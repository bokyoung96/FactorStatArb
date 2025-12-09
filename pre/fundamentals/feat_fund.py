from __future__ import annotations

from pre.base.feature import F, add_feature
from pre.base.reg import FE, register
from pre.base.util import safe_div


@register(FE.BM)
class BM(F):
    def run(self, df):
        bm = safe_div(df["equity"], df["mcap"], allow_negative_denom=False)
        return add_feature(df, FE.BM.value, bm)


@register(FE.EP)
class EP(F):
    def run(self, df):
        ep = safe_div(df["ni_ttm"], df["mcap"], allow_negative_denom=False)
        return add_feature(df, FE.EP.value, ep)


@register(FE.ROE)
class ROE(F):
    def run(self, df):
        roe = safe_div(df["ni_ttm"], df["equity_avg"], allow_negative_denom=False)
        return add_feature(df, FE.ROE.value, roe)


@register(FE.GP_A)
class GP_A(F):
    def run(self, df):
        gp_a = safe_div(df["gp_ttm"], df["assets_avg"], allow_negative_denom=False)
        return add_feature(df, FE.GP_A.value, gp_a)


@register(FE.ACC)
class ACC(F):
    def run(self, df):
        acc_num = df["ni_ttm"] - df["ocf_ttm"]
        acc = safe_div(acc_num, df["assets_avg"], allow_negative_denom=False)
        return add_feature(df, FE.ACC.value, acc)


@register(FE.OPM)
class OPM(F):
    def run(self, df):
        if "rev_ttm" not in df:
            return df
        opm = safe_div(df["op_ttm"], df["rev_ttm"], allow_negative_denom=False)
        return add_feature(df, FE.OPM.value, opm)


@register(FE.SG)
class SG(F):
    def run(self, df):
        if "rev_g" not in df:
            return df
        return add_feature(df, FE.SG.value, df["rev_g"])


@register(FE.AG)
class AG(F):
    def run(self, df):
        return add_feature(df, FE.AG.value, df["assets_g"])


@register(FE.LEV)
class LEV(F):
    def run(self, df):
        lev = safe_div(df["liab"], df["assets"], allow_negative_denom=False)
        return add_feature(df, FE.LEV.value, lev)


@register(FE.TURN)
class TURN(F):
    def run(self, df):
        turn = safe_div(df["op_ttm"], df["assets_avg"], allow_negative_denom=False)
        return add_feature(df, FE.TURN.value, turn)
