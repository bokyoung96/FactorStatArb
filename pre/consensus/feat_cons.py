from __future__ import annotations

from pre.base.feature import F, add_feature
from pre.base.reg import FE, register
from pre.base.util import safe_div
from pre.consensus.cal import rev_ffill_pct, turn_flag


@register(FE.OP_FQ1)
class OP_FQ1(F):
    def run(self, df):
        return df


@register(FE.OP_FQ2)
class OP_FQ2(F):
    def run(self, df):
        return df


@register(FE.OP_FY1)
class OP_FY1(F):
    def run(self, df):
        return df


@register(FE.EPS_FQ1)
class EPS_FQ1(F):
    def run(self, df):
        return df


@register(FE.EPS_FQ2)
class EPS_FQ2(F):
    def run(self, df):
        return df


@register(FE.EPS_FY1)
class EPS_FY1(F):
    def run(self, df):
        return df


@register(FE.OP_FQ_SPRD)
class OP_FQ_SPRD(F):
    def run(self, df):
        delta = df["op_fq2_raw"] - df["op_fq1_raw"]
        sprd = safe_div(delta, df["op_fq1_raw"].abs(), allow_negative_denom=True)
        return add_feature(df, FE.OP_FQ_SPRD.value, sprd)


@register(FE.EPS_FQ_SPRD)
class EPS_FQ_SPRD(F):
    def run(self, df):
        delta = df["eps_fq2_raw"] - df["eps_fq1_raw"]
        sprd = safe_div(delta, df["eps_fq1_raw"].abs(), allow_negative_denom=True)
        return add_feature(df, FE.EPS_FQ_SPRD.value, sprd)


@register(FE.OP_FQ_TURN)
class OP_FQ_TURN(F):
    def run(self, df):
        turn = turn_flag(df["op_fq1_raw"])
        return add_feature(df, FE.OP_FQ_TURN.value, turn)


@register(FE.EPS_FQ_TURN)
class EPS_FQ_TURN(F):
    def run(self, df):
        turn = turn_flag(df["eps_fq1_raw"])
        return add_feature(df, FE.EPS_FQ_TURN.value, turn)


@register(FE.REV_OP_FQ1)
class REV_OP_FQ1(F):
    def run(self, df):
        rev = rev_ffill_pct(df["op_fq1_raw"])
        return add_feature(df, FE.REV_OP_FQ1.value, rev)


@register(FE.REV_OP_FQ2)
class REV_OP_FQ2(F):
    def run(self, df):
        rev = rev_ffill_pct(df["op_fq2_raw"])
        return add_feature(df, FE.REV_OP_FQ2.value, rev)


@register(FE.REV_OP_FY1)
class REV_OP_FY1(F):
    def run(self, df):
        rev = rev_ffill_pct(df["op_fy1_raw"])
        return add_feature(df, FE.REV_OP_FY1.value, rev)


@register(FE.REV_EPS_FQ1)
class REV_EPS_FQ1(F):
    def run(self, df):
        rev = rev_ffill_pct(df["eps_fq1_raw"])
        return add_feature(df, FE.REV_EPS_FQ1.value, rev)


@register(FE.REV_EPS_FQ2)
class REV_EPS_FQ2(F):
    def run(self, df):
        rev = rev_ffill_pct(df["eps_fq2_raw"])
        return add_feature(df, FE.REV_EPS_FQ2.value, rev)
