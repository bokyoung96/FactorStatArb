from __future__ import annotations

from pre.base.feature import F, add_feature
from pre.base.reg import FE, register


@register(FE.OP_FQ1)
class OP_FQ1(F):
    def run(self, df):
        return add_feature(df, FE.OP_FQ1.value, df["op_fq1_raw"])


@register(FE.OP_FQ2)
class OP_FQ2(F):
    def run(self, df):
        return add_feature(df, FE.OP_FQ2.value, df["op_fq2_raw"])


@register(FE.OP_FY1)
class OP_FY1(F):
    def run(self, df):
        return add_feature(df, FE.OP_FY1.value, df["op_fy1_raw"])


@register(FE.EPS_FQ1)
class EPS_FQ1(F):
    def run(self, df):
        return add_feature(df, FE.EPS_FQ1.value, df["eps_fq1_raw"])


@register(FE.EPS_FQ2)
class EPS_FQ2(F):
    def run(self, df):
        return add_feature(df, FE.EPS_FQ2.value, df["eps_fq2_raw"])


@register(FE.EPS_FY1)
class EPS_FY1(F):
    def run(self, df):
        return add_feature(df, FE.EPS_FY1.value, df["eps_fy1_raw"])


@register(FE.REV_OP_FQ1)
class REV_OP_FQ1(F):
    def run(self, df):
        rev = df["op_fq1_raw"].pct_change(fill_method=None)
        return add_feature(df, FE.REV_OP_FQ1.value, rev)


@register(FE.REV_OP_FQ2)
class REV_OP_FQ2(F):
    def run(self, df):
        rev = df["op_fq2_raw"].pct_change(fill_method=None)
        return add_feature(df, FE.REV_OP_FQ2.value, rev)


@register(FE.REV_OP_FY1)
class REV_OP_FY1(F):
    def run(self, df):
        rev = df["op_fy1_raw"].pct_change(fill_method=None)
        return add_feature(df, FE.REV_OP_FY1.value, rev)


@register(FE.REV_EPS_FQ1)
class REV_EPS_FQ1(F):
    def run(self, df):
        rev = df["eps_fq1_raw"].pct_change(fill_method=None)
        return add_feature(df, FE.REV_EPS_FQ1.value, rev)


@register(FE.REV_EPS_FQ2)
class REV_EPS_FQ2(F):
    def run(self, df):
        rev = df["eps_fq2_raw"].pct_change(fill_method=None)
        return add_feature(df, FE.REV_EPS_FQ2.value, rev)
