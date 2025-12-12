import numpy as np

from pre.base.feature import F, add_feature
from pre.base.reg import FE, register


def _zscore(x, window: int):
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std


def _ema(x, span: int):
    return x.ewm(span=span, adjust=False).mean()


def _safe_div(num, denom):
    out = num.divide(denom)
    return out.replace([np.inf, -np.inf], np.nan)


def _trading_value(df):
    return df["close"] * df["vol"]


def _rsi(series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@register(FE.RET1)
class RET1(F):
    def run(self, df):
        ret = df["close"].pct_change(1, fill_method=None)
        return add_feature(df, FE.RET1.value, ret)


@register(FE.RET5)
class RET5(F):
    def run(self, df):
        ret = df["close"].pct_change(5, fill_method=None)
        return add_feature(df, FE.RET5.value, ret)


@register(FE.RET10)
class RET10(F):
    def run(self, df):
        ret = df["close"].pct_change(10, fill_method=None)
        return add_feature(df, FE.RET10.value, ret)


@register(FE.M1)
class M1(F):
    def run(self, df):
        ret = df["close"].pct_change(21, fill_method=None)
        return add_feature(df, FE.M1.value, ret)


@register(FE.M3)
class M3(F):
    def run(self, df):
        ret = df["close"].pct_change(63, fill_method=None)
        return add_feature(df, FE.M3.value, ret)


@register(FE.M6)
class M6(F):
    def run(self, df):
        ret = df["close"].pct_change(126, fill_method=None)
        return add_feature(df, FE.M6.value, ret)


@register(FE.M12)
class M12(F):
    def run(self, df):
        ret = df["close"].pct_change(252, fill_method=None)
        return add_feature(df, FE.M12.value, ret)


@register(FE.M1_VA)
class M1_VA(F):
    def run(self, df):
        m1 = df["close"].pct_change(21, fill_method=None)
        vol20 = df["close"].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.M1_VA.value, _safe_div(m1, vol20))


@register(FE.M12_VA)
class M12_VA(F):
    def run(self, df):
        m12 = df["close"].pct_change(252, fill_method=None)
        vol20 = df["close"].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.M12_VA.value, _safe_div(m12, vol20))


@register(FE.REV5)
class REV5(F):
    def run(self, df):
        ret = -df["close"].pct_change(5, fill_method=None)
        return add_feature(df, FE.REV5.value, ret)


@register(FE.VOL5)
class VOL5(F):
    def run(self, df):
        vol = df["close"].pct_change(fill_method=None).rolling(5).std()
        return add_feature(df, FE.VOL5.value, vol)


@register(FE.VOL10)
class VOL10(F):
    def run(self, df):
        vol = df["close"].pct_change(fill_method=None).rolling(10).std()
        return add_feature(df, FE.VOL10.value, vol)


@register(FE.VOL20)
class VOL20(F):
    def run(self, df):
        vol = df["close"].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.VOL20.value, vol)


@register(FE.VOL60)
class VOL60(F):
    def run(self, df):
        vol = df["close"].pct_change(fill_method=None).rolling(60).std()
        return add_feature(df, FE.VOL60.value, vol)


@register(FE.VOL120)
class VOL120(F):
    def run(self, df):
        vol = df["close"].pct_change(fill_method=None).rolling(120).std()
        return add_feature(df, FE.VOL120.value, vol)


@register(FE.HLR)
class HLR(F):
    def run(self, df):
        hlr = (df["high"] - df["low"]) / df["close"]
        return add_feature(df, FE.HLR.value, hlr)


@register(FE.IDV)
class IDV(F):
    def run(self, df):
        idv = (df["high"] - df["low"]) / df["open"]
        return add_feature(df, FE.IDV.value, idv)


@register(FE.TRANGE)
class TRANGE(F):
    def run(self, df):
        prev_close = df["close"].shift(1)
        range1 = df["high"] - df["low"]
        range2 = (df["high"] - prev_close).abs()
        range3 = (df["low"] - prev_close).abs()
        tr = np.maximum(range1, np.maximum(range2, range3))
        return add_feature(df, FE.TRANGE.value, tr)


@register(FE.VOLZ)
class VOLZ(F):
    def run(self, df):
        volz = _zscore(df["vol"], 20)
        return add_feature(df, FE.VOLZ.value, volz)


@register(FE.VOLMA20)
class VOLMA20(F):
    def run(self, df):
        volma = df["vol"].rolling(20).mean()
        return add_feature(df, FE.VOLMA20.value, volma)


@register(FE.VOLMA_R)
class VOLMA_R(F):
    def run(self, df):
        volma20 = df["vol"].rolling(20).mean()
        volr = _safe_div(df["vol"], volma20)
        return add_feature(df, FE.VOLMA_R.value, volr)


@register(FE.TURNOVER)
class TURNOVER(F):
    def run(self, df):
        shares_out = _safe_div(df["mcap"], df["close"])
        turnover = _safe_div(df["vol"], shares_out)
        return add_feature(df, FE.TURNOVER.value, turnover)


@register(FE.AMIHUD)
class AMIHUD(F):
    def run(self, df):
        ret = df["close"].pct_change(fill_method=None)
        trading_value = _trading_value(df)
        amihud = _safe_div(ret.abs(), trading_value)
        return add_feature(df, FE.AMIHUD.value, amihud)


@register(FE.SPRD)
class SPRD(F):
    def run(self, df):
        sprd = (df["high"] - df["low"]) / df["close"]
        return add_feature(df, FE.SPRD.value, sprd)


@register(FE.PIMPACT)
class PIMPACT(F):
    def run(self, df):
        ret = df["close"].pct_change(fill_method=None)
        trading_value = _trading_value(df)
        pimpact = _safe_div(ret.abs(), trading_value)
        return add_feature(df, FE.PIMPACT.value, pimpact)


@register(FE.VOLSHOCK)
class VOLSHOCK(F):
    def run(self, df):
        volma20 = df["vol"].rolling(20).mean()
        volstd20 = df["vol"].rolling(20).std()
        volshock = _safe_div(df["vol"] - volma20, volstd20)
        return add_feature(df, FE.VOLSHOCK.value, volshock)


@register(FE.MA5)
class MA5(F):
    def run(self, df):
        ma = df["close"].rolling(5).mean()
        return add_feature(df, FE.MA5.value, ma)


@register(FE.MA20)
class MA20(F):
    def run(self, df):
        ma = df["close"].rolling(20).mean()
        return add_feature(df, FE.MA20.value, ma)


@register(FE.MA60)
class MA60(F):
    def run(self, df):
        ma = df["close"].rolling(60).mean()
        return add_feature(df, FE.MA60.value, ma)


@register(FE.MA120)
class MA120(F):
    def run(self, df):
        ma = df["close"].rolling(120).mean()
        return add_feature(df, FE.MA120.value, ma)


@register(FE.MACD)
class MACD(F):
    def run(self, df):
        ema12 = _ema(df["close"], 12)
        ema26 = _ema(df["close"], 26)
        macd = ema12 - ema26
        return add_feature(df, FE.MACD.value, macd)


@register(FE.MACDS)
class MACDS(F):
    def run(self, df):
        ema12 = _ema(df["close"], 12)
        ema26 = _ema(df["close"], 26)
        macd = ema12 - ema26
        macds = _ema(macd, 9)
        return add_feature(df, FE.MACDS.value, macds)


@register(FE.RSI14)
class RSI14(F):
    def run(self, df):
        rsi = _rsi(df["close"], 14)
        return add_feature(df, FE.RSI14.value, rsi)


@register(FE.STO_K)
class STO_K(F):
    def run(self, df):
        low_k = df["low"].rolling(14).min()
        high_k = df["high"].rolling(14).max()
        sto_k = _safe_div(df["close"] - low_k, (high_k - low_k))
        return add_feature(df, FE.STO_K.value, sto_k)


@register(FE.STO_D)
class STO_D(F):
    def run(self, df):
        low_k = df["low"].rolling(14).min()
        high_k = df["high"].rolling(14).max()
        sto_k = (df["close"] - low_k) / (high_k - low_k)
        sto_d = sto_k.rolling(3).mean()
        return add_feature(df, FE.STO_D.value, sto_d)


@register(FE.BOLL_UP)
class BOLL_UP(F):
    def run(self, df):
        mean = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        boll_up = mean + 2 * std
        return add_feature(df, FE.BOLL_UP.value, boll_up)


@register(FE.BOLL_LOW)
class BOLL_LOW(F):
    def run(self, df):
        mean = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        boll_low = mean - 2 * std
        return add_feature(df, FE.BOLL_LOW.value, boll_low)


@register(FE.BOLL_W)
class BOLL_W(F):
    def run(self, df):
        mean = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        bw = _safe_div((upper - lower), mean)
        return add_feature(df, FE.BOLL_W.value, bw)


@register(FE.HIGH52)
class HIGH52(F):
    def run(self, df):
        high52 = _safe_div(df["close"], df["close"].rolling(252).max())
        return add_feature(df, FE.HIGH52.value, high52)


@register(FE.LOW52)
class LOW52(F):
    def run(self, df):
        low52 = _safe_div(df["close"], df["close"].rolling(252).min())
        return add_feature(df, FE.LOW52.value, low52)


@register(FE.PRICE_Z)
class PRICE_Z(F):
    def run(self, df):
        pz = _zscore(df["close"], 252)
        return add_feature(df, FE.PRICE_Z.value, pz)


@register(FE.DIST_MA20)
class DIST_MA20(F):
    def run(self, df):
        ma20 = df["close"].rolling(20).mean()
        dist = _safe_div(df["close"], ma20) - 1
        return add_feature(df, FE.DIST_MA20.value, dist)


@register(FE.BREAKOUT)
class BREAKOUT(F):
    def run(self, df):
        rolling_max = df["close"].rolling(20).max().shift(1)
        br = (df["close"] > rolling_max).astype(int)
        return add_feature(df, FE.BREAKOUT.value, br)
