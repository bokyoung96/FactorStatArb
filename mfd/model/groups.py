from __future__ import annotations


PRICE_FEATURES = [
    "ret1",
    "ret5",
    "ret10",
    "price_z",
    "dist_ma20",
    "breakout",
    "ret1_cs_rank",
]

MOMENTUM_FEATURES = [
    "m1",
    "m3",
    "m6",
    "m12",
    "ma5",
    "ma20",
    "ma60",
    "ma120",
    "macd",
    "macds",
    "m1_cs_z",
]

VOLATILITY_FEATURES = [
    "vol5",
    "vol10",
    "vol20",
    "vol60",
    "vol120",
    "volz",
    "volma20",
    "volma_r",
    "trange",
    "idv",
    "volshock",
    "vol20_cs_z",
]

LIQUIDITY_FEATURES = [
    "turnover",
    "amihud",
    "sprd",
    "pimpact",
]

TECHNICAL_FEATURES = [
    "rsi14",
    "sto_k",
    "sto_d",
    "boll_up",
    "boll_low",
    "boll_w",
    "high52",
    "low52",
    "hlr",
]

FEATURE_ORDER = (
    PRICE_FEATURES
    + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES
    + LIQUIDITY_FEATURES
    + TECHNICAL_FEATURES
)

GROUP_SIZES = {
    "price": len(PRICE_FEATURES),
    "momentum": len(MOMENTUM_FEATURES),
    "volatility": len(VOLATILITY_FEATURES),
    "liquidity": len(LIQUIDITY_FEATURES),
    "technical": len(TECHNICAL_FEATURES),
}


def group_dims_for(d_model: int) -> tuple[int, int, int, int, int]:
    """
    NOTE:
    Fixed allocation of embedding dimensions across feature groups.
    This is a design choice (not a hyperparameter) to stabilize attention
    and enable interpretability across rolling windows.
    """
    if int(d_model) == 64:
        # PRICE 20%, MOM 25%, VOL 25%, LIQ 15%, TECH 15%
        return (12, 16, 16, 10, 10)
    if int(d_model) == 128:
        # PRICE 20%, MOM 25%, VOL 25%, LIQ 15%, TECH 15%
        return (26, 32, 32, 19, 19)
    raise ValueError(f"Unsupported d_model={d_model}; add an explicit mapping in group_dims_for().")
