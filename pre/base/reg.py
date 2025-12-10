from __future__ import annotations

from enum import Enum, unique
from typing import Callable, Dict


@unique
class FE(Enum):
    RET1 = "ret1"
    RET5 = "ret5"
    RET10 = "ret10"
    M1 = "m1"
    M3 = "m3"
    M6 = "m6"
    M12 = "m12"
    M1_VA = "m1_va"
    M12_VA = "m12_va"
    REV5 = "rev5"
    VOL5 = "vol5"
    VOL10 = "vol10"
    VOL20 = "vol20"
    VOL60 = "vol60"
    VOL120 = "vol120"
    HLR = "hlr"
    IDV = "idv"
    TRANGE = "trange"
    VOLZ = "volz"
    VOLMA20 = "volma20"
    VOLMA_R = "volma_r"
    TURNOVER = "turnover"
    AMIHUD = "amihud"
    SPRD = "sprd"
    PIMPACT = "pimpact"
    VOLSHOCK = "volshock"
    MA5 = "ma5"
    MA20 = "ma20"
    MA60 = "ma60"
    MA120 = "ma120"
    MACD = "macd"
    MACDS = "macds"
    RSI14 = "rsi14"
    STO_K = "sto_k"
    STO_D = "sto_d"
    BOLL_UP = "boll_up"
    BOLL_LOW = "boll_low"
    BOLL_W = "boll_w"
    HIGH52 = "high52"
    LOW52 = "low52"
    PRICE_Z = "price_z"
    DIST_MA20 = "dist_ma20"
    BREAKOUT = "breakout"
    BM = "bm"
    EP = "ep"
    ROE = "roe"
    GP_A = "gp_a"
    OPM = "opm"
    SG = "sg"
    AG = "ag"
    LEV = "lev"
    TURN = "turn"
    ACC = "acc"
    OP_FQ1 = "op_fq1"
    OP_FQ2 = "op_fq2"
    OP_FY1 = "op_fy1"
    EPS_FQ1 = "eps_fq1"
    EPS_FQ2 = "eps_fq2"
    EPS_FY1 = "eps_fy1"
    OP_FQ_SPRD = "op_fq_sprd"
    EPS_FQ_SPRD = "eps_fq_sprd"
    OP_FQ_TURN = "op_fq_turn"
    EPS_FQ_TURN = "eps_fq_turn"
    REV_OP_FQ1 = "rev_op_fq1"
    REV_OP_FQ2 = "rev_op_fq2"
    REV_OP_FY1 = "rev_op_fy1"
    REV_EPS_FQ1 = "rev_eps_fq1"
    REV_EPS_FQ2 = "rev_eps_fq2"
    SECTOR_OH = "sector_oh"
    SECTOR_ID = "sector_id"


REG: Dict[FE, Callable] = {}


def register(name: FE):
    def wrap(cls):
        REG[name] = cls
        return cls

    return wrap
