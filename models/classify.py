from __future__ import annotations

from typing import Iterable, Tuple

from pre.base.reg import FE

ALREADY_STANDARDIZED = {
    FE.VOLZ.value,
    FE.PRICE_Z.value,
}

BOUNDED_01 = {
    FE.STO_K.value,
    FE.STO_D.value,
    FE.HIGH52.value,
    FE.LOW52.value,
    FE.BREAKOUT.value,
}

BOOLEAN_FLAGS = {
    FE.OP_FQ_TURN.value,
    FE.EPS_FQ_TURN.value,
}

CATEGORICAL = {
    FE.SECTOR_OH.value,
    FE.SECTOR_ID.value,
}

EXEMPT = ALREADY_STANDARDIZED | BOUNDED_01 | BOOLEAN_FLAGS | CATEGORICAL


def _key(col) -> str:
    if isinstance(col, tuple) and col:
        return str(col[0]).lower()
    if isinstance(col, FE):
        return col.value.lower()
    return str(col).lower()


def should_normalize(col) -> bool:
    return _key(col) not in EXEMPT


def classify(columns: Iterable) -> Tuple[list, list]:
    to_norm = []
    exempt = []
    seen_norm = set()
    seen_exempt = set()
    for col in columns:
        key = _key(col)
        if key in EXEMPT:
            if key not in seen_exempt:
                exempt.append(col)
                seen_exempt.add(key)
        else:
            if key not in seen_norm:
                to_norm.append(col)
                seen_norm.add(key)
    return to_norm, exempt
