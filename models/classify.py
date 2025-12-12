from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

from pre.base.reg import FE


@dataclass(frozen=True)
class NormalizationRules:
    exempt_price: frozenset = field(
        default_factory=lambda: frozenset(
            {
                FE.PRICE_Z.value,
                FE.VOLZ.value,
                FE.STO_K.value,
                FE.STO_D.value,
                FE.HIGH52.value,
                FE.LOW52.value,
                FE.BREAKOUT.value,
                FE.RSI14.value,
            }
        )
    )
    exempt_fund: frozenset = field(default_factory=frozenset)
    exempt_consensus: frozenset = field(default_factory=frozenset)
    exempt_sector: frozenset = field(default_factory=frozenset)
    boolean_flags: frozenset = field(
        default_factory=lambda: frozenset(
            {
                FE.OP_FQ_TURN.value,
                FE.EPS_FQ_TURN.value,
            }
        )
    )
    categorical: frozenset = field(
        default_factory=lambda: frozenset(
            {
                FE.SECTOR_OH.value,
                FE.SECTOR_ID.value,
            }
        )
    )
    log1p_price: frozenset = field(
        default_factory=lambda: frozenset(
            {
                FE.TURNOVER.value,
                FE.AMIHUD.value,
                FE.PIMPACT.value,
                FE.VOLMA20.value,
                FE.VOLMA_R.value,
                FE.VOLSHOCK.value,
                FE.M1_VA.value,
                FE.M12_VA.value,
            }
        )
    )

    @property
    def exempt(self) -> frozenset:
        return (
            self.exempt_price
            | self.exempt_fund
            | self.exempt_consensus
            | self.exempt_sector
            | self.boolean_flags
            | self.categorical
        )

    @property
    def log1p(self) -> frozenset:
        return self.log1p_price

    def classify(self, columns: Iterable, include_log1p: bool = False) -> Tuple[list, list]:
        to_norm, exempt, to_log = [], [], []
        seen = set()
        for col in columns:
            key = _key(col)
            if key in seen:
                continue
            seen.add(key)
            if key in self.exempt:
                exempt.append(col)
            else:
                to_norm.append(col)
            if include_log1p and key in self.log1p:
                to_log.append(col)
        if include_log1p:
            return to_norm, exempt, to_log
        return to_norm, exempt

    def get_normalize(self, columns: Iterable) -> list:
        to_norm, _ = self.classify(columns, include_log1p=False)
        return to_norm

    def get_exempt(self, columns: Iterable) -> list:
        _, exempt = self.classify(columns, include_log1p=False)
        return exempt

    def get_log1p(self, columns: Iterable) -> list:
        _, _, to_log = self.classify(columns, include_log1p=True)
        return to_log


DEFAULT_RULES = NormalizationRules()


def _key(col) -> str:
    if isinstance(col, tuple) and col:
        return str(col[0]).lower()
    if isinstance(col, FE):
        return col.value.lower()
    return str(col).lower()


DEFAULT_RULES = NormalizationRules()


def get_normalize(columns: Iterable) -> list:
    return DEFAULT_RULES.get_normalize(columns)


def get_exempt(columns: Iterable) -> list:
    return DEFAULT_RULES.get_exempt(columns)


def get_log1p(columns: Iterable) -> list:
    return DEFAULT_RULES.get_log1p(columns)


def classify(columns: Iterable, include_log1p: bool = False) -> Tuple[list, list]:
    return DEFAULT_RULES.classify(columns, include_log1p=include_log1p)
