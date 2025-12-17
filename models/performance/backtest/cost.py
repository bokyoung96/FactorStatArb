from __future__ import annotations

from dataclasses import dataclass


def _bps(bps: float) -> float:
    return max(0.0, float(bps)) / 10_000.0


@dataclass(frozen=True)
class Cost:
    on: bool = False
    buy: float = 0.0
    sell: float = 0.0
    tax: float = 0.0

    def __post_init__(self) -> None:
        for k in ("buy", "sell", "tax"):
            if getattr(self, k) < 0:
                raise ValueError(f"{k} must be non-negative.")

    @classmethod
    def off(cls) -> "Cost":
        return cls(on=False)

    def in_cap(self, cap: float) -> float:
        if not self.on or cap <= 0:
            return float(cap)
        return float(cap * (1.0 - _bps(self.buy)))

    def out_cap(self, gross: float, frac: float = 1.0) -> float:
        if not self.on or gross <= 0:
            return float(gross)
        frac = min(max(float(frac), 0.0), 1.0)
        out_bps = self.sell + self.tax
        return float(gross * max(0.0, 1.0 - _bps(out_bps) * frac))

    def bps(self) -> float:
        if not self.on:
            return 0.0
        return float(self.buy + self.sell + self.tax)

