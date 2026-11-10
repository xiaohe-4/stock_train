"""Example A-share costs for 2024-01-01 through 2026-11-09. Not a broker quote."""

from __future__ import annotations

COMMISSION_RATE = 0.00025
MINIMUM_COMMISSION = 5.0
STAMP_DUTY_SELL = 0.0005
SLIPPAGE_RATE = 0.0005
APPLICABLE = "2024-01-01..2026-11-09; sell-side stamp duty uses the 0.05% rate in force after 2023-08-28"
STRESS_MULTIPLIERS = (1, 2, 3)


def charge(notional: float, side: str, multiplier: int = 1) -> float:
    commission = max(MINIMUM_COMMISSION, float(notional) * COMMISSION_RATE) * multiplier
    stamp = (float(notional) * STAMP_DUTY_SELL if side == "sell" else 0.0) * multiplier
    slippage = float(notional) * SLIPPAGE_RATE * multiplier
    return commission + stamp + slippage
