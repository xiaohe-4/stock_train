"""Comparable open-to-open holding. Four trading-day steps, not five sessions."""

from __future__ import annotations

BASELINE_HOLDING_ID = "open_t_plus_1_to_open_t_plus_5"


def resolve_holding(
    trading_days: list[str],
    signal_date: str,
    holding_id: str = BASELINE_HOLDING_ID,
) -> dict:
    if holding_id != BASELINE_HOLDING_ID:
        raise ValueError(f"only {BASELINE_HOLDING_ID} is the comparable baseline")
    index = trading_days.index(signal_date)
    if index + 5 >= len(trading_days):
        raise ValueError("signal does not have an exit inside the calendar")
    return {
        "holding_id": holding_id,
        "signal_date": signal_date,
        "entry_date": trading_days[index + 1],
        "exit_date": trading_days[index + 5],
        "entry_price": "open",
        "exit_price": "open",
        "holding_steps": 4,
    }


def rebalance_signals(trading_days: list[str], first: str, last: str) -> list[str]:
    start = trading_days.index(first)
    stop = trading_days.index(last)
    return [trading_days[index] for index in range(start, stop + 1, 5)]
