"""Price-limit checks for unadjusted bars. Missing inputs fail closed."""

from __future__ import annotations

import math


def limit_price(preclose: float, ratio: float) -> float:
    return math.floor(float(preclose) * (1.0 + ratio) * 100.0 + 0.5) / 100.0


def _board(code: str) -> str | None:
    text = str(code).split(".")[-1].zfill(6)
    if text.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "main"
    if text.startswith(("300", "301")):
        return "chinext"
    if text.startswith(("688", "689")):
        return "star"
    return None


def _st(value) -> bool | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"1", "1.0"}:
        return True
    if text in {"0", "0.0"}:
        return False
    return None


def limit_status(code: str, is_st, ipo_date, trade_date: str, trading_days: list[str]) -> dict:
    st = _st(is_st)
    board = _board(code)
    if st is None or not ipo_date or board is None or trade_date not in trading_days:
        return {"state": "unverified", "reason": "rule_unverified"}
    if ipo_date == trade_date and board == "main":
        return {"state": "unverified", "reason": "rule_unverified"}
    if ipo_date in trading_days:
        age = trading_days.index(trade_date) - trading_days.index(ipo_date)
        if age < 0:
            return {"state": "unverified", "reason": "rule_unverified"}
        if board in {"chinext", "star"} and age < 5:
            return {"state": "no_limit"}
    elif ipo_date > trading_days[0]:
        return {"state": "unverified", "reason": "rule_unverified"}
    if board == "main":
        ratio = 0.05 if st else 0.10
    else:
        ratio = 0.20
    return {"state": "ratio", "up": ratio, "down": ratio}


def reject_reason(bar: dict, side: str, status: dict) -> str | None:
    if status.get("state") == "unverified":
        return "rule_unverified"
    volume = bar.get("volume")
    try:
        volume_value = float(volume)
    except (TypeError, ValueError):
        volume_value = 0.0
    if str(bar.get("tradestatus")) not in {"1", "1.0"} or volume_value <= 0 or volume_value != volume_value:
        return "halt"
    if status.get("state") == "no_limit":
        return None
    preclose = float(bar["preclose"])
    open_price = float(bar["open"])
    if side == "buy":
        cap = limit_price(preclose, float(status["up"]))
        if open_price + 1e-9 >= cap and float(bar["high"]) <= open_price + 1e-9:
            return "limit_up_sealed"
    if side == "sell":
        floor = limit_price(preclose, -float(status["down"]))
        if open_price <= floor + 1e-9 and float(bar["low"]) + 1e-9 >= open_price:
            return "limit_down_sealed"
    return None
