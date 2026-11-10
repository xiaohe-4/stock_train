"""Selection score and the pre-registered non-overlapping NAV.

Overlapping 5-day signal returns are averaged for model selection.
They are not compounded and they are not a daily Sharpe.
The tradable path rebalances every 5 scored dates and steps the NAV once per holding period.
"""

from __future__ import annotations

import math

from research.protocol import EXAMPLE_COST_RATE, HOLDING_TRADING_DAYS


def select_nonoverlapping(dates: list[str]) -> list[str]:
    chosen = []
    next_index = 0
    for index, date in enumerate(dates):
        if index >= next_index:
            chosen.append(date)
            next_index = index + HOLDING_TRADING_DAYS
    return chosen


def _net_return(record: dict) -> float:
    return float(record["port_return"]) - EXAMPLE_COST_RATE * float(record["turnover"])


def mean_excess(records: list[dict], split: str) -> float:
    chosen = [row for row in records if row["split"] == split]
    if not chosen:
        return 0.0
    excesses = [
        float(row["port_return"]) - float(row["ew_return"]) - EXAMPLE_COST_RATE * float(row["turnover"])
        for row in chosen
    ]
    return float(sum(excesses) / len(excesses))


def period_nav(records: list[dict], split: str) -> dict:
    ordered = [row for row in records if row["split"] == split]
    ordered.sort(key=lambda row: row["date"])
    chosen_dates = set(select_nonoverlapping([row["date"] for row in ordered]))
    trades = [row for row in ordered if row["date"] in chosen_dates]
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    period_returns = []
    fees = 0.0
    log = []
    for row in trades:
        fee = EXAMPLE_COST_RATE * float(row["turnover"])
        net = _net_return(row)
        nav *= 1.0 + net
        peak = max(peak, nav)
        if peak > 0:
            max_drawdown = min(max_drawdown, nav / peak - 1.0)
        period_returns.append(net)
        fees += fee
        log.append(
            {
                "date": row["date"],
                "port_return": float(row["port_return"]),
                "ew_return": float(row["ew_return"]),
                "turnover": float(row["turnover"]),
                "fee": fee,
                "net_return": net,
                "nav": nav,
            }
        )
    periods = len(period_returns)
    warning = "样本持有期少于 30 段，年化收益和夏普不稳定，不能当成稳定绩效。"
    sharpe = None
    annualized = None
    if periods >= 2:
        mean = sum(period_returns) / periods
        variance = sum((value - mean) ** 2 for value in period_returns) / (periods - 1)
        if variance > 0:
            sharpe = mean / math.sqrt(variance) * math.sqrt(252 / HOLDING_TRADING_DAYS)
        annualized = (nav ** (252 / HOLDING_TRADING_DAYS / periods)) - 1.0
    return {
        "periods": periods,
        "nav": nav,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sharpe_observations": periods,
        "annualized": annualized,
        "annualization_unstable": periods < 30,
        "annualization_warning": warning,
        "fees": fees,
        "trades": log,
    }


def apply_runtime_budget(status: str, median_epoch_seconds: dict[str, float]) -> str:
    if median_epoch_seconds["memory"] > 2.0 * median_epoch_seconds["transformer"]:
        return "do_not_adopt"
    return status


def decide(dev_means: dict[str, float]) -> str:
    if dev_means["memory"] > dev_means["transformer"] and dev_means["memory"] > dev_means["lstm"]:
        return "research_candidate"
    return "do_not_adopt"


def entry_blocked(volume) -> bool:
    if volume is None:
        return True
    try:
        value = float(volume)
    except (TypeError, ValueError):
        return True
    if value != value or value <= 0:
        return True
    return False


def limit_ratio(code: str) -> float:
    text = str(code).zfill(6)
    if text.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def buy_blocked_unadjusted(open_price: float, high: float, previous_close: float, code: str, volume) -> bool:
    """Seal test for unadjusted prices. Do not call this on the 后复权 panel."""
    if entry_blocked(volume):
        return True
    if previous_close <= 0:
        return False
    gap = open_price / previous_close - 1.0
    sealed = abs(open_price - high) <= 1e-8
    return gap >= limit_ratio(code) - 0.002 and sealed


def limits_applied_to_hfq_panel() -> bool:
    return False


def day_portfolio(codes, scores, raw, tradable, previous: dict[str, float]) -> dict:
    """Apply the fixed top-5 weights after dropping names that cannot be traded."""
    from research.protocol import WEIGHTS

    order = sorted(range(len(codes)), key=lambda index: float(scores[index]), reverse=True)
    chosen = order[: len(WEIGHTS)]
    raw_weights = {codes[index]: 0.0 for index in range(len(codes))}
    for rank, index in enumerate(chosen):
        if tradable[index]:
            raw_weights[codes[index]] = WEIGHTS[rank]
    tradable_sum = sum(raw_weights.values())
    weights = {code: (value / tradable_sum if tradable_sum > 0 else 0.0) for code, value in raw_weights.items()}
    port = sum(weights[codes[index]] * float(raw[index]) for index in range(len(codes)))
    names = set(previous) | set(weights)
    turnover = 0.5 * sum(abs(weights.get(name, 0.0) - previous.get(name, 0.0)) for name in names)
    return {"weights": weights, "port_return": port, "turnover": turnover}


def top_k_jaccard(left: list[str], right: list[str], k: int = 5) -> float:
    a = set(left[:k])
    b = set(right[:k])
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
