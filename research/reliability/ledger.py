"""Daily cash account. Flat days stay in the NAV series."""

from __future__ import annotations

from research.reliability.costs import charge
from research.reliability.limits import limit_status, reject_reason


def _lot(code: str) -> tuple[int, int]:
    text = str(code).split(".")[-1].zfill(6)
    if text.startswith(("688", "689")):
        return 200, 1
    return 100, 100


def _affordable(cash: float, price: float, code: str, multiplier: int) -> int:
    minimum, step = _lot(code)
    shares = int(cash // price)
    if shares < minimum:
        return 0
    if step > 1:
        shares = (shares // step) * step
    while shares >= minimum:
        notional = shares * price
        if notional + charge(notional, "buy", multiplier) <= cash + 1e-8:
            return shares
        shares -= step
    return 0


def run_account(
    bars,
    schedules: list[dict],
    listings: dict,
    trading_days: list[str],
    starting_cash: float = 1_000_000,
    multiplier: int = 1,
    enforce_limits: bool = True,
) -> dict:
    frame = bars.copy()
    frame["code"] = frame["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    frame["date"] = frame["date"].astype(str)
    lookup = {(row.date, row.code): row for row in frame.itertuples(index=False)}
    start = min(item["signal_date"] for item in schedules)
    end = max(item["exit_date"] for item in schedules)
    end_index = min(trading_days.index(end) + 1, len(trading_days) - 1)
    span = trading_days[trading_days.index(start) : end_index + 1]
    cash = float(starting_cash)
    shares = {
        str(code).split(".")[-1].zfill(6): 0
        for schedule in schedules
        for code in schedule["weights"]
    }
    daily = []
    fills = []
    rejects = []

    def bar_for(date: str, code: str):
        return lookup.get((date, code))

    def blocked(date: str, code: str, side: str):
        row = bar_for(date, code)
        if row is None:
            return "missing_raw_price"
        if not enforce_limits:
            volume = float(row.volume)
            if str(row.tradestatus) not in {"1", "1.0"} or volume <= 0:
                return "halt"
            return None
        status = limit_status(code, row.isST, listings.get(code), date, trading_days)
        return reject_reason(row._asdict(), side, status)

    for date in span:
        bought_today = set()
        traded = 0.0
        day_fees = 0.0
        for schedule in schedules:
            if schedule["entry_date"] == date and schedule["exit_date"] == date:
                for code in schedule["weights"]:
                    rejects.append({"date": date, "side": "sell", "code": code, "reason": "t_plus_one"})
                continue
            if schedule["exit_date"] == date:
                for code in list(shares):
                    if shares.get(code, 0) <= 0:
                        continue
                    if code in bought_today:
                        rejects.append({"date": date, "side": "sell", "code": code, "reason": "t_plus_one"})
                        continue
                    reason = blocked(date, code, "sell")
                    if reason:
                        rejects.append({"date": date, "side": "sell", "code": code, "reason": reason})
                        continue
                    price = float(bar_for(date, code).open)
                    notional = shares[code] * price
                    fee = charge(notional, "sell", multiplier)
                    cash += notional - fee
                    traded += notional
                    day_fees += fee
                    fills.append({"date": date, "side": "sell", "code": code, "shares": shares[code], "price": price, "fees": fee})
                    shares[code] = 0
            if schedule["entry_date"] == date:
                for code, weight in schedule["weights"].items():
                    reason = blocked(date, code, "buy")
                    if reason:
                        rejects.append({"date": date, "side": "buy", "code": code, "reason": reason})
                        continue
                    price = float(bar_for(date, code).open)
                    budget = cash * float(weight)
                    quantity = _affordable(budget, price, code, multiplier)
                    if quantity <= 0:
                        rejects.append({"date": date, "side": "buy", "code": code, "reason": "below_lot"})
                        continue
                    notional = quantity * price
                    fee = charge(notional, "buy", multiplier)
                    cash -= notional + fee
                    shares[code] = shares.get(code, 0) + quantity
                    bought_today.add(code)
                    traded += notional
                    day_fees += fee
                    fills.append({"date": date, "side": "buy", "code": code, "shares": quantity, "price": price, "fees": fee})
        market = 0.0
        for code, quantity in shares.items():
            row = bar_for(date, code)
            price = float(row.close) if row is not None and quantity else 0.0
            market += quantity * price
        nav = cash + market
        turnover = traded / nav if nav else 0.0
        daily.append(
            {
                "date": date,
                "nav": nav,
                "cash": cash,
                "shares": {code: int(quantity) for code, quantity in shares.items()},
                "turnover": turnover,
                "fees": day_fees,
                "limits_enforced": enforce_limits,
            }
        )
    return {"daily": daily, "fills": fills, "rejects": rejects}
