"""Decide whether inputs can support strict mode. Never relabel 后复权 as strict."""

from __future__ import annotations

RAW_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "tradestatus",
    "isST",
    "adjustflag",
)


def audit_frame(prices, membership, listings) -> dict:
    columns = {str(column) for column in getattr(prices, "columns", [])}
    blockers: list[str] = []
    if "股票代码" in columns or "开盘" in columns:
        blockers.append("现有面板只含当前成分的后复权行情，不能消除幸存者偏差")
        blockers.append("缺少不复权前收、交易状态和是否ST")
    missing = [name for name in RAW_FIELDS if name not in columns]
    if missing:
        blockers.append("缺少不复权字段")
    if membership is None or getattr(membership, "empty", True):
        blockers.append("缺少按日历史成分股")
    if listings is None or getattr(listings, "empty", True):
        blockers.append("缺少上市日")
    if "adjustflag" in columns:
        flags = set(prices["adjustflag"].astype(str))
        if flags - {"3"}:
            blockers.append("存在非不复权价格")
    return {
        "strict_ready": not blockers,
        "mode": "strict" if not blockers else "research",
        "blockers": blockers,
    }
