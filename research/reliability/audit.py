"""Decide whether inputs can support strict mode. Never relabel 后复权 as strict."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from research.reliability.fetch_pit import BAR_FIELDS, END, START

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_store(
    root,
    expected_days: int = 689,
    constituents_per_day: int = 300,
    start: str = START,
    end: str = END,
) -> dict:
    root = Path(root)
    blockers: list[str] = []
    calendar_path = root / "calendar.json"
    manifest_path = root / "manifest.json"
    if not calendar_path.is_file() or not manifest_path.is_file():
        blockers.append("缺少交易日历或 manifest，不能把后复权快照当成严格数据")
        return {"strict_ready": False, "mode": "research", "blockers": blockers}
    dates = json.loads(calendar_path.read_text(encoding="utf-8"))
    if len(dates) != expected_days or not dates or dates[0] != start or dates[-1] != end:
        blockers.append(f"交易日数量或区间不符: {len(dates)} {dates[:1]} {dates[-1:]}")
    codes = set()
    for date in dates:
        path = root / "membership" / f"{date}.json"
        if not path.is_file():
            blockers.append(f"缺失成分股文件 {date}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != constituents_per_day:
            blockers.append(f"{date} 成分股数量 {len(rows)} != {constituents_per_day}")
        codes.update(row["code"] for row in rows)
    required = [field for field in BAR_FIELDS if field != "turn"]
    for code in sorted(codes):
        path = root / "bars" / f"{code.replace('.', '_')}.csv"
        if not path.is_file():
            blockers.append(f"缺失不复权行情 {code}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or any(field not in reader.fieldnames for field in required):
                blockers.append(f"{code} 缺少不复权字段")
                continue
            for row in reader:
                if row.get("adjustflag") != "3":
                    blockers.append(f"{code} 存在非不复权价格")
                    break
                if row.get("date", "") > end:
                    blockers.append(f"{code} 含有 {end} 之后的行情")
                    break
    listings_path = root / "listings.json"
    listings = json.loads(listings_path.read_text(encoding="utf-8")) if listings_path.is_file() else {}
    for code in sorted(codes):
        if not listings.get(code, {}).get("ipoDate"):
            blockers.append(f"缺少上市日 {code}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, meta in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            blockers.append(f"manifest 指向缺失文件 {relative}")
            continue
        if _sha256(path) != meta.get("sha256"):
            blockers.append(f"sha256 不一致 {relative}")
    return {"strict_ready": not blockers, "mode": "strict" if not blockers else "research", "blockers": blockers}
