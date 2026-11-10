#!/usr/bin/env python3
"""Refresh CSI 300 daily bars with the repository's BaoStock method.

The executable project window starts at 2024-01-01 (see THU-BDC2026-main/get_stock_data.py).
Bars use adjustflag="1" (后复权) and the same column mapping as get_stock_history.

An unfinished A-share session is not a complete daily bar. Before 15:00
Asia/Shanghai on a trading day, that calendar day is excluded even if a data
vendor would emit a partial row. The saved panel is then checked against the
latest date that was actually returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import traceback
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
CLOSE_TIME = dtime(15, 0)
DEFAULT_START = "2024-01-01"
REQUIRED_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fetcher(repo_root: Path):
    thu = repo_root / "THU-BDC2026-main"
    module_path = thu / "get_stock_data.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing repository fetcher: {module_path}")
    sys.path.insert(0, str(thu))
    import get_stock_data as fetcher

    return fetcher


def trading_days_between(start_date: str, end_date: str) -> list[str]:
    import baostock as bs

    result = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if result.error_code != "0":
        raise RuntimeError(f"query_trade_dates failed: {result.error_code} {result.error_msg}")
    days = []
    while result.error_code == "0" and result.next():
        calendar_date, is_trading = result.get_row_data()
        if str(is_trading) == "1":
            days.append(calendar_date)
    if not days:
        raise RuntimeError("BaoStock returned no trading days")
    return days


def calendar_cap(now: datetime, trading_days: list[str]) -> str:
    """Latest session that can be complete at `now`.

    `now` must be timezone-aware. A trading day is eligible only at or after
    the 15:00 Asia/Shanghai cash close. The function does not invent a bar
    when the vendor has not published one; callers still have to read the
    returned data.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(SHANGHAI)
    today = local.date().isoformat()
    elapsed = [day for day in trading_days if day < today]
    if today in trading_days and local.time() >= CLOSE_TIME:
        return today
    if not elapsed:
        raise RuntimeError("no completed trading day before the cutoff")
    return elapsed[-1]


def fetch_universe(fetcher, start_date: str, end_date: str, relogin_every: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    fetcher.login()
    constituents = fetcher.get_hs300_stocks()
    frames = []
    failures = []
    total = len(constituents)
    for index, row in constituents.iterrows():
        bs_code = row.get("code", "")
        stock_name = row.get("code_name", "")
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"[{index + 1}/{total}] {bs_code} {stock_name} {start_date}..{end_date}", flush=True)
                frame = fetcher.get_stock_history(bs_code, start_date, end_date)
                if frame is None or frame.empty:
                    raise RuntimeError("empty history")
                frame = fetcher.normalize_stock_codes(frame)
                frames.append(frame)
                print(f"  rows={len(frame)} max={frame['日期'].max()}", flush=True)
                break
            except Exception as exc:
                print(f"  attempt {attempt} failed: {exc}", flush=True)
                if attempt >= 3:
                    failures.append(
                        {
                            "code": str(bs_code),
                            "name": str(stock_name),
                            "error": str(exc),
                        }
                    )
                    break
                fetcher.ensure_login(force=True)
                time.sleep(1)
        if frames and len(frames) % relogin_every == 0:
            fetcher.ensure_login(force=True)
    if not frames:
        raise RuntimeError("no stock histories were downloaded")
    panel = pd.concat(frames, ignore_index=True)
    panel = fetcher.normalize_stock_codes(panel)
    panel = panel.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    return constituents, panel, failures


def audit_panel(panel: pd.DataFrame, constituents: pd.DataFrame, end_date: str) -> dict:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in panel.columns]
    if missing_columns:
        raise RuntimeError(f"downloaded panel missing columns: {missing_columns}")
    codes = panel["股票代码"].astype(str)
    dates = pd.to_datetime(panel["日期"], errors="coerce")
    if dates.isna().any():
        raise RuntimeError(f"unparseable dates: {int(dates.isna().sum())}")
    duplicate_rows = int(panel.duplicated(["股票代码", "日期"]).sum())
    later = sorted({day for day in dates.dt.strftime("%Y-%m-%d") if day > end_date})
    max_date = dates.max().strftime("%Y-%m-%d")
    min_date = dates.min().strftime("%Y-%m-%d")
    on_max = panel.loc[dates == dates.max(), "股票代码"].astype(str)
    constituent_codes = (
        constituents["code"].astype(str).str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)
    )
    missing_on_max = sorted(set(constituent_codes) - set(on_max))
    bad_codes = sorted({code for code in codes.unique() if len(code) != 6 or not code.isdigit()})
    null_counts = {column: int(panel[column].isna().sum()) for column in REQUIRED_COLUMNS}
    on_max_mask = dates == dates.max()
    null_volume_on_max = int((on_max_mask & panel["成交量"].isna()).sum())
    per_code_max = panel.assign(_date=dates).groupby("股票代码")["_date"].max()
    return {
        "rows": int(len(panel)),
        "stocks": int(codes.nunique()),
        "constituents": int(constituent_codes.nunique()),
        "min_date": min_date,
        "max_date": max_date,
        "requested_end_date": end_date,
        "rows_on_max_date": int(len(on_max)),
        "stocks_missing_max_date": missing_on_max,
        "dates_after_requested_end": later,
        "duplicate_code_date_rows": duplicate_rows,
        "bad_stock_codes": bad_codes,
        "null_counts": null_counts,
        "null_volume_rows_on_max_date": null_volume_on_max,
        "trading_days": int(dates.dt.strftime("%Y-%m-%d").nunique()),
        "earliest_stock_max": per_code_max.min().strftime("%Y-%m-%d"),
    }


def completeness_errors(audit: dict, failures: list[dict]) -> list[str]:
    errors = []
    if failures:
        errors.append(f"{len(failures)} constituents failed to download")
    if audit["constituents"] != 300:
        errors.append(f"constituent count is {audit['constituents']}, expected 300")
    if audit["stocks"] != audit["constituents"]:
        errors.append("downloaded stock count does not match the constituent list")
    if audit["dates_after_requested_end"]:
        errors.append(f"panel contains dates after the cutoff: {audit['dates_after_requested_end']}")
    if audit["max_date"] != audit["requested_end_date"]:
        errors.append(
            f"actual max date {audit['max_date']} is earlier than the requested complete session {audit['requested_end_date']}"
        )
    if audit["duplicate_code_date_rows"]:
        errors.append("duplicate stock/date rows")
    if audit["bad_stock_codes"]:
        errors.append(f"stock codes are not 6 digits: {audit['bad_stock_codes'][:10]}")
    # A full CSI 300 close should be present for almost every name. A small
    # miss can be a real suspension; a large miss means the snapshot is not complete.
    if audit["rows_on_max_date"] < 290:
        errors.append(
            f"only {audit['rows_on_max_date']} stocks have a bar on {audit['max_date']}; refusing to call the panel complete"
        )
    for column in ("开盘", "收盘", "最高", "最低"):
        if audit["null_counts"].get(column, 0):
            errors.append(f"null prices in {column}")
    # Halted sessions often have a price and an empty volume. They are not an
    # unfinished market day. A missing volume on the requested final session is.
    if audit.get("null_volume_rows_on_max_date", 0):
        errors.append("the final session has rows with empty volume")
    return errors


def write_outputs(output_dir: Path, constituents: pd.DataFrame, panel: pd.DataFrame, audit: dict, failures: list[dict], meta: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "hs300_daily_hfq_20240101_complete.csv"
    list_path = output_dir / "hs300_constituents.csv"
    manifest_path = output_dir / "hs300_data_manifest.json"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    constituents.to_csv(list_path, index=False, encoding="utf-8-sig")
    manifest = {
        **meta,
        "audit": audit,
        "failures": failures,
        "files": {
            "panel": {
                "path": str(panel_path),
                "sha256": sha256_file(panel_path),
                "bytes": panel_path.stat().st_size,
            },
            "constituents": {
                "path": str(list_path),
                "sha256": sha256_file(list_path),
                "bytes": list_path.stat().st_size,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the latest complete CSI 300 daily panel via BaoStock")
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/cursor/artifacts"))
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--now", default="", help="Override Asia/Shanghai now, ISO format, for tests only")
    args = parser.parse_args()

    socket.setdefaulttimeout(90)
    fetcher = load_fetcher(args.repo_root)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(SHANGHAI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI)

    fetcher.login()
    try:
        trading_days = trading_days_between("2024-01-01", "2027-06-30")
        end_date = calendar_cap(now, trading_days)
        print(f"Asia/Shanghai now: {now.isoformat()}", flush=True)
        print(f"Complete-session cap: {end_date}", flush=True)
        constituents, panel, failures = fetch_universe(fetcher, args.start_date, end_date)
        audit = audit_panel(panel, constituents, end_date)
        errors = completeness_errors(audit, failures)
        meta = {
            "source": "baostock",
            "endpoint": "public-api.baostock.com:10030",
            "adjustflag": "1",
            "adjustflag_meaning": "后复权",
            "fields": "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            "repository_fetcher": "THU-BDC2026-main/get_stock_data.py:get_stock_history",
            "start_date": args.start_date,
            "asia_shanghai_now": now.isoformat(),
            "complete_session_cap": end_date,
            "complete": not errors,
            "completeness_errors": errors,
            "pinned_commit_expected": "c735797c86ec3908379f7f54b70953c6a62b23b9",
            "trading_days": trading_days,
        }
        manifest = write_outputs(args.output_dir, constituents, panel, audit, failures, meta)
    finally:
        fetcher.logout()

    print(json.dumps({"complete": manifest["complete"], "audit": manifest["audit"], "errors": manifest["completeness_errors"]}, ensure_ascii=False, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
