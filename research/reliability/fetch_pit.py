"""Resumable anonymous BaoStock pull for the strict universe."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

START = "2024-01-02"
END = "2026-11-09"
BAR_FIELDS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "tradestatus",
    "isST",
    "turn",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _rows_in(path: Path) -> int:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "codes" in payload:
            return len(payload["codes"])
        if isinstance(payload, list):
            return len(payload)
        return len(payload)
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _call(action, retries: int, sleep, reconnect):
    delay = 1.0
    last = None
    for _attempt in range(retries):
        try:
            value = action()
            sleep(0)
            return value
        except Exception as exc:
            last = exc
            sleep(delay)
            delay = min(delay * 2, 30)
            reconnect()
    raise RuntimeError(f"BaoStock call failed after {retries} attempts: {last}") from last


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_pit(
    output_dir,
    client,
    start: str = START,
    end: str = END,
    constituents_per_day: int = 300,
    sleep=time.sleep,
    retries: int = 5,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started_path = root / "started_at.txt"
    if not started_path.exists():
        started_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    calendar_path = root / "calendar.json"
    if calendar_path.exists():
        dates = json.loads(calendar_path.read_text(encoding="utf-8"))
    else:
        dates = _call(lambda: client.trading_days(start, end), retries, sleep, client.reconnect)
        dates = [day for day in dates if start <= day <= end]
        _write_json(calendar_path, dates)
    membership_dir = root / "membership"
    for date in dates:
        path = membership_dir / f"{date}.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            if len(rows) == constituents_per_day:
                continue
        rows = _call(lambda date=date: client.constituents(date), retries, sleep, client.reconnect)
        if len(rows) != constituents_per_day:
            raise RuntimeError(f"constituent count {len(rows)} != {constituents_per_day} on {date}")
        _write_json(path, rows)
    codes = []
    seen = set()
    for date in dates:
        for row in json.loads((membership_dir / f"{date}.json").read_text(encoding="utf-8")):
            if row["code"] not in seen:
                seen.add(row["code"])
                codes.append(row["code"])
    bars_dir = root / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    for code in codes:
        path = bars_dir / f"{code.replace('.', '_')}.csv"
        if path.exists() and path.stat().st_size > 0:
            continue
        rows = _call(lambda code=code: client.bars(code, start, end), retries, sleep, client.reconnect)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BAR_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in BAR_FIELDS})
        temporary.replace(path)
    listings_path = root / "listings.json"
    listings = json.loads(listings_path.read_text(encoding="utf-8")) if listings_path.exists() else {}
    for code in codes:
        if code in listings and listings[code].get("ipoDate"):
            continue
        listings[code] = _call(lambda code=code: client.listing(code), retries, sleep, client.reconnect)
        _write_json(listings_path, listings)
    manifest = _manifest(root, start, end, dates, started_path.read_text(encoding="utf-8").strip())
    manifest["data_source"] = "downloaded"
    _write_json(root / "manifest.json", manifest)
    return manifest


def _cache_status(root: Path, start: str, end: str) -> str:
    manifest_path = root / "manifest.json"
    calendar_path = root / "calendar.json"
    if not manifest_path.is_file() or not calendar_path.is_file():
        return "missing manifest or calendar"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = json.loads(calendar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable manifest or calendar"
    if not dates or dates[0] > start or dates[-1] < end:
        return f"date range {dates[:1]}..{dates[-1:]} does not cover {start}..{end}"
    files = manifest.get("files") or {}
    if not files:
        return "manifest has no files"
    for relative, meta in files.items():
        path = root / relative
        if not path.is_file():
            return f"missing {relative}"
        if _sha256(path) != meta.get("sha256"):
            return f"sha256 mismatch {relative}"
    return ""


def locate_reusable(search_roots, start: str = START, end: str = END) -> tuple[Path | None, str]:
    """Return a verified PIT directory whose calendar covers start..end, else a rejection reason."""
    candidates = []
    seen = set()
    for base in search_roots:
        base = Path(base)
        if not base.exists():
            continue
        for manifest_path in base.rglob("manifest.json"):
            parent = manifest_path.parent
            if parent in seen:
                continue
            seen.add(parent)
            if (parent / "calendar.json").is_file() and (parent / "membership").is_dir():
                candidates.append(parent)
    if not candidates:
        return None, "no manifest found"

    def rank(path: Path) -> tuple:
        text = path.as_posix()
        return (0 if "hs300_reliability" in text else 1, len(text), text)

    reasons = []
    for parent in sorted(candidates, key=rank):
        reason = _cache_status(parent, start, end)
        if not reason:
            return parent, ""
        reasons.append(f"{parent}: {reason}")
    return None, "; ".join(reasons[:4])


def resolve_pit(output_dir, client=None, search_roots=(), **fetch_kwargs) -> dict:
    start = fetch_kwargs.get("start", START)
    end = fetch_kwargs.get("end", END)
    found, rejection = locate_reusable(search_roots, start, end)
    if found is not None:
        dates = json.loads((found / "calendar.json").read_text(encoding="utf-8"))
        return {
            "data_dir": str(found),
            "data_source": "reused",
            "reuse_rejection": None,
            "trading_days": len(dates),
        }
    if client is None:
        client = BaoStockClient()
    manifest = fetch_pit(output_dir, client, **fetch_kwargs)
    return {
        "data_dir": str(Path(output_dir).resolve()),
        "data_source": "downloaded",
        "reuse_rejection": rejection,
        "trading_days": manifest["trading_days"],
        "bar_rows": manifest["bar_rows"],
        "listing_rows": manifest["listing_rows"],
    }


def _manifest(root: Path, start: str, end: str, dates: list[str], started_at: str) -> dict:
    files = {}
    membership_rows = 0
    bar_rows = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "started_at.txt"} or path.suffix == ".tmp":
            continue
        relative = path.relative_to(root).as_posix()
        rows = _rows_in(path)
        files[relative] = {"sha256": _sha256(path), "rows": rows}
        if relative.startswith("membership/"):
            membership_rows += rows
        if relative.startswith("bars/"):
            bar_rows += rows
    listings = json.loads((root / "listings.json").read_text(encoding="utf-8"))
    return {
        "source": "baostock",
        "membership_api": "query_hs300_stocks",
        "bars_api": "query_history_k_data_plus",
        "listing_api": "query_stock_basic",
        "adjustflag": "3",
        "anonymous": True,
        "account_required": False,
        "start": start,
        "end": end,
        "retrieved_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "trading_days": len(dates),
        "membership_rows": membership_rows,
        "bar_rows": bar_rows,
        "listing_rows": len(listings),
        "files": files,
    }


class BaoStockClient:
    """Anonymous session. Login is not an account registration."""

    def __init__(self):
        import baostock as bs

        self._bs = bs
        self._logged = False

    def reconnect(self):
        if self._logged:
            self._bs.logout()
            self._logged = False
        self._ensure()

    def _ensure(self):
        if self._logged:
            return
        result = self._bs.login()
        if result.error_code != "0":
            raise RuntimeError(result.error_msg)
        self._logged = True

    def _rows(self, result):
        if result.error_code != "0":
            raise RuntimeError(result.error_msg)
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        return result.fields, rows

    def trading_days(self, start, end):
        self._ensure()
        fields, rows = self._rows(self._bs.query_trade_dates(start_date=start, end_date=end))
        flag = fields.index("is_trading_day")
        day = fields.index("calendar_date")
        return [row[day] for row in rows if row[flag] == "1"]

    def constituents(self, date):
        self._ensure()
        fields, rows = self._rows(self._bs.query_hs300_stocks(date=date))
        return [dict(zip(fields, row)) for row in rows]

    def bars(self, code, start, end):
        self._ensure()
        fields, rows = self._rows(
            self._bs.query_history_k_data_plus(
                code,
                ",".join(BAR_FIELDS),
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3",
            )
        )
        return [dict(zip(fields, row)) for row in rows]

    def listing(self, code):
        self._ensure()
        fields, rows = self._rows(self._bs.query_stock_basic(code=code))
        if not rows:
            raise RuntimeError(f"no listing for {code}")
        return dict(zip(fields, rows[0]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--reuse-search", action="append", default=None)
    args = parser.parse_args()
    roots = list(args.reuse_search or [])
    if not roots and Path("/kaggle/input").is_dir():
        roots = ["/kaggle/input"]
    info = resolve_pit(
        args.output_dir,
        client=None if roots else BaoStockClient(),
        search_roots=roots,
        sleep=lambda seconds: time.sleep(max(seconds, args.sleep)),
        retries=args.retries,
    )
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
