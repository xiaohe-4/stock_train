import json
from pathlib import Path

import pytest

from research.reliability.fetch_pit import fetch_pit


class FlakyClient:
    def __init__(self, dates, codes):
        self.dates = dates
        self.codes = codes
        self.constituent_calls = 0
        self.failures_left = 1
        self.bars_calls = 0

    def trading_days(self, start, end):
        return [day for day in self.dates if start <= day <= end]

    def constituents(self, date):
        self.constituent_calls += 1
        if self.failures_left:
            self.failures_left -= 1
            raise ConnectionError("temporary reset")
        return [{"updateDate": date, "code": code, "code_name": code} for code in self.codes]

    def bars(self, code, start, end):
        self.bars_calls += 1
        rows = []
        previous = 10.0
        for index, date in enumerate(self.dates):
            if date < start or date > end:
                continue
            price = 10.0 + (index % 5) * 0.05
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": f"{price:.4f}",
                    "high": f"{price + 0.2:.4f}",
                    "low": f"{price - 0.1:.4f}",
                    "close": f"{price + 0.05:.4f}",
                    "preclose": f"{previous:.4f}",
                    "volume": "100000",
                    "amount": "1000000.0000",
                    "adjustflag": "3",
                    "tradestatus": "1",
                    "isST": "0",
                    "turn": "0.100000",
                }
            )
            previous = price + 0.05
        return rows

    def listing(self, code):
        return {"code": code, "code_name": code, "ipoDate": "1999-11-10", "outDate": "", "type": "1", "status": "1"}

    def reconnect(self):
        return None


def test_fetch_retries_resumes_and_writes_manifest(tmp_path):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    slept = []
    root = tmp_path / "pit"
    manifest = fetch_pit(
        root,
        client,
        start="2024-01-02",
        end="2024-01-04",
        constituents_per_day=2,
        sleep=slept.append,
        retries=3,
    )
    assert slept
    assert client.constituent_calls == 4
    assert manifest["source"] == "baostock"
    assert manifest["anonymous"] is True
    assert manifest["adjustflag"] == "3"
    assert manifest["trading_days"] == 3
    assert manifest["membership_rows"] == 6
    assert manifest["listing_rows"] == 2
    assert manifest["bar_rows"] == 6
    assert manifest["retrieved_at"]
    stored = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert stored["files"]["membership/2024-01-02.json"]["rows"] == 2
    assert len(stored["files"]["membership/2024-01-02.json"]["sha256"]) == 64
    calls = client.constituent_calls
    bar_calls = client.bars_calls
    fetch_pit(root, client, start="2024-01-02", end="2024-01-04", constituents_per_day=2, sleep=slept.append, retries=3)
    assert client.constituent_calls == calls
    assert client.bars_calls == bar_calls


def test_bad_constituent_count_is_not_saved(tmp_path):
    class ShortClient(FlakyClient):
        def constituents(self, date):
            return [{"updateDate": date, "code": "sh.600000", "code_name": "one"}]

    client = ShortClient(["2024-01-02"], ["sh.600000"])
    with pytest.raises(RuntimeError, match="300|constituent"):
        fetch_pit(tmp_path / "pit", client, start="2024-01-02", end="2024-01-02", retries=1, sleep=lambda _seconds: None)
    assert not list((tmp_path / "pit").glob("membership/*.json"))
