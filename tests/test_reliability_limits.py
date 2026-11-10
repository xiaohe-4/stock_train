import json
from pathlib import Path

import pandas as pd

from research.reliability.audit import audit_frame
from research.reliability.limits import limit_price, limit_status, reject_reason

PROBE = Path("research/reliability/fixtures/baostock_pit_probe_2026-11-10.json")
DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


def test_main_board_limit_rounds_half_up_to_fen():
    assert limit_price(10.06, 0.10) == 11.07
    assert limit_price(6.62, 0.10) == 7.28


def test_sealed_limit_up_blocks_a_buy_on_raw_prices():
    status = limit_status("600000", is_st=0, ipo_date="1999-11-10", trade_date="2024-01-03", trading_days=DAYS)
    reason = reject_reason(
        {"open": 11.07, "high": 11.07, "low": 11.07, "preclose": 10.06, "volume": 100, "tradestatus": "1"},
        "buy",
        status,
    )
    assert reason == "limit_up_sealed"


def test_missing_st_flag_fails_closed():
    status = limit_status("600000", is_st=None, ipo_date="1999-11-10", trade_date="2024-01-03", trading_days=DAYS)
    assert status["state"] == "unverified"
    assert reject_reason({"open": 10, "high": 10.2, "low": 9.8, "preclose": 10, "volume": 100, "tradestatus": "1"}, "buy", status) == "rule_unverified"


def test_main_board_st_is_five_percent():
    status = limit_status("600000", is_st=1, ipo_date="1999-11-10", trade_date="2024-01-03", trading_days=DAYS)
    assert status["state"] == "ratio"
    assert status["up"] == 0.05
    assert status["down"] == 0.05


def test_chinext_st_uses_twenty_percent_after_the_listing_window():
    status = limit_status("300001", is_st=1, ipo_date="2009-10-30", trade_date="2024-01-03", trading_days=DAYS)
    assert status["state"] == "ratio"
    assert status["up"] == 0.20


def test_recent_chinext_listing_has_no_limit_on_day_five():
    calendar = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
    status = limit_status("300999", is_st=0, ipo_date="2024-01-02", trade_date="2024-01-08", trading_days=calendar)
    assert status["state"] == "no_limit"


def test_hfq_snapshot_is_not_strict():
    prices = pd.DataFrame([{"股票代码": "600000", "日期": "2024-01-02", "开盘": 80.0, "收盘": 79.0, "最高": 81.0, "最低": 78.0, "成交量": 10}])
    audit = audit_frame(prices, membership=None, listings=None)
    assert audit["strict_ready"] is False
    assert audit["mode"] == "research"
    text = " ".join(audit["blockers"])
    assert "幸存者" in text
    assert "不复权" in text


def test_probe_fixture_has_source_hash_and_raw_adjustflag():
    source = json.loads(Path("research/reliability/fixtures/SOURCE.json").read_text(encoding="utf-8"))
    payload = json.loads(PROBE.read_text(encoding="utf-8"))
    assert source["sha256"]
    assert source["source"] == "baostock.query_hs300_stocks"
    assert payload["sh600000_bars"][0]["adjustflag"] == "3"
    assert "preclose" in payload["sh600000_bars"][0]
