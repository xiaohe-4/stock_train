import pandas as pd

from research.reliability.costs import charge
from research.reliability.folds import FOLDS, clip_bounds, purge_train
from research.reliability.ledger import run_account

DAYS = [f"2024-01-{day:02d}" for day in range(2, 12)]


def _bars(days, code="600000", tradestatus="1", open_price=10.0, close_price=10.0):
    return pd.DataFrame(
        [
            {
                "date": day,
                "code": code,
                "open": open_price if day != days[5] else 12.0,
                "high": 12.5,
                "low": 9.5,
                "close": close_price,
                "preclose": 10.0,
                "volume": 100000,
                "tradestatus": tradestatus,
                "isST": "0",
            }
            for day in days
        ]
    )


def test_purge_drops_labels_whose_exit_is_after_the_train_end():
    rows = [
        {"signal_date": "2024-01-02", "exit_date": "2024-01-08", "label": 0.1},
        {"signal_date": "2024-01-03", "exit_date": "2024-01-09", "label": 0.2},
    ]
    kept = purge_train(rows, "2024-01-08")
    assert [row["signal_date"] for row in kept] == ["2024-01-02"]


def test_clip_bounds_come_from_the_training_labels_only():
    narrow = clip_bounds([0.0, 0.5, 1.0])
    wide = clip_bounds([0.0, 0.5, 1.0, 9.0])
    assert wide[1] > narrow[1]


def test_folds_are_the_preregistered_three():
    assert [fold["id"] for fold in FOLDS] == ["A", "B", "C"]
    assert FOLDS[0]["train_end"] == "2024-09-30"
    assert FOLDS[0]["dev_start"] == "2024-10-08"
    assert FOLDS[2]["test_end"] == "2026-03-31"
    assert {fold["evidence_grade"] for fold in FOLDS} == {"already_observed"}


def test_cost_stress_doubles_the_base_schedule():
    assert charge(10000, "buy", 2) == 2 * charge(10000, "buy", 1)
    assert charge(10000, "sell", 1) > charge(10000, "buy", 1)


def test_account_holds_round_lots_and_records_the_cash_day():
    account = run_account(
        _bars(DAYS),
        [{"signal_date": DAYS[0], "entry_date": DAYS[1], "exit_date": DAYS[5], "weights": {"600000": 1.0}}],
        listings={"600000": "1999-11-10"},
        trading_days=DAYS,
        starting_cash=100000,
    )
    by_date = {row["date"]: row for row in account["daily"]}
    assert by_date[DAYS[1]]["shares"]["600000"] % 100 == 0
    assert by_date[DAYS[1]]["shares"]["600000"] > 0
    assert by_date[DAYS[5]]["shares"]["600000"] == 0
    assert by_date[DAYS[6]]["shares"]["600000"] == 0
    assert by_date[DAYS[6]]["nav"] == by_date[DAYS[5]]["nav"]
    assert account["daily"][0]["limits_enforced"] is True
    assert sum(row["fees"] for row in account["fills"]) > 0


def test_halt_and_same_day_sell_are_rejected():
    halted = _bars(DAYS, tradestatus="0")
    account = run_account(
        halted,
        [{"signal_date": DAYS[0], "entry_date": DAYS[1], "exit_date": DAYS[5], "weights": {"600000": 1.0}}],
        listings={"600000": "1999-11-10"},
        trading_days=DAYS,
        starting_cash=100000,
    )
    assert account["rejects"][0]["reason"] == "halt"
    assert all(row["shares"]["600000"] == 0 for row in account["daily"])
    same_day = run_account(
        _bars(DAYS),
        [{"signal_date": DAYS[1], "entry_date": DAYS[1], "exit_date": DAYS[1], "weights": {"600000": 1.0}}],
        listings={"600000": "1999-11-10"},
        trading_days=DAYS,
        starting_cash=100000,
    )
    assert any(row["reason"] == "t_plus_one" for row in same_day["rejects"])
    assert all(row["shares"]["600000"] == 0 for row in same_day["daily"])


def test_small_cash_cannot_buy_a_star_board_lot():
    bars = _bars(DAYS, code="688001")
    account = run_account(
        bars,
        [{"signal_date": DAYS[0], "entry_date": DAYS[1], "exit_date": DAYS[5], "weights": {"688001": 1.0}}],
        listings={"688001": "2020-01-01"},
        trading_days=DAYS,
        starting_cash=1500,
    )
    assert any(row["reason"] == "below_lot" for row in account["rejects"])
