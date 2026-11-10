import pytest

from research.reliability.holding import rebalance_signals, resolve_holding

DAYS = [f"2024-01-{day:02d}" for day in range(2, 21)]


def test_baseline_holding_is_four_trading_steps():
    holding = resolve_holding(DAYS, "2024-01-02")
    assert holding["holding_id"] == "open_t_plus_1_to_open_t_plus_5"
    assert holding["entry_date"] == "2024-01-03"
    assert holding["exit_date"] == "2024-01-07"
    assert holding["entry_price"] == "open"
    assert holding["exit_price"] == "open"
    assert holding["holding_steps"] == 4


def test_other_holding_ids_are_rejected():
    with pytest.raises(ValueError, match="open_t_plus_1_to_open_t_plus_5"):
        resolve_holding(DAYS, "2024-01-02", holding_id="five_full_sessions")


def test_rebalance_leaves_the_exit_date_in_cash():
    signals = rebalance_signals(DAYS, "2024-01-02", "2024-01-16")
    first = resolve_holding(DAYS, signals[0])
    second = resolve_holding(DAYS, signals[1])
    assert signals == ["2024-01-02", "2024-01-07", "2024-01-12"]
    assert first["exit_date"] == signals[1]
    assert second["entry_date"] == "2024-01-08"
    assert first["exit_date"] not in {first["entry_date"], second["entry_date"]}
