import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.backtest import (
    buy_blocked_unadjusted,
    day_portfolio,
    decide,
    entry_blocked,
    limits_applied_to_hfq_panel,
    mean_excess,
    period_nav,
    select_nonoverlapping,
    top_k_jaccard,
)


def test_nonoverlapping_nav_does_not_compound_every_overlapping_signal():
    dates = [f"2026-06-{day:02d}" for day in range(1, 11)]
    chosen = select_nonoverlapping(dates)
    assert chosen == [dates[0], dates[5]]
    records = []
    for index, date in enumerate(dates):
        port = 0.01 if index == 0 else 0.03 if index == 5 else 0.20
        records.append(
            {
                "date": date,
                "split": "dev",
                "port_return": port,
                "ew_return": 0.0,
                "turnover": 0.0,
            }
        )
    result = period_nav(records, split="dev")
    assert result["periods"] == 2
    assert result["nav"] == math.prod([1.01, 1.03])
    assert abs(result["nav"] - (1.20**10)) > 0.1
    assert result["sharpe_observations"] == 2
    assert result["annualization_unstable"] is True
    assert "年化" in result["annualization_warning"]


def test_runtime_budget_can_reject_without_reading_returns():
    from research.backtest import apply_runtime_budget

    assert apply_runtime_budget("research_candidate", {"memory": 3.0, "transformer": 1.0}) == "do_not_adopt"
    assert apply_runtime_budget("research_candidate", {"memory": 1.5, "transformer": 1.0}) == "research_candidate"


def test_appendix_rows_do_not_change_development_selection_score():
    dev = {"date": "2026-06-01", "split": "dev", "port_return": 0.02, "ew_return": 0.01, "turnover": 1.0}
    appendix = {"date": "2026-09-09", "split": "appendix", "port_return": 5.0, "ew_return": 0.0, "turnover": 0.0}
    alone = mean_excess([dev], split="dev")
    with_appendix = mean_excess([dev, appendix], split="dev")
    assert alone == with_appendix
    assert decide({"memory": 0.02, "transformer": 0.01, "lstm": 0.0}) == "research_candidate"
    assert decide({"memory": 0.0, "transformer": 0.01, "lstm": 0.02}) == "do_not_adopt"


def test_halt_blocks_entry_and_unadjusted_limit_up_is_sealed():
    assert entry_blocked(0)
    assert entry_blocked(float("nan"))
    assert entry_blocked(None) is True
    assert entry_blocked(100) is False
    assert buy_blocked_unadjusted(11.0, 11.0, 10.0, "600000", 1000) is True
    assert buy_blocked_unadjusted(10.2, 10.8, 10.0, "600000", 1000) is False
    assert limits_applied_to_hfq_panel() is False


def test_untradable_top_name_is_removed_before_the_fixed_weights_are_applied():
    result = day_portfolio(
        codes=["000001", "000002", "000003", "000004", "000005"],
        scores=[5, 4, 3, 2, 1],
        raw=[0.10, 0.20, 0.0, 0.0, 0.0],
        tradable=[False, True, True, True, True],
        previous={},
    )
    assert result["weights"]["000001"] == 0.0
    assert abs(result["port_return"] - 0.188) < 1e-9


def test_top_k_overlap_is_jaccard_of_codes():
    assert top_k_jaccard(["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "6"]) == 4 / 6
