import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.protocol import PARAM_MAX, PARAM_MIN
from research.study import assign_split, panel_date_errors, smoke_three_arms


def test_smoke_reports_finite_steps_for_three_arms():
    report = smoke_three_arms()
    assert set(report) == {"transformer", "lstm", "memory"}
    for stats in report.values():
        assert stats["seconds"] > 0
        assert stats["grad_norm"] > 0
        assert PARAM_MIN <= stats["parameters"] <= PARAM_MAX


def test_research_calendar_is_fixed_and_keeps_the_v4_window_in_the_appendix():
    assert panel_date_errors(["2026-11-09"]) == []
    assert panel_date_errors(["2026-11-10"])
    assert assign_split("2026-05-22") == "train"
    assert assign_split("2026-06-01") == "dev"
    assert assign_split("2026-08-25") == "dev"
    assert assign_split("2026-09-09") == "appendix"
    assert assign_split("2026-10-26") == "appendix"
    assert assign_split("2026-11-02") == "ignore"
