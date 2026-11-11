import json

import pandas as pd

from research.reliability.fetch_pit import fetch_pit
from research.reliability.strict_run import run_strict
from tests.test_reliability_fetch import FlakyClient


def _dates(count):
    return [day.strftime("%Y-%m-%d") for day in pd.bdate_range("2024-01-02", periods=count)]


def _fetch(tmp_path, count, name):
    dates = _dates(count)
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    root = tmp_path / name
    fetch_pit(root, client, start=dates[0], end=dates[-1], constituents_per_day=2, sleep=lambda _seconds: None, retries=3)
    return root, dates


def test_audit_failure_does_not_train(tmp_path):
    root, dates = _fetch(tmp_path, 3, "pit")
    next((root / "bars").glob("*.csv")).unlink()
    result = run_strict(
        root,
        tmp_path / "out",
        expected_days=3,
        constituents_per_day=2,
        start=dates[0],
        end=dates[-1],
    )
    assert result["trained"] is False
    assert result["strict"] is False
    assert result["status"] == "fail"
    report = json.loads((tmp_path / "out" / "audit_report.json").read_text(encoding="utf-8"))
    assert report["strict_ready"] is False
    assert not list((tmp_path / "out").glob("fold_*"))


def test_tiny_strict_run_writes_every_blotter(tmp_path):
    root, dates = _fetch(tmp_path, 80, "pit")
    fold = {
        "id": "S",
        "train_end": dates[55],
        "dev_start": dates[56],
        "dev_end": dates[64],
        "test_start": dates[65],
        "test_end": dates[74],
        "evidence_grade": "smoke",
    }
    result = run_strict(
        root,
        tmp_path / "out",
        folds=[fold],
        max_epochs=1,
        patience=12,
        seeds=(42, 123),
        min_names=2,
        expected_days=80,
        constituents_per_day=2,
        start=dates[0],
        end=dates[-1],
        deadline_seconds=3600,
    )
    assert result["strict"] is True
    assert result["trained"] is True
    assert result["complete"] is True
    assert result["status"] in {"abstain", "process_only"}
    assert result["status"] != "adopt"
    assert result["replaces_v4"] is False
    assert result["feature_map"] == "feature_cloums_map"
    assert result["models"] == ["equal_weight", "ridge", "tree", "transformer"]
    assert result["decision_uses"] == "dev_only"
    assert result["test_reports"]
    daily = pd.read_csv(tmp_path / "out" / "fold_S_transformer_seed_42_x1" / "daily_nav.csv")
    trades = pd.read_csv(tmp_path / "out" / "fold_S_transformer_seed_42_x1" / "trades.csv")
    rejects = pd.read_csv(tmp_path / "out" / "fold_S_equal_weight_seed_123_x3" / "rejects.csv")
    assert len(daily) >= 2
    assert "nav" in daily.columns
    assert "reason" in trades.columns or "reason" in rejects.columns
    assert (tmp_path / "out" / "fold_S_tree_seed_123_x2" / "trades.csv").is_file()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["enable_memory"] is False
    assert "memory" not in summary["models"]
    assert summary["nav_metrics_gated"] is False
    assert summary["early_stop"] == "inner_train_top5_label"
    assert summary["data_source"] == "downloaded"
    assert "sample_std" in summary["sharpe_formula"]
    assert "sqrt(252)" in summary["sharpe_formula"]
    transformer_rows = [row for row in summary["dev_reports"] if row["model"] == "transformer"]
    assert {row["seed"] for row in transformer_rows} == {42, 123}
    assert summary["feature_diagnostics"]
    assert (tmp_path / "out" / "fold_S_diagnostics.json").is_file()
    for row in summary["dev_reports"] + summary["test_reports"]:
        assert isinstance(row["stress_excess_3x"], float)
        assert row["nonfinite_features_after"] == 0
        assert row["nonfinite_scores_after"] == 0
        assert isinstance(row["cumulative_return"], float)
        assert isinstance(row["benchmark_cumulative_return"], float)
        assert isinstance(row["max_drawdown"], float)
        assert isinstance(row["average_turnover"], float)
        assert isinstance(row["total_fees"], float)
        assert "sharpe_annualized" in row
