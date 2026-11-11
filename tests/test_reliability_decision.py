import json

import numpy as np
import pytest

from research.reliability.baselines import equal_weights, fit_linear, fit_tree
from research.reliability.decision import decide
from research.reliability.prospective import archive_predictions, freeze_config


def test_linear_and_tree_predictions_are_finite():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float)
    linear = fit_linear(x, y)
    tree = fit_tree(x, y, seed=42)
    assert np.isfinite(linear.predict(x)).all()
    assert np.isfinite(tree.predict(x)).all()
    assert equal_weights(["000001", "000002"]) == {"000001": 0.5, "000002": 0.5}


def test_decision_ignores_test_results_and_abstains_on_disagreement():
    dev = [
        {"model": "ridge", "fold": "A", "seed": 42, "excess": 0.01, "stress_excess": 0.01, "test_excess": -1.0},
        {"model": "ridge", "fold": "A", "seed": 123, "excess": -0.02, "stress_excess": 0.01, "test_excess": 1.0},
    ]
    assert decide("strict", True, dev) == "abstain"


def test_gate_ignores_three_x_drawdown_and_sharpe():
    dev = [
        {
            "model": "ridge",
            "fold": "A",
            "seed": seed,
            "excess": 0.01,
            "stress_excess": 0.02,
            "stress_excess_3x": -1.0,
            "sharpe_annualized": -3.0,
            "max_drawdown": 0.8,
            "cumulative_return": -0.5,
        }
        for seed in (42, 123, 2024)
    ]
    assert decide("strict", True, dev) == "process_only"


def test_research_mode_abstains_even_when_dev_excess_is_positive():
    dev = [
        {"model": "ridge", "fold": "A", "seed": 42, "excess": 0.01, "stress_excess": 0.01},
        {"model": "ridge", "fold": "A", "seed": 123, "excess": 0.02, "stress_excess": 0.01},
    ]
    assert decide("research", False, dev) == "abstain"
    assert decide("strict", True, dev) == "process_only"


def test_archive_refuses_a_strict_claim_when_the_audit_failed(tmp_path):
    frozen = freeze_config(tmp_path / "config.json", "abc123")
    assert frozen["commit"] == "abc123"
    assert frozen["holding_id"] == "open_t_plus_1_to_open_t_plus_5"
    with pytest.raises(RuntimeError, match="strict"):
        archive_predictions(
            tmp_path / "predictions.jsonl",
            [{"asof": "2026-11-09", "strict": True, "score": 0.1}],
            {"strict_ready": False},
        )
