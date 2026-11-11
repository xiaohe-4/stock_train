import numpy as np
import torch

from research.protocol import SEEDS
from research.reliability.strict_run import RELIABILITY_SEEDS, _ranked, _scores, _seed, _train_transformer, nav_metrics


def _samples():
    samples = []
    for day in range(4):
        generator = np.random.default_rng(1000 + day)
        features = generator.normal(size=(4, 30, 8)).astype(np.float32)
        features += np.arange(4, dtype=np.float32)[:, None, None]
        labels = generator.normal(size=(4,)).astype(np.float32)
        date = f"2024-01-{day + 2:02d}"
        samples.append(
            {
                "date": date,
                "codes": ["600000", "600132", "600519", "601318"],
                "x": features,
                "raw": labels,
                "train_target": labels.copy(),
                "entry_date": date,
                "exit_date": date,
            }
        )
    return samples


def _flat(model):
    return torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])


def test_default_seeds_are_three_and_the_old_study_seeds_stay_put():
    assert RELIABILITY_SEEDS == (42, 123, 2024)
    assert SEEDS == (42, 123)


def test_seed_sets_cudnn_determinism():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    _seed(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_transformer_seeds_change_weights_and_predictions_and_repeat():
    device = torch.device("cpu")
    samples = _samples()
    first, first_meta = _train_transformer(samples, 42, 1, 12, device, deadline=1e18)
    second, _second_meta = _train_transformer(samples, 123, 1, 12, device, deadline=1e18)
    repeat, _repeat_meta = _train_transformer(samples, 42, 1, 12, device, deadline=1e18)
    assert first_meta["uses_fold_dev"] is False
    assert first_meta["early_stop"] == "inner_train_top5_label"
    assert first_meta["inner_val_dates"]
    assert set(first_meta["inner_val_dates"]).isdisjoint(first_meta["fit_dates"])
    weights_first = _flat(first)
    weights_second = _flat(second)
    weights_repeat = _flat(repeat)
    assert not torch.equal(weights_first, weights_second)
    assert torch.equal(weights_first, weights_repeat)
    pred_first = _scores("transformer", samples[0], first, device)
    pred_second = _scores("transformer", samples[0], second, device)
    pred_repeat = _scores("transformer", samples[0], repeat, device)
    assert np.isfinite(pred_first).all()
    assert not np.array_equal(pred_first, pred_second)
    assert np.array_equal(pred_first, pred_repeat)


def test_non_finite_scores_are_not_ranked_into_a_stable_book():
    try:
        _ranked(["600000", "600132"], np.array([float("nan"), 1.0]))
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("tied NaN scores must not become a seed-invariant portfolio")


def test_nav_metrics_are_computed_from_daily_nav_and_not_a_gate():
    daily = [
        {"nav": 1_000_000.0, "turnover": 0.0, "fees": 0.0},
        {"nav": 1_100_000.0, "turnover": 0.2, "fees": 10.0},
        {"nav": 900_000.0, "turnover": 0.1, "fees": 5.0},
    ]
    metrics = nav_metrics(daily)
    assert metrics["cumulative_return"] == 900_000.0 / 1_000_000.0 - 1.0
    assert abs(metrics["max_drawdown"] - (1_100_000.0 - 900_000.0) / 1_100_000.0) < 1e-12
    assert metrics["average_turnover"] == (0.0 + 0.2 + 0.1) / 3
    assert metrics["total_fees"] == 15.0
    assert metrics["sharpe_annualized"] is not None
