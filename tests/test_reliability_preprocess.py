import numpy as np
import pandas as pd
import torch

from research.reliability.baselines import fit_linear
from research.reliability.preprocess import apply_standardizer, fit_standardizer, prepare_fold
from research.reliability.strict_run import _ranked


def _sample(date, column_value, labels=(0.01, 0.02)):
    values = np.zeros((2, 4, 3), dtype=np.float32)
    values[..., 0] = column_value
    labels = np.asarray(labels, dtype=np.float32)
    return {
        "date": date,
        "codes": ["600000", "600519"],
        "x": values,
        "raw": labels.copy(),
        "train_target": labels.copy(),
        "entry_date": date,
        "exit_date": date,
    }


def test_scaler_is_fit_on_train_only_and_nonfinite_names_are_dropped():
    train = [_sample("2024-01-02", 1.0), _sample("2024-01-03", 3.0)]
    broken = _sample("2024-01-04", 1.0)
    broken["raw"] = np.asarray([0.01, np.nan], dtype=np.float32)
    broken["x"][1] = np.nan
    train.append(broken)
    original = train[0]["x"].copy()
    dev = [_sample("2024-02-01", 1e20)]
    stats = fit_standardizer(train[:2])
    assert stats["mean"][0] == np.float64(2.0) or abs(stats["mean"][0] - 2.0) < 1e-12
    assert stats["scale"][0] >= stats["epsilon"]
    train_out, dev_out, test_out, report = prepare_fold(train, dev, [], min_names=1)
    assert test_out == []
    assert report["names_dropped"] == 1
    assert report["nonfinite_features_after"] == 0
    assert report["max_abs_feature_after"] <= report["clip"]
    assert float(np.max(np.abs(dev_out[0]["x"]))) <= report["clip"]
    assert np.array_equal(train[0]["x"], original)
    assert len(train_out[-1]["codes"]) == 1


def test_ranker_still_rejects_nonfinite_scores():
    try:
        _ranked(["600000", "600519"], np.asarray([np.nan, 0.2]))
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite scores must fail closed")


def test_zero_volume_ratio_is_finite_but_breaks_the_transformer_until_cleaned():
    import sys
    from pathlib import Path

    code = Path(__file__).resolve().parents[1] / "app" / "code"
    sys.path.insert(0, str(code))
    from config import config
    from research.models import build_model
    from train import feature_cloums_map
    from utils import engineer_features_158plus39

    rng = np.random.default_rng(1)
    columns = feature_cloums_map["158+39"]
    matrices = []
    for code_i in range(3):
        price = [8.0, 1500.0, 40.0][code_i]
        rows = []
        for day in range(70):
            halted = code_i == 1 and day == 65
            previous = price
            shock = 0.0 if halted else float(rng.normal(0, 0.01))
            price = max(0.5, price * (1.0 + shock))
            volume = 0.0 if halted else float(rng.uniform(2e6, 5e8))
            amount = 0.0 if halted else price * volume * 100.0
            rows.append(
                {
                    "开盘": previous,
                    "收盘": price,
                    "最高": max(previous, price) * 1.01,
                    "最低": min(previous, price) * 0.99,
                    "成交量": volume,
                    "成交额": amount,
                    "振幅": 0.0,
                    "涨跌额": price - previous,
                    "换手率": 0.0 if halted else 0.5,
                    "涨跌幅": 0.0 if previous == 0 else (price - previous) / previous * 100.0,
                }
            )
        frame = engineer_features_158plus39(pd.DataFrame(rows))
        matrices.append(frame[columns].to_numpy(dtype=np.float32))
    window = np.stack([matrix[-30:] for matrix in matrices])
    assert int((~np.isfinite(window)).sum()) == 0
    assert float(np.max(np.abs(window))) > 1e10
    torch.manual_seed(42)
    raw_model = build_model("transformer", window.shape[-1], num_stocks=window.shape[0], config=config)
    raw_model.eval()
    with torch.no_grad():
        raw_scores = raw_model(torch.from_numpy(window).unsqueeze(0))
    assert not torch.isfinite(raw_scores).all()

    samples = []
    for row in range(40):
        block = np.stack([matrix[row : row + 30] for matrix in matrices]).astype(np.float32)
        samples.append(
            {
                "date": f"2024-03-{row + 1:02d}" if row < 9 else f"2024-04-{row - 8:02d}",
                "codes": ["600000", "600519", "300750"],
                "x": block,
                "raw": np.asarray([0.01, -0.02, 0.03], dtype=np.float32),
                "train_target": np.asarray([0.01, -0.02, 0.03], dtype=np.float32),
                "entry_date": "2024-05-01",
                "exit_date": "2024-05-08",
            }
        )
    train_out, dev_out, _, report = prepare_fold(samples[:30], samples[30:], [], min_names=2)
    assert report["nonfinite_features_before"] == 0
    assert report["nonfinite_features_after"] == 0
    assert report["max_abs_feature_before"] > 1e10
    assert report["max_abs_feature_after"] <= 10.0
    torch.manual_seed(42)
    cleaned_model = build_model("transformer", dev_out[0]["x"].shape[-1], num_stocks=dev_out[0]["x"].shape[0], config=config)
    cleaned_model.eval()
    with torch.no_grad():
        for sample in dev_out:
            cleaned_scores = cleaned_model(torch.from_numpy(sample["x"]).unsqueeze(0))
            assert torch.isfinite(cleaned_scores).all()

    raw_rows = np.concatenate([sample["x"][:, -1, :] for sample in samples[:30]]).astype(np.float64)
    labels = np.resize(np.asarray([0.01, -0.02, 0.03], dtype=np.float64), len(raw_rows))
    raw_ridge = fit_linear(np.nan_to_num(raw_rows), labels)
    clean_rows = np.concatenate([sample["x"][:, -1, :] for sample in train_out]).astype(np.float64)
    clean_ridge = fit_linear(clean_rows, labels[: len(clean_rows)])
    assert np.isfinite(clean_ridge.predict(clean_rows[:6])).all()
    assert np.isfinite(apply_standardizer(window, fit_standardizer(train_out))).all()
    assert raw_ridge.coef_.shape[-1] == window.shape[-1]
