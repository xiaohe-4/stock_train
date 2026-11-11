"""Train-only feature cleaning. Dev and test never enter the scaler."""

from __future__ import annotations

import numpy as np

SCALER_EPSILON = 1e-6
FEATURE_CLIP = 10.0


def _feature_report(samples: list[dict]) -> tuple[int, float]:
    nonfinite = 0
    max_abs = 0.0
    for sample in samples:
        values = np.asarray(sample["x"])
        finite = np.isfinite(values)
        nonfinite += int(values.size - int(finite.sum()))
        if finite.any():
            max_abs = max(max_abs, float(np.max(np.abs(values[finite]))))
    return nonfinite, max_abs


def _keep_mask(sample: dict) -> np.ndarray:
    values = np.asarray(sample["x"])
    labels = np.asarray(sample["raw"], dtype=np.float64)
    finite_label = np.isfinite(labels)
    finite_window = np.isfinite(values).reshape(values.shape[0], -1).any(axis=1)
    return finite_label & finite_window


def _copy_kept(sample: dict, mask: np.ndarray) -> dict:
    index = np.flatnonzero(mask)
    copied = {
        "date": sample["date"],
        "codes": [sample["codes"][i] for i in index],
        "x": np.array(np.asarray(sample["x"])[index], copy=True),
        "raw": np.array(np.asarray(sample["raw"])[index], copy=True),
        "entry_date": sample["entry_date"],
        "exit_date": sample["exit_date"],
    }
    if "train_target" in sample:
        copied["train_target"] = np.array(np.asarray(sample["train_target"])[index], copy=True)
    return copied


def _filter_split(samples: list[dict], min_names: int) -> tuple[list[dict], int, int]:
    kept = []
    names_dropped = 0
    dates_dropped = 0
    for sample in samples:
        mask = _keep_mask(sample)
        names_dropped += int((~mask).sum())
        if int(mask.sum()) < min_names:
            dates_dropped += 1
            continue
        kept.append(_copy_kept(sample, mask))
    return kept, names_dropped, dates_dropped


def fit_standardizer(samples: list[dict], epsilon: float = SCALER_EPSILON) -> dict:
    """Mean and scale from finite training entries only. A zero-variance column uses epsilon."""
    width = int(np.asarray(samples[0]["x"]).shape[-1])
    count = np.zeros(width, dtype=np.float64)
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for sample in samples:
        values = np.asarray(sample["x"], dtype=np.float64)
        finite = np.isfinite(values)
        axes = tuple(range(values.ndim - 1))
        count += finite.sum(axis=axes)
        total += np.where(finite, values, 0.0).sum(axis=axes)
        total_sq += np.where(finite, values * values, 0.0).sum(axis=axes)
    mean = np.zeros(width, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    variance = np.zeros(width, dtype=np.float64)
    np.divide(total_sq, count, out=variance, where=count > 0)
    variance = np.maximum(variance - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(count > 0, np.maximum(scale, epsilon), 1.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return {"mean": mean, "scale": scale, "epsilon": float(epsilon), "clip": FEATURE_CLIP}


def apply_standardizer(values, stats: dict) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    filled = np.where(np.isfinite(raw), raw, stats["mean"])
    scaled = (filled - stats["mean"]) / stats["scale"]
    clipped = np.clip(scaled, -stats["clip"], stats["clip"])
    return clipped.astype(np.float32)


def prepare_fold(train: list[dict], dev: list[dict], test: list[dict], min_names: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Drop invalid names, then scale every split with statistics fit on the training split only."""
    train_kept, train_names, train_dates = _filter_split(train, min_names)
    dev_kept, dev_names, dev_dates = _filter_split(dev, min_names)
    test_kept, test_names, test_dates = _filter_split(test, min_names)
    if not train_kept:
        raise RuntimeError("no training rows left after dropping non-finite labels")
    before_count, before_max = _feature_report(train_kept + dev_kept + test_kept)
    stats = fit_standardizer(train_kept)

    def _transform(samples: list[dict]) -> list[dict]:
        transformed = []
        for sample in samples:
            item = dict(sample)
            item["x"] = apply_standardizer(sample["x"], stats)
            transformed.append(item)
        return transformed

    train_out = _transform(train_kept)
    dev_out = _transform(dev_kept)
    test_out = _transform(test_kept)
    after_count, after_max = _feature_report(train_out + dev_out + test_out)
    report = {
        "nonfinite_features_before": before_count,
        "nonfinite_features_after": after_count,
        "max_abs_feature_before": before_max,
        "max_abs_feature_after": after_max,
        "names_dropped": train_names + dev_names + test_names,
        "dates_dropped": train_dates + dev_dates + test_dates,
        "scaler_epsilon": SCALER_EPSILON,
        "clip": FEATURE_CLIP,
        "scaler_fit_on": "train",
        "ridge_alpha": 1.0,
    }
    return train_out, dev_out, test_out, report
