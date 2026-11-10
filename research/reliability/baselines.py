"""Frozen lightweight comparisons. No hyperparameter search."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def fit_linear(x, y):
    model = Ridge(alpha=1.0)
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return model


def fit_tree(x, y, seed: int):
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=50,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return model
