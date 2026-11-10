"""Fixed folds. Their test windows are already inside the observed snapshot."""

from __future__ import annotations

import pandas as pd

FOLDS = (
    {
        "id": "A",
        "train_end": "2024-09-30",
        "dev_start": "2024-10-08",
        "dev_end": "2024-12-31",
        "test_start": "2025-01-02",
        "test_end": "2025-03-31",
        "evidence_grade": "already_observed",
    },
    {
        "id": "B",
        "train_end": "2025-03-31",
        "dev_start": "2025-04-01",
        "dev_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-09-30",
        "evidence_grade": "already_observed",
    },
    {
        "id": "C",
        "train_end": "2025-09-30",
        "dev_start": "2025-10-09",
        "dev_end": "2025-12-31",
        "test_start": "2026-01-05",
        "test_end": "2026-03-31",
        "evidence_grade": "already_observed",
    },
)


def purge_train(rows: list[dict], train_end: str) -> list[dict]:
    return [row for row in rows if row["exit_date"] <= train_end]


def clip_bounds(labels: list[float]) -> tuple[float, float]:
    series = pd.Series(list(labels), dtype="float64")
    return float(series.quantile(0.01)), float(series.quantile(0.99))
