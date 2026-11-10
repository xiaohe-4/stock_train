import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.windows import fixed_window_end_indices, legacy_drops_extra_five, window_end_dates


def test_legacy_predicate_drops_last_labeled_day_after_label_rows_were_removed():
    n_labeled = 35
    sequence_length = 30
    last = n_labeled - 1
    assert last == 34
    assert legacy_drops_extra_five(n_labeled, last)
    kept = fixed_window_end_indices(n_labeled, sequence_length)
    assert kept[0] == sequence_length - 1
    assert kept[-1] == last


def test_frame_keeps_the_last_finite_label():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    label = np.ones(40)
    label[-5:] = np.nan
    frame = pd.DataFrame({"日期": dates.strftime("%Y-%m-%d"), "feat": np.arange(40), "label": label})
    ends = window_end_dates(frame, ["feat"], sequence_length=30)
    assert ends[-1] == dates[34].strftime("%Y-%m-%d")
