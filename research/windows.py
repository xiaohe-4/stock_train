def legacy_drops_extra_five(n_labeled: int, end_idx: int) -> bool:
    """The v4 predicate after labels have already been dropped.

    `end_idx + 5 >= n` on the shortened series discards five extra labeled days.
    """
    return end_idx + 5 >= n_labeled


def window_end_dates(frame, feature_columns: list[str], sequence_length: int) -> list[str]:
    """Keep a row when it has a finite label and a full history. Do not drop five more."""
    del feature_columns
    ordered = frame.sort_values("日期")
    labels = ordered["label"].to_numpy()
    dates = ordered["日期"].astype(str).tolist()
    ends = []
    for index, label in enumerate(labels):
        if index < sequence_length - 1:
            continue
        if label == label:
            ends.append(dates[index])
    return ends


def fixed_window_end_indices(n_labeled: int, sequence_length: int) -> list[int]:
    """Window ends on a series that already contains only rows with labels."""
    last = n_labeled - 1
    first = sequence_length - 1
    if last < first:
        return []
    return list(range(first, last + 1))
