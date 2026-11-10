"""Dev-only stability gate. There is no adopt status."""

from __future__ import annotations


def decide(mode: str, strict_ready: bool, dev_reports: list[dict]) -> str:
    if mode != "strict" or not strict_ready:
        return "abstain"
    grouped: dict[tuple, list] = {}
    for row in dev_reports:
        grouped.setdefault((row["model"], row["fold"]), []).append(row)
    if not grouped:
        return "abstain"
    for rows in grouped.values():
        excesses = [float(row["excess"]) for row in rows]
        stressed = [float(row["stress_excess"]) for row in rows]
        if any(value <= 0 for value in excesses + stressed):
            return "abstain"
        signs = {value > 0 for value in excesses}
        if len(signs) != 1:
            return "abstain"
    return "process_only"
