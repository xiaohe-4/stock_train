"""Dev-only stability gate. There is no adopt status."""

from __future__ import annotations


def choose(mode: str, strict_ready: bool, dev_reports: list[dict]) -> dict:
    """Pick one model that passes every dev fold. Test rows are not accepted here."""
    if mode != "strict" or not strict_ready or not dev_reports:
        return {"status": "abstain", "model": None}
    passing = []
    for model in sorted({row["model"] for row in dev_reports}):
        subset = [row for row in dev_reports if row["model"] == model]
        if decide("strict", True, subset) == "process_only":
            mean = sum(float(row["excess"]) for row in subset) / len(subset)
            passing.append((mean, model))
    if not passing:
        return {"status": "abstain", "model": None}
    passing.sort(key=lambda item: (-item[0], item[1]))
    return {"status": "process_only", "model": passing[0][1]}


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
