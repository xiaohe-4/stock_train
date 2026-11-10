"""Offline config freeze and prediction archive. No monitor and no broker."""

from __future__ import annotations

import json
from pathlib import Path

from research.reliability.costs import APPLICABLE, STRESS_MULTIPLIERS
from research.reliability.folds import FOLDS
from research.reliability.holding import BASELINE_HOLDING_ID


def freeze_config(path, commit: str) -> dict:
    payload = {
        "commit": commit,
        "holding_id": BASELINE_HOLDING_ID,
        "holding_steps": 4,
        "folds": list(FOLDS),
        "evidence_grade": "already_observed",
        "cost_assumption": APPLICABLE,
        "stress_multipliers": list(STRESS_MULTIPLIERS),
        "statuses": ["abstain", "process_only"],
        "replaces_v4": False,
        "broker": None,
        "monitor": None,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def archive_predictions(path, rows: list[dict], audit: dict) -> None:
    if any(row.get("strict") for row in rows) and not audit.get("strict_ready"):
        raise RuntimeError("strict archive refused: audit is not strict_ready")
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
