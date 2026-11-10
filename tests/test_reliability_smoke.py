import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.reliability.smoke import run_full, run_smoke

SNAPSHOT = Path("/opt/cursor/artifacts/hs300_daily_hfq_20240101_complete.csv")
PROBE = Path("research/reliability/fixtures/baostock_pit_probe_2026-11-10.json")


def test_smoke_uses_real_features_and_abstains(tmp_path):
    summary = run_smoke(SNAPSHOT, PROBE, tmp_path)
    assert summary["mode"] == "research"
    assert summary["strict"] is False
    assert summary["status"] == "abstain"
    assert summary["feature_map"] == "feature_cloums_map"
    assert summary["steps"]["transformer"]["loss_finite"] is True
    assert summary["steps"]["transformer"]["grad_norm"] > 0
    assert summary["steps"]["linear"]["finite"] is True
    assert summary["steps"]["tree"]["finite"] is True
    assert summary["steps"]["equal_weight"]["names"] >= 2
    assert summary["raw_limit_check"] == "limit_up_sealed"
    assert (tmp_path / "daily_nav.csv").is_file()
    assert (tmp_path / "trades.csv").is_file()
    daily = pd.read_csv(tmp_path / "daily_nav.csv")
    assert len(daily) >= 2
    assert not (tmp_path / "result.csv").exists()
    for fold in ("A", "B", "C"):
        for seed in (42, 123):
            path = tmp_path / f"fold_{fold}_seed_{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["trained"] is False
            assert payload["evidence_grade"] == "already_observed"
            assert "trades" in payload
    blocked = run_full(SNAPSHOT, tmp_path / "full")
    assert blocked["trained"] is False
    assert blocked["status"] == "abstain"
