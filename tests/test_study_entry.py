import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.study import build_samples

SNAPSHOT = Path("/opt/cursor/artifacts/hs300_daily_hfq_20240101_complete.csv")


def test_build_samples_uses_the_real_train_feature_map():
    import train

    assert hasattr(train, "feature_cloums_map")
    assert "158+39" in train.feature_cloums_map
    assert "158+39" in train.feature_engineer_func_map
    source = Path("research/study.py").read_text(encoding="utf-8")
    assert "feature_cloums_map" in source
    assert "feature_cols_map" not in source


def test_real_snapshot_smoke_reaches_outputs(tmp_path):
    from research.study import smoke_snapshot_entry
    assert SNAPSHOT.is_file(), SNAPSHOT
    panel = pd.read_csv(SNAPSHOT, dtype={"股票代码": str})
    panel["股票代码"] = panel["股票代码"].astype(str).str.zfill(6)
    present = set(panel.loc[panel["日期"].astype(str).str.slice(0, 10) == "2026-11-09", "股票代码"])
    codes = sorted(present)[:16]
    subset = panel[panel["股票代码"].isin(codes)].copy()
    summary = smoke_snapshot_entry(subset, tmp_path)
    assert set(summary["steps"]) == {"transformer", "lstm", "memory"}
    for arm, step in summary["steps"].items():
        assert step["loss_finite"] is True
        assert step["grad_norm"] > 0
    assert summary["dev_days"] >= 1
    assert summary["status"] in {"research_candidate", "do_not_adopt"}
    assert (tmp_path / "research_summary.json").is_file()
    assert (tmp_path / "research_trades.csv").is_file()
    written = json.loads((tmp_path / "research_summary.json").read_text(encoding="utf-8"))
    assert written["asof"] == "2026-11-09"
    assert "nav" in written["runs"][0]["dev_nav"]
    # The import path inside build_samples must have succeeded on this frame.
    samples, _, _ = build_samples(subset)
    assert any(sample["split"] == "train" for sample in samples)
    assert any(sample["split"] == "dev" for sample in samples)
