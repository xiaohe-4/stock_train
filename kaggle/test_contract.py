#!/usr/bin/env python3
"""Contract checks that do not train a model or download a market panel."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_driver import (  # noqa: E402
    PINNED_COMMIT,
    materialize_split,
    verify_pinned_sources,
    verify_ranking,
)
from refresh_hs300 import calendar_cap  # noqa: E402


def test_calendar_cap() -> None:
    days = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]
    shanghai = ZoneInfo("Asia/Shanghai")
    noon = datetime(2026, 11, 10, 12, 0, tzinfo=shanghai)
    assert calendar_cap(noon, days) == "2026-11-09"
    assert calendar_cap(datetime(2026, 11, 10, 15, 0, tzinfo=shanghai), days) == "2026-11-10"
    assert calendar_cap(datetime(2026, 11, 8, 12, 0, tzinfo=shanghai), days) == "2026-11-06"
    try:
        calendar_cap(datetime(2026, 11, 10, 12, 0), days)
    except ValueError:
        pass
    else:
        raise AssertionError("naive timestamps must be rejected")


def test_pinned_sources() -> None:
    verify_pinned_sources(Path("/workspace"))
    assert PINNED_COMMIT == "c735797c86ec3908379f7f54b70953c6a62b23b9"


def test_materialize_and_ranking_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        panel = root / "panel.csv"
        frame = pd.DataFrame(
            {
                "股票代码": ["000001", "600000"],
                "日期": ["2026-11-06", "2026-11-09"],
                "开盘": [1, 2],
            }
        )
        # The materializer only needs code and date plus the other columns it copies through.
        frame.to_csv(panel, index=False, encoding="utf-8-sig")
        app = root / "app"
        split = materialize_split(panel, app)
        assert split["train_sha256"] == split["test_sha256"]
        assert split["max_date"] == "2026-11-09"

        ranking = root / "ranking_full.csv"
        result = root / "result.csv"
        rows = []
        for rank, code, score in [(1, "600000", 1.5), (2, "000001", 1.2), (3, "600004", 0.2), (4, "600005", 0.1), (5, "600006", -0.1)]:
            rows.append(
                {
                    "rank": rank,
                    "stock_code": code,
                    "stock_name": "名称" + code,
                    "model_score": score,
                    "horizon": "未来5个交易日开盘到开盘收益: (open[t+5]-open[t+1])/open[t+1]",
                    "asof_date": "2026-11-09",
                    "stale_bar": False,
                    "seeds": "42,123",
                }
            )
        # Verification requires at least 50 rows.
        for offset in range(50):
            rows.append(
                {
                    "rank": 6 + offset,
                    "stock_code": f"{700000 + offset:06d}",
                    "stock_name": "名称",
                    "model_score": -1 - offset / 100,
                    "horizon": "未来5个交易日开盘到开盘收益: (open[t+5]-open[t+1])/open[t+1]",
                    "asof_date": "2026-11-09",
                    "stale_bar": False,
                    "seeds": "42,123",
                }
            )
        pd.DataFrame(rows).to_csv(ranking, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "stock_id": ["600000", "000001", "600004", "600005", "600006"],
                "weight": [0.50, 0.47, 0.01, 0.01, 0.01],
            }
        ).to_csv(result, index=False)
        errors = verify_ranking(ranking, result, {"run_mode": "full", "expected_asof_date": "2026-11-09"})
        assert errors == [], errors
        broken = pd.read_csv(ranking, encoding="utf-8-sig")
        broken.loc[0, "model_score"] = -5
        broken.to_csv(ranking, index=False, encoding="utf-8-sig")
        errors = verify_ranking(ranking, result, {"run_mode": "full", "expected_asof_date": "2026-11-09"})
        assert any("sorted" in item for item in errors)


def test_predict_alignment() -> None:
    predict = Path("/workspace/app/code/predict.py").read_text(encoding="utf-8")
    export = Path("/workspace/kaggle/export_ranking.py").read_text(encoding="utf-8")
    for snippet in ["vol_mean + 2.0 * vol_std", "predict_use_leader_pool", "best_model.pth"]:
        assert snippet in predict
        assert snippet in export
    assert "np.mean(all_scores, axis=0)" in predict
    assert "np.mean(per_seed, axis=0)" in export
    config = Path("/workspace/app/code/config.py").read_text(encoding="utf-8")
    assert "'ensemble_seeds': [42, 123]" in config
    assert "'num_epochs': 80" in config
    assert "'train_leader_ratio': 0.75" in config
    assert "'feature_num': feature_num" in config


if __name__ == "__main__":
    test_calendar_cap()
    test_pinned_sources()
    test_materialize_and_ranking_contract()
    test_predict_alignment()
    print(json.dumps({"contract_tests": "passed"}, ensure_ascii=False))
