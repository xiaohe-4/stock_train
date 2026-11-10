"""CPU smoke for the reliability protocol. It does not run the 80-epoch budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parents[2] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from research.models import build_model
from research.reliability.audit import audit_frame
from research.reliability.baselines import equal_weights, fit_linear, fit_tree
from research.reliability.decision import decide
from research.reliability.folds import FOLDS
from research.reliability.holding import resolve_holding
from research.reliability.ledger import run_account
from research.reliability.limits import limit_price, limit_status, reject_reason
from research.study import build_samples


def _subset(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["股票代码"] = frame["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d")
    present = set(frame.loc[frame["日期"] == "2026-11-09", "股票代码"])
    codes = sorted(present)[:16]
    return frame[frame["股票代码"].isin(codes)].copy()


def _transformer_step(sample: dict) -> dict:
    from config import config as model_config
    from train import CombinedRankingLoss

    model = build_model("transformer", int(sample["x"].shape[-1]), num_stocks=int(sample["x"].shape[0]), config=model_config)
    loss_fn = CombinedRankingLoss(
        k=model_config.get("topk_mle_k", 3),
        portfolio_weight=model_config.get("portfolio_loss_weight", 0.4),
        temperature=model_config.get("portfolio_temperature", 0.35),
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])
    optimizer.zero_grad(set_to_none=True)
    prediction = model(torch.from_numpy(sample["x"]).unsqueeze(0))
    target = torch.from_numpy(sample["train_target"]).unsqueeze(0)
    loss = loss_fn(prediction, target)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    return {"loss_finite": bool(torch.isfinite(loss.detach())), "grad_norm": grad_norm}


def _raw_limit_check(probe: dict) -> str:
    bars = probe["sh600000_bars"]
    preclose = float(bars[0]["preclose"])
    cap = limit_price(preclose, 0.10)
    status = limit_status("600000", 0, "1999-11-10", bars[0]["date"], [row["date"] for row in bars])
    reason = reject_reason(
        {"open": cap, "high": cap, "low": cap, "preclose": preclose, "volume": 100, "tradestatus": "1"},
        "buy",
        status,
    )
    return reason or ""


def run_smoke(panel_path, probe_path, output_dir) -> dict:
    import train

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(panel_path, dtype={"股票代码": str})
    subset = _subset(panel)
    audit = audit_frame(subset, None, None)
    feature_name = "feature_cloums_map"
    if "158+39" not in getattr(train, feature_name):
        raise RuntimeError("real feature map is unavailable")
    samples, _, _ = build_samples(subset)
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    if not train_samples:
        raise RuntimeError("smoke produced no training windows")
    sample = train_samples[0]
    features = np.nan_to_num(np.concatenate([item["x"][:, -1, :] for item in train_samples]).astype(np.float64))
    labels = np.nan_to_num(np.concatenate([item["raw"] for item in train_samples]).astype(np.float64))
    linear = fit_linear(features, labels)
    tree = fit_tree(features, labels, seed=42)
    linear_pred = linear.predict(features[:8])
    tree_pred = tree.predict(features[:8])
    code = str(subset["股票代码"].iloc[0])
    history = subset[subset["股票代码"] == code].sort_values("日期")
    calendar = history["日期"].tolist()
    holding = resolve_holding(calendar, calendar[30])
    bars = pd.DataFrame(
        {
            "date": history["日期"],
            "code": history["股票代码"],
            "open": history["开盘"],
            "high": history["最高"],
            "low": history["最低"],
            "close": history["收盘"],
            "preclose": history["收盘"].shift(1).fillna(history["收盘"]),
            "volume": history["成交量"],
            "tradestatus": np.where(history["成交量"].fillna(0).to_numpy() > 0, "1", "0"),
            "isST": "0",
        }
    )
    account = run_account(
        bars,
        [{**holding, "weights": {code: 1.0}}],
        listings={code: "1990-01-01"},
        trading_days=calendar,
        starting_cash=100000,
        enforce_limits=False,
    )
    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    trades = account["fills"] + account["rejects"]
    status = decide("research", False, [{"model": "linear", "fold": "A", "seed": 42, "excess": 0.01, "stress_excess": 0.01}])
    summary = {
        "mode": "research",
        "strict": False,
        "status": status,
        "feature_map": feature_name,
        "blockers": audit["blockers"],
        "limits_enforced": False,
        "raw_limit_check": _raw_limit_check(probe),
        "steps": {
            "transformer": _transformer_step(sample),
            "linear": {"finite": bool(np.isfinite(linear_pred).all())},
            "tree": {"finite": bool(np.isfinite(tree_pred).all())},
            "equal_weight": {"names": len(equal_weights(sample["codes"]))},
        },
        "smoke": True,
        "replaces_v4": False,
    }
    (output / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(account["daily"]).assign(shares=lambda frame: frame["shares"].map(json.dumps)).to_csv(output / "daily_nav.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "trades.csv", index=False)
    for fold in FOLDS:
        for seed in (42, 123):
            payload = {
                "fold": fold["id"],
                "seed": seed,
                "trained": False,
                "evidence_grade": fold["evidence_grade"],
                "boundaries": fold,
                "trades": trades,
                "daily": account["daily"],
                "reason": "cpu smoke does not train this fold",
            }
            (output / f"fold_{fold['id']}_seed_{seed}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
    return summary


def run_full(panel_path, output_dir) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(panel_path, nrows=5, dtype={"股票代码": str})
    audit = audit_frame(prices, None, None)
    summary = {
        "trained": False,
        "status": "abstain",
        "mode": audit["mode"],
        "strict": audit["strict_ready"],
        "blockers": audit["blockers"],
        "reason": "strict inputs are missing, so the full rolling run does not start",
    }
    (output / "full_blocked.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        result = run_full(args.panel, args.output_dir)
        print(json.dumps({"status": result["status"], "trained": result["trained"]}, ensure_ascii=False))
        raise SystemExit(2)
    summary = run_smoke(args.panel, args.probe, args.output_dir)
    print(json.dumps({"status": summary["status"], "mode": summary["mode"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
