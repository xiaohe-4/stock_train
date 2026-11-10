"""Strict rolling run. It starts only after the point-in-time audit passes."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parents[2] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from research.models import build_model
from research.protocol import LEARNING_RATE, MAX_EPOCHS, PATIENCE, SEEDS, WEIGHT_DECAY, WEIGHTS
from research.reliability.audit import audit_store
from research.reliability.baselines import equal_weights, fit_linear, fit_tree
from research.reliability.dataset import build_samples, load_store, split_fold
from research.reliability.decision import choose
from research.reliability.fetch_pit import END, START
from research.reliability.folds import FOLDS
from research.reliability.holding import rebalance_signals
from research.reliability.ledger import run_account

MODELS = ("equal_weight", "ridge", "tree", "transformer")
STRESS = (1, 2, 3)
DEADLINE_SECONDS = 11 * 60 * 60


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ranked(codes: list[str], scores) -> dict[str, float]:
    order = sorted(range(len(codes)), key=lambda index: float(scores[index]), reverse=True)
    raw = {code: 0.0 for code in codes}
    for rank, index in enumerate(order[: len(WEIGHTS)]):
        raw[codes[index]] = WEIGHTS[rank]
    total = sum(raw.values())
    if total <= 0:
        return equal_weights(codes)
    return {code: value / total for code, value in raw.items()}


def _selected(calendar: list[str], samples: list[dict]) -> list[dict]:
    if not samples:
        return []
    stepped = set(rebalance_signals(calendar, calendar[0], calendar[-1]))
    return [sample for sample in samples if sample["date"] in stepped]


def _schedules(samples: list[dict], weight_of) -> list[dict]:
    rows = []
    for sample in samples:
        rows.append(
            {
                "signal_date": sample["date"],
                "entry_date": sample["entry_date"],
                "exit_date": sample["exit_date"],
                "weights": weight_of(sample),
            }
        )
    return rows


def _mean_return(account: dict, schedules: list[dict], starting: float = 1_000_000.0) -> float:
    navs = {row["date"]: float(row["nav"]) for row in account["daily"]}
    previous = starting
    values = []
    for schedule in schedules:
        nav = navs.get(schedule["exit_date"])
        if nav is None:
            continue
        values.append(nav / previous - 1.0)
        previous = nav
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _scores(kind: str, sample: dict, fitted, device):
    if kind == "equal_weight":
        return None
    if kind in {"ridge", "tree"}:
        return fitted.predict(np.nan_to_num(sample["x"][:, -1, :].astype(np.float64)))
    fitted.eval()
    with torch.no_grad():
        prediction = fitted(torch.from_numpy(sample["x"]).unsqueeze(0).to(device))
    return prediction.squeeze(0).detach().cpu().numpy()


def _weights(kind: str, sample: dict, fitted, device) -> dict[str, float]:
    if kind == "equal_weight":
        return equal_weights(sample["codes"])
    return _ranked(sample["codes"], _scores(kind, sample, fitted, device))


def _train_transformer(train_samples, dev_mean_fn, seed, max_epochs, patience, device, deadline):
    from config import config as model_config
    from train import CombinedRankingLoss

    _seed(seed)
    first = train_samples[0]
    model = build_model("transformer", int(first["x"].shape[-1]), num_stocks=int(first["x"].shape[0]), config=model_config)
    model.to(device)
    loss_fn = CombinedRankingLoss(
        k=model_config.get("topk_mle_k", 3),
        portfolio_weight=model_config.get("portfolio_loss_weight", 0.4),
        temperature=model_config.get("portfolio_temperature", 0.35),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_excess = -1e9
    waited = 0
    epochs = 0
    loss_finite = False
    for _epoch in range(max_epochs):
        if time.monotonic() >= deadline:
            break
        model.train()
        for sample in train_samples:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(sample["x"]).unsqueeze(0).to(device))
            target = torch.from_numpy(sample["train_target"]).unsqueeze(0).to(device)
            loss = loss_fn(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_finite = bool(torch.isfinite(loss.detach()))
        epochs += 1
        excess = dev_mean_fn(model)
        if excess > best_excess:
            best_excess = excess
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs": epochs, "loss_finite": loss_finite, "best_dev_excess": best_excess}


def _write_book(directory: Path, account: dict, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(account["daily"]).to_csv(directory / "daily_nav.csv", index=False)
    fills = [{**row, "reason": ""} for row in account["fills"]]
    trades = fills + list(account["rejects"])
    pd.DataFrame(trades).to_csv(directory / "trades.csv", index=False)
    pd.DataFrame(account["rejects"], columns=["date", "side", "code", "reason"]).to_csv(directory / "rejects.csv", index=False)
    (directory / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _fail(output: Path, audit: dict, status: str, reason: str, strict: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "trained": False,
        "strict": strict,
        "complete": False,
        "status": status,
        "reason": reason,
        "blockers": audit.get("blockers", []),
        "replaces_v4": False,
        "enable_memory": False,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_strict(
    data_dir,
    output_dir,
    folds=None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seeds=SEEDS,
    min_names: int = 10,
    expected_days: int = 689,
    constituents_per_day: int = 300,
    start: str = START,
    end: str = END,
    deadline_seconds: int = DEADLINE_SECONDS,
    enable_memory: bool = False,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_store(data_dir, expected_days=expected_days, constituents_per_day=constituents_per_day, start=start, end=end)
    (output / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["strict_ready"]:
        return _fail(output, audit, "fail", "audit_failed")
    deadline = time.monotonic() + deadline_seconds
    if time.monotonic() >= deadline:
        return _fail(output, audit, "abstain", "stopped_before_12h", strict=True)
    prices, membership, listings, calendar = load_store(data_dir)
    listing_dates = {code: row["ipoDate"] for code, row in listings.items()}
    samples = build_samples(prices, membership, calendar, min_names=min_names, cache_dir=output / "feature_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_list = list(FOLDS if folds is None else folds)
    dev_reports = []
    test_reports = []
    fitted = {}
    for fold in fold_list:
        if time.monotonic() >= deadline:
            return _fail(output, audit, "abstain", "stopped_before_12h", strict=True)
        parts = split_fold(samples, fold)
        if not parts["train"]:
            raise RuntimeError(f"fold {fold['id']} has no training samples")
        dev_samples = _selected(calendar, parts["dev"])
        test_samples = _selected(calendar, parts["test"])
        features = np.nan_to_num(np.concatenate([sample["x"][:, -1, :] for sample in parts["train"]]).astype(np.float64))
        labels = np.nan_to_num(np.concatenate([sample["train_target"] for sample in parts["train"]]).astype(np.float64))
        fitted[("ridge", fold["id"])] = fit_linear(features, labels)
        for seed in seeds:
            if time.monotonic() >= deadline:
                return _fail(output, audit, "abstain", "stopped_before_12h", strict=True)
            _seed(int(seed))
            fitted[("tree", fold["id"], seed)] = fit_tree(features, labels, seed=int(seed))

            def dev_mean(model, fold=fold, dev_samples=dev_samples):
                schedules = _schedules(dev_samples, lambda sample, model=model: _weights("transformer", sample, model, device))
                if not schedules:
                    return 0.0
                account = run_account(prices, schedules, listing_dates, calendar, multiplier=1, enforce_limits=True)
                base = _schedules(dev_samples, lambda sample: equal_weights(sample["codes"]))
                baseline = run_account(prices, base, listing_dates, calendar, multiplier=1, enforce_limits=True)
                return _mean_return(account, schedules) - _mean_return(baseline, base)

            transformer, train_meta = _train_transformer(parts["train"], dev_mean, int(seed), max_epochs, patience, device, deadline)
            fitted[("transformer", fold["id"], seed)] = (transformer, train_meta)
            predictors = {
                "equal_weight": None,
                "ridge": fitted[("ridge", fold["id"])],
                "tree": fitted[("tree", fold["id"], seed)],
                "transformer": transformer,
            }
            for kind, model in predictors.items():
                for multiplier in STRESS:
                    for split_name, split_samples in (("dev", dev_samples), ("test", test_samples)):
                        schedules = _schedules(split_samples, lambda sample, kind=kind, model=model: _weights(kind, sample, model, device))
                        account = run_account(prices, schedules, listing_dates, calendar, multiplier=multiplier, enforce_limits=True) if schedules else {"daily": [], "fills": [], "rejects": []}
                        directory = output / f"fold_{fold['id']}_{kind}_seed_{seed}_x{multiplier}"
                        if split_name == "dev":
                            _write_book(directory, account, {"fold": fold["id"], "model": kind, "seed": seed, "multiplier": multiplier, "split": "dev", "train": train_meta if kind == "transformer" else {"epochs": 0}})
                        else:
                            pd.DataFrame(account["daily"]).to_csv(directory / "test_daily_nav.csv", index=False)
                            fills = [{**row, "reason": ""} for row in account["fills"]]
                            pd.DataFrame(fills + list(account["rejects"])).to_csv(directory / "test_trades.csv", index=False)
                        if multiplier in {1, 2} and schedules:
                            baseline = _schedules(split_samples, lambda sample: equal_weights(sample["codes"]))
                            base_account = run_account(prices, baseline, listing_dates, calendar, multiplier=multiplier, enforce_limits=True)
                            excess = _mean_return(account, schedules) - _mean_return(base_account, baseline)
                            bucket = dev_reports if split_name == "dev" else test_reports
                            if multiplier == 1:
                                bucket.append({"model": kind, "fold": fold["id"], "seed": seed, "excess": excess, "stress_excess": None, "split": split_name})
                            else:
                                for row in bucket:
                                    if row["model"] == kind and row["fold"] == fold["id"] and row["seed"] == seed and row["split"] == split_name:
                                        row["stress_excess"] = excess
    for row in dev_reports + test_reports:
        if row["stress_excess"] is None:
            row["stress_excess"] = 0.0
    decision = choose("strict", True, dev_reports)
    result = {
        "trained": True,
        "strict": True,
        "complete": True,
        "status": decision["status"],
        "selected_model": decision["model"],
        "models": list(MODELS),
        "enable_memory": enable_memory,
        "feature_map": "feature_cloums_map",
        "decision_uses": "dev_only",
        "replaces_v4": False,
        "holding_id": "open_t_plus_1_to_open_t_plus_5",
        "holding_steps": 4,
        "price_basis": "unadjusted",
        "max_epochs": max_epochs,
        "patience": patience,
        "seeds": list(seeds),
        "cost_stress": list(STRESS),
        "evidence_grade": "already_observed" if all(fold.get("evidence_grade") == "already_observed" for fold in fold_list) else "smoke",
        "dev_reports": dev_reports,
        "test_reports": test_reports,
        "limitation": "标签是不复权开盘到开盘收益。持有期内的分红除权没有再用后复权修正。证据等级不是独立最终测试，也不替换 v4。",
    }
    if enable_memory:
        result["memory"] = "requested but this run does not add a new memory search"
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit = audit_store(args.data_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"strict_ready": audit["strict_ready"], "blockers": audit["blockers"]}, ensure_ascii=False), flush=True)
        raise SystemExit(0 if audit["strict_ready"] else 2)
    result = run_strict(args.data_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "strict": result["strict"], "trained": result["trained"], "complete": result["complete"]}, ensure_ascii=False), flush=True)
    if result["status"] == "fail" or result["strict"] is False:
        raise SystemExit(2)
    if not result["complete"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
