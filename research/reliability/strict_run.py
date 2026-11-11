"""Strict rolling run. It starts only after the point-in-time audit passes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parents[2] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from research.models import build_model
from research.protocol import LEARNING_RATE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, WEIGHTS
from research.reliability.audit import audit_store
from research.reliability.baselines import equal_weights, fit_linear, fit_tree
from research.reliability.dataset import build_samples, load_store, split_fold
from research.reliability.decision import choose
from research.reliability.fetch_pit import END, START
from research.reliability.folds import FOLDS
from research.reliability.holding import rebalance_signals
from research.reliability.ledger import run_account
from research.reliability.preprocess import prepare_fold

MODELS = ("equal_weight", "ridge", "tree", "transformer")
STRESS = (1, 2, 3)
RELIABILITY_SEEDS = (42, 123, 2024)
DEADLINE_SECONDS = 11 * 60 * 60
INNER_VAL_FRACTION = 0.2
EARLY_STOP = "inner_train_top5_label"
SHARPE_FORMULA = (
    "annualized Sharpe = mean(r) / sample_std(r) * sqrt(252), "
    "where r_t = nav_t / nav_{t-1} - 1, the series starts at the account starting cash, "
    "and daily NAV includes cash days. sample_std uses ddof=1. "
    "This is not the 5-day period Sharpe."
)
MAX_DRAWDOWN_DEFINITION = "non-negative (peak - nav) / peak on the daily NAV, with the starting cash as the initial peak"


def _seed(seed: int) -> None:
    """Seed every generator the training step can read, including after CUDA starts."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ranked(codes: list[str], scores) -> dict[str, float]:
    values = [float(scores[index]) for index in range(len(codes))]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("non-finite scores are not ranked")
    order = sorted(range(len(codes)), key=lambda index: values[index], reverse=True)
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


def _count_nonfinite_scores(kind: str, samples: list[dict], fitted, device) -> int:
    if kind == "equal_weight" or fitted is None or not samples:
        return 0
    total = 0
    for sample in samples:
        scores = np.asarray(_scores(kind, sample, fitted, device))
        total += int(scores.size - np.isfinite(scores).sum())
    return total


def _last_rows(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([np.asarray(sample["x"])[:, -1, :] for sample in samples]).astype(np.float64)
    labels = np.concatenate([np.asarray(sample["train_target"]) for sample in samples]).astype(np.float64)
    return features, labels


def _inner_split(train_samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out the last 20% of training signal dates. Fold dev and test are not used."""
    dates = sorted({sample["date"] for sample in train_samples})
    if len(dates) < 2:
        return list(train_samples), list(train_samples)
    cut = int(len(dates) * (1.0 - INNER_VAL_FRACTION))
    cut = min(max(cut, 1), len(dates) - 1)
    fit_dates = set(dates[:cut])
    val_dates = set(dates[cut:])
    fit = [sample for sample in train_samples if sample["date"] in fit_dates]
    val = [sample for sample in train_samples if sample["date"] in val_dates]
    return fit, val


def _inner_label(model, val_samples: list[dict], device) -> float:
    if not val_samples:
        return float("-inf")
    values = []
    for sample in val_samples:
        weights = _weights("transformer", sample, model, device)
        labels = sample["train_target"]
        values.append(sum(weights[code] * float(labels[index]) for index, code in enumerate(sample["codes"])))
    return float(sum(values) / len(values))


def _train_transformer(train_samples, seed, max_epochs, patience, device, deadline):
    from config import config as model_config
    from train import CombinedRankingLoss

    fit_samples, val_samples = _inner_split(train_samples)
    _seed(seed)
    first = fit_samples[0]
    model = build_model("transformer", int(first["x"].shape[-1]), num_stocks=int(first["x"].shape[0]), config=model_config)
    model.to(device)
    _seed(seed)
    loss_fn = CombinedRankingLoss(
        k=model_config.get("topk_mle_k", 3),
        portfolio_weight=model_config.get("portfolio_loss_weight", 0.4),
        temperature=model_config.get("portfolio_temperature", 0.35),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_label = -1e9
    waited = 0
    epochs = 0
    loss_finite = False
    for _epoch in range(max_epochs):
        if time.monotonic() >= deadline:
            break
        model.train()
        finite_epoch = True
        for sample in fit_samples:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(sample["x"]).unsqueeze(0).to(device))
            target = torch.from_numpy(sample["train_target"]).unsqueeze(0).to(device)
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss.detach()):
                finite_epoch = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_finite = True
        epochs += 1
        if not finite_epoch:
            waited += 1
            if waited >= patience:
                break
            continue
        try:
            label = _inner_label(model, val_samples, device)
        except RuntimeError:
            label = float("-inf")
        if label > best_label:
            best_label = label
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "epochs": epochs,
        "loss_finite": loss_finite,
        "best_inner_label": None if best_label <= -1e8 else best_label,
        "early_stop": EARLY_STOP,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "inner_val_dates": sorted({sample["date"] for sample in val_samples}),
        "fit_dates": sorted({sample["date"] for sample in fit_samples}),
        "uses_fold_dev": False,
        "uses_fold_test": False,
    }


def nav_metrics(daily: list[dict], starting: float = 1_000_000.0) -> dict:
    """Report-only path statistics. Nothing here is an adoption gate."""
    empty = {
        "cumulative_return": None,
        "max_drawdown": None,
        "sharpe_annualized": None,
        "average_turnover": None,
        "total_fees": None,
    }
    if not daily:
        return empty
    navs = [float(row["nav"]) for row in daily]
    cumulative = navs[-1] / starting - 1.0 if starting else None
    peak = starting
    max_dd = 0.0
    for nav in navs:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, (peak - nav) / peak)
    series = [starting, *navs]
    returns = []
    for previous, nav in zip(series, series[1:]):
        if previous == 0 or not math.isfinite(previous) or not math.isfinite(nav):
            continue
        returns.append(nav / previous - 1.0)
    sharpe = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std > 0 and math.isfinite(std):
            sharpe = mean / std * math.sqrt(252)
    turnover = [float(row.get("turnover") or 0.0) for row in daily]
    fees = [float(row.get("fees") or 0.0) for row in daily]
    return {
        "cumulative_return": cumulative,
        "max_drawdown": max_dd,
        "sharpe_annualized": sharpe,
        "average_turnover": sum(turnover) / len(turnover),
        "total_fees": sum(fees),
    }


def recorded_source(data_dir, explicit, explicit_path) -> tuple[str, str]:
    source = explicit
    if not source:
        manifest_path = Path(data_dir) / "manifest.json"
        if manifest_path.is_file():
            try:
                source = json.loads(manifest_path.read_text(encoding="utf-8")).get("data_source")
            except json.JSONDecodeError:
                source = None
    return source or "unspecified", str(explicit_path or data_dir)


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
        "data_source": audit.get("data_source", "unspecified"),
        "data_source_path": audit.get("data_source_path"),
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_strict(
    data_dir,
    output_dir,
    folds=None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seeds=RELIABILITY_SEEDS,
    min_names: int = 10,
    expected_days: int = 689,
    constituents_per_day: int = 300,
    start: str = START,
    end: str = END,
    deadline_seconds: int = DEADLINE_SECONDS,
    enable_memory: bool = False,
    data_source: str | None = None,
    data_source_path: str | None = None,
    reuse_rejection: str | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_store(data_dir, expected_days=expected_days, constituents_per_day=constituents_per_day, start=start, end=end)
    source, source_path = recorded_source(data_dir, data_source, data_source_path)
    audit["data_source"] = source
    audit["data_source_path"] = source_path
    if reuse_rejection:
        audit["reuse_rejection"] = reuse_rejection
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
    feature_diagnostics = []
    fitted = {}
    for fold in fold_list:
        if time.monotonic() >= deadline:
            return _fail(output, audit, "abstain", "stopped_before_12h", strict=True)
        parts = split_fold(samples, fold)
        if not parts["train"]:
            raise RuntimeError(f"fold {fold['id']} has no training samples")
        raw_dev = _selected(calendar, parts["dev"])
        raw_test = _selected(calendar, parts["test"])
        cleaned_train, cleaned_dev, cleaned_test, feature_report = prepare_fold(parts["train"], parts["dev"], parts["test"], min_names)
        dev_samples = _selected(calendar, cleaned_dev)
        test_samples = _selected(calendar, cleaned_test)
        features, labels = _last_rows(cleaned_train)
        fitted[("ridge", fold["id"])] = fit_linear(features, labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_features, raw_labels = _last_rows(parts["train"])
            raw_ridge = fit_linear(np.nan_to_num(raw_features), np.nan_to_num(raw_labels))
        ridge_before = _count_nonfinite_scores("ridge", raw_dev, raw_ridge, device) + _count_nonfinite_scores("ridge", raw_test, raw_ridge, device)
        ridge_after = _count_nonfinite_scores("ridge", dev_samples, fitted[("ridge", fold["id"])], device) + _count_nonfinite_scores("ridge", test_samples, fitted[("ridge", fold["id"])], device)
        fold_diag = {
            "fold": fold["id"],
            **feature_report,
            "score_count_definition": (
                "nonfinite_scores_before counts Ridge and the tree fit on the uncleaned training rows and scored on the uncleaned dev/test windows. "
                "For the transformer it is one eval forward, under that seed, of an untrained network on the uncleaned window. "
                "nonfinite_scores_after counts the production model on the cleaned window. Non-finite production scores still fail closed."
            ),
            "models": {
                "equal_weight": {"nonfinite_scores_before": 0, "nonfinite_scores_after": 0},
                "ridge": {"nonfinite_scores_before": ridge_before, "nonfinite_scores_after": ridge_after},
            },
        }
        for seed in seeds:
            if time.monotonic() >= deadline:
                return _fail(output, audit, "abstain", "stopped_before_12h", strict=True)
            _seed(int(seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_tree = fit_tree(np.nan_to_num(raw_features), np.nan_to_num(raw_labels), seed=int(seed))
            tree_before = _count_nonfinite_scores("tree", raw_dev, raw_tree, device) + _count_nonfinite_scores("tree", raw_test, raw_tree, device)
            fitted[("tree", fold["id"], seed)] = fit_tree(features, labels, seed=int(seed))
            tree_after = _count_nonfinite_scores("tree", dev_samples, fitted[("tree", fold["id"], seed)], device) + _count_nonfinite_scores("tree", test_samples, fitted[("tree", fold["id"], seed)], device)
            _seed(int(seed))
            from config import config as model_config

            probe = build_model(
                "transformer",
                int(cleaned_train[0]["x"].shape[-1]),
                num_stocks=int(cleaned_train[0]["x"].shape[0]),
                config=model_config,
            )
            probe.to(device)
            transformer_before = _count_nonfinite_scores("transformer", raw_dev, probe, device) + _count_nonfinite_scores("transformer", raw_test, probe, device)
            del probe
            transformer, train_meta = _train_transformer(cleaned_train, int(seed), max_epochs, patience, device, deadline)
            fitted[("transformer", fold["id"], seed)] = (transformer, train_meta)
            transformer_after = _count_nonfinite_scores("transformer", dev_samples, transformer, device) + _count_nonfinite_scores("transformer", test_samples, transformer, device)
            fold_diag["models"]["tree"] = fold_diag["models"].get("tree", {})
            fold_diag["models"]["tree"][str(seed)] = {"nonfinite_scores_before": tree_before, "nonfinite_scores_after": tree_after}
            fold_diag["models"]["transformer"] = fold_diag["models"].get("transformer", {})
            fold_diag["models"]["transformer"][str(seed)] = {"nonfinite_scores_before": transformer_before, "nonfinite_scores_after": transformer_after}
            (output / f"fold_{fold['id']}_diagnostics.json").write_text(json.dumps(fold_diag, ensure_ascii=False, indent=2), encoding="utf-8")
            score_counts = {
                "equal_weight": fold_diag["models"]["equal_weight"],
                "ridge": fold_diag["models"]["ridge"],
                "tree": fold_diag["models"]["tree"][str(seed)],
                "transformer": fold_diag["models"]["transformer"][str(seed)],
            }
            predictors = {
                "equal_weight": None,
                "ridge": fitted[("ridge", fold["id"])],
                "tree": fitted[("tree", fold["id"], seed)],
                "transformer": transformer,
            }
            for kind, model in predictors.items():
                for split_name, split_samples in (("dev", dev_samples), ("test", test_samples)):
                    books = {}
                    for multiplier in STRESS:
                        schedules = _schedules(split_samples, lambda sample, kind=kind, model=model: _weights(kind, sample, model, device))
                        account = run_account(prices, schedules, listing_dates, calendar, multiplier=multiplier, enforce_limits=True) if schedules else {"daily": [], "fills": [], "rejects": []}
                        books[multiplier] = (account, schedules)
                        directory = output / f"fold_{fold['id']}_{kind}_seed_{seed}_x{multiplier}"
                        if split_name == "dev":
                            _write_book(directory, account, {"fold": fold["id"], "model": kind, "seed": seed, "multiplier": multiplier, "split": "dev", "train": train_meta if kind == "transformer" else {"epochs": 0}})
                        else:
                            directory.mkdir(parents=True, exist_ok=True)
                            pd.DataFrame(account["daily"]).to_csv(directory / "test_daily_nav.csv", index=False)
                            fills = [{**row, "reason": ""} for row in account["fills"]]
                            pd.DataFrame(fills + list(account["rejects"])).to_csv(directory / "test_trades.csv", index=False)
                    schedules = books[1][1]
                    if not schedules:
                        continue
                    baseline = _schedules(split_samples, lambda sample: equal_weights(sample["codes"]))
                    excesses = {}
                    base_accounts = {}
                    for multiplier in STRESS:
                        base_account = run_account(prices, baseline, listing_dates, calendar, multiplier=multiplier, enforce_limits=True)
                        base_accounts[multiplier] = base_account
                        excesses[multiplier] = _mean_return(books[multiplier][0], schedules) - _mean_return(base_account, baseline)
                    metrics = nav_metrics(books[1][0]["daily"])
                    benchmark = nav_metrics(base_accounts[1]["daily"])
                    bucket = dev_reports if split_name == "dev" else test_reports
                    bucket.append(
                        {
                            "model": kind,
                            "fold": fold["id"],
                            "seed": seed,
                            "excess": excesses[1],
                            "stress_excess": excesses[2],
                            "stress_excess_3x": excesses[3],
                            "cumulative_return": metrics["cumulative_return"],
                            "benchmark_cumulative_return": benchmark["cumulative_return"],
                            "max_drawdown": metrics["max_drawdown"],
                            "sharpe_annualized": metrics["sharpe_annualized"],
                            "average_turnover": metrics["average_turnover"],
                            "total_fees": metrics["total_fees"],
                            "nonfinite_features_before": feature_report["nonfinite_features_before"],
                            "nonfinite_features_after": feature_report["nonfinite_features_after"],
                            "max_abs_feature_before": feature_report["max_abs_feature_before"],
                            "max_abs_feature_after": feature_report["max_abs_feature_after"],
                            "nonfinite_scores_before": score_counts[kind]["nonfinite_scores_before"],
                            "nonfinite_scores_after": score_counts[kind]["nonfinite_scores_after"],
                            "split": split_name,
                        }
                    )
        feature_diagnostics.append(fold_diag)
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
        "sharpe_formula": SHARPE_FORMULA,
        "max_drawdown_definition": MAX_DRAWDOWN_DEFINITION,
        "nav_metrics_gated": False,
        "early_stop": EARLY_STOP,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "data_source": source,
        "data_source_path": source_path,
        "reuse_rejection": reuse_rejection,
        "evidence_grade": "already_observed" if all(fold.get("evidence_grade") == "already_observed" for fold in fold_list) else "smoke",
        "feature_diagnostics": feature_diagnostics,
        "dev_reports": dev_reports,
        "test_reports": test_reports,
        "limitation": "标签是不复权开盘到开盘收益。持有期内的分红除权没有再用后复权修正。特征在训练折上估计均值和标准差，标准差下限 1e-6，标准化后再截断到 ±10，开发验证和测试只用这组统计量。Transformer 早停只用训练集内部最后 20% 信号日的 top-5 标签均值，不看折的开发验证或测试，也没有按这两段调过学习率、dropout 或深度。这些日期已经被看过两次，证据等级保持 already_observed（本折标明 smoke 时除外）。这不是独立最终测试，也不替换 v4。",
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
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--data-source-path", default=None)
    parser.add_argument("--reuse-rejection", default=None)
    args = parser.parse_args()
    if args.audit_only:
        audit = audit_store(args.data_dir)
        source, source_path = recorded_source(args.data_dir, args.data_source, args.data_source_path)
        audit["data_source"] = source
        audit["data_source_path"] = source_path
        if args.reuse_rejection:
            audit["reuse_rejection"] = args.reuse_rejection
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"strict_ready": audit["strict_ready"], "blockers": audit["blockers"], "data_source": source}, ensure_ascii=False), flush=True)
        raise SystemExit(0 if audit["strict_ready"] else 2)
    result = run_strict(
        args.data_dir,
        args.output_dir,
        data_source=args.data_source,
        data_source_path=args.data_source_path,
        reuse_rejection=args.reuse_rejection,
    )
    print(json.dumps({"status": result["status"], "strict": result["strict"], "trained": result["trained"], "complete": result["complete"]}, ensure_ascii=False), flush=True)
    if result["status"] == "fail" or result["strict"] is False:
        raise SystemExit(2)
    if not result["complete"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
