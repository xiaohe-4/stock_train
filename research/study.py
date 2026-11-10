"""From-scratch three-arm study. This does not load or overwrite v4 checkpoints."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parents[1] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from research.backtest import (
    apply_runtime_budget,
    day_portfolio,
    decide,
    entry_blocked,
    mean_excess,
    period_nav,
    top_k_jaccard,
)
from research.models import benchmark_step, build_model, count_parameters
from research.protocol import (
    ADOPT_USES,
    ASOF,
    APPENDIX_END,
    APPENDIX_START,
    DEV_VAL_END,
    DEV_VAL_START,
    HFQ_LIMIT_NOTE,
    LEADER_LOOKBACK,
    LEADER_RATIO,
    MAX_EPOCHS,
    NAV_RULE,
    PANEL_ROWS,
    PANEL_SHA256,
    PANEL_STOCKS,
    PATIENCE,
    SEEDS,
    SELECTION_RULE,
    TRAIN_WINDOW_END,
)
from research.windows import window_end_dates

ARMS = ("transformer", "lstm", "memory")


def assign_split(date: str) -> str:
    if date <= TRAIN_WINDOW_END:
        return "train"
    if DEV_VAL_START <= date <= DEV_VAL_END:
        return "dev"
    if APPENDIX_START <= date <= APPENDIX_END:
        return "appendix"
    return "ignore"


def panel_date_errors(dates: list[str]) -> list[str]:
    errors = []
    if not dates:
        return ["panel has no dates"]
    latest = max(dates)
    if latest != ASOF:
        errors.append(f"max date is {latest}, research as-of is fixed at {ASOF}")
    later = sorted(date for date in dates if date > ASOF)
    if later:
        errors.append(f"rows after {ASOF}: {later[:5]}")
    return errors


def smoke_three_arms() -> dict:
    from config import config as base

    report = {}
    for arm in ARMS:
        full = build_model(arm, 196, 8, base)
        tiny = dict(base)
        tiny.update(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            sequence_length=6,
            cross_stock_layers=1,
        )
        model = build_model(arm, 5, 3, tiny)
        stats = benchmark_step(model, torch.randn(2, 3, tiny["sequence_length"], 5))
        report[arm] = {
            "parameters": count_parameters(full),
            "seconds": stats["seconds"],
            "grad_norm": stats["grad_norm"],
        }
    return report


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trading_shift(calendar: list[str], date: str, step: int) -> str | None:
    try:
        index = calendar.index(date)
    except ValueError:
        return None
    moved = index + step
    if moved < 0 or moved >= len(calendar):
        return None
    return calendar[moved]


def _tradable(market: dict, code: str, entry: str | None, exit_date: str | None) -> bool:
    if entry is None or exit_date is None:
        return False
    entry_row = market.get((code, entry))
    exit_row = market.get((code, exit_date))
    if entry_row is None or exit_row is None:
        return False
    return not entry_blocked(entry_row["volume"]) and not entry_blocked(exit_row["volume"])


def score_day(sample: dict, scores: np.ndarray, market: dict, calendar: list[str], previous: dict) -> dict:
    entry = _trading_shift(calendar, sample["date"], 1)
    exit_date = _trading_shift(calendar, sample["date"], 5)
    tradable = [
        _tradable(market, code, entry, exit_date)
        for code in sample["codes"]
    ]
    priced = day_portfolio(sample["codes"], scores, sample["raw"], tradable, previous)
    tradable_raw = [float(sample["raw"][i]) for i, flag in enumerate(tradable) if flag]
    ew = float(sum(tradable_raw) / len(tradable_raw)) if tradable_raw else 0.0
    return {
        "date": sample["date"],
        "split": sample["split"],
        "port_return": priced["port_return"],
        "ew_return": ew,
        "turnover": priced["turnover"],
        "weights": priced["weights"],
        "entry": entry,
        "exit": exit_date,
        "ranked": [sample["codes"][i] for i in np.argsort(-scores)[:5]],
    }


def evaluate_split(model, samples: list[dict], market, calendar, split: str) -> list[dict]:
    model.eval()
    device = next(model.parameters()).device
    rows = []
    previous: dict[str, float] = {}
    chosen = [sample for sample in samples if sample["split"] == split]
    with torch.no_grad():
        for sample in chosen:
            tensor = torch.from_numpy(sample["x"]).unsqueeze(0).to(device)
            scores = model(tensor).squeeze(0).detach().cpu().numpy()
            row = score_day(sample, scores, market, calendar, previous)
            previous = row["weights"]
            rows.append(row)
    return rows


def _fit_arrays(samples: list[dict]) -> None:
    train = [sample for sample in samples if sample["split"] == "train"]
    if not train:
        raise RuntimeError("no training windows")
    raw = np.concatenate([sample["raw"] for sample in train])
    low, high = np.quantile(raw, [0.01, 0.99])
    flat = np.concatenate([sample["x"].reshape(-1, sample["x"].shape[-1]) for sample in train], axis=0)
    median = np.nanmedian(flat, axis=0)
    flat = np.where(np.isnan(flat), median, flat)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    for sample in samples:
        values = np.where(np.isnan(sample["x"]), median, sample["x"])
        sample["x"] = ((values - mean) / std).astype(np.float32)
        clipped = np.clip(sample["raw"], low, high).astype(np.float32)
        sample["train_target"] = (clipped - float(clipped.mean())).astype(np.float32)


def train_one(arm: str, samples: list[dict], market, calendar, seed: int) -> dict:
    from config import config as model_config
    from train import CombinedRankingLoss

    if list(model_config["ensemble_seeds"]) != list(SEEDS):
        raise RuntimeError("repository seeds differ from the frozen research protocol")
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = samples[0]["x"].shape[-1]
    model = build_model(arm, input_dim, num_stocks=len({code for sample in samples for code in sample["codes"]}), config=model_config)
    model.to(device)
    parameters = count_parameters(model)
    loss_fn = CombinedRankingLoss(
        k=model_config.get("topk_mle_k", 3),
        portfolio_weight=model_config.get("portfolio_loss_weight", 0.4),
        temperature=model_config.get("portfolio_temperature", 0.35),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config.get("weight_decay", 2e-3),
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=model_config.get("warmup_epochs", 5)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model_config["num_epochs"] - model_config.get("warmup_epochs", 5),
        eta_min=model_config["learning_rate"] * 0.05,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[model_config.get("warmup_epochs", 5)]
    )
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    best_score = -1e18
    best_state = None
    patience = 0
    epoch_seconds = []
    grad_norms = []
    batch_size = int(model_config["batch_size"])
    for epoch in range(MAX_EPOCHS):
        started = time.perf_counter()
        model.train()
        order = list(range(len(train_samples)))
        random.shuffle(order)
        grad_sum = 0.0
        grad_steps = 0
        for start in range(0, len(order), batch_size):
            batch = [train_samples[index] for index in order[start : start + batch_size]]
            width = max(sample["x"].shape[0] for sample in batch)
            length = batch[0]["x"].shape[1]
            features = batch[0]["x"].shape[2]
            inputs = np.zeros((len(batch), width, length, features), np.float32)
            targets = np.zeros((len(batch), width), np.float32)
            mask = np.zeros((len(batch), width), np.float32)
            for row, sample in enumerate(batch):
                count = sample["x"].shape[0]
                inputs[row, :count] = sample["x"]
                targets[row, :count] = sample["train_target"]
                mask[row, :count] = 1.0
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(inputs).to(device), torch.from_numpy(mask).to(device))
            target = torch.from_numpy(targets).to(device)
            valid = torch.from_numpy(mask).to(device) > 0
            batch_loss = None
            for row in range(len(batch)):
                picked = valid[row]
                if int(picked.sum()) < 2:
                    continue
                loss = loss_fn(prediction[row, picked].unsqueeze(0), target[row, picked].unsqueeze(0))
                batch_loss = loss if batch_loss is None else batch_loss + loss
            if batch_loss is None:
                continue
            batch_loss = batch_loss / len(batch)
            if not torch.isfinite(batch_loss.detach()):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.get("max_grad_norm", 1.0))
            optimizer.step()
            grad_sum += float(grad_norm)
            grad_steps += 1
        scheduler.step()
        dev_rows = evaluate_split(model, samples, market, calendar, "dev")
        score = mean_excess(dev_rows, "dev")
        epoch_seconds.append(time.perf_counter() - started)
        grad_norms.append(grad_sum / max(grad_steps, 1))
        print(
            f"{arm} seed {seed} epoch {epoch + 1} dev_excess {score:.6f} grad {grad_norms[-1]:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError(f"{arm} seed {seed} produced no checkpoint")
    model.load_state_dict(best_state)
    dev_rows = evaluate_split(model, samples, market, calendar, "dev")
    appendix_rows = evaluate_split(model, samples, market, calendar, "appendix")
    return {
        "arm": arm,
        "seed": seed,
        "parameters": parameters,
        "best_dev_excess": mean_excess(dev_rows, "dev"),
        "epoch_seconds": epoch_seconds,
        "grad_norms": grad_norms,
        "dev_rows": dev_rows,
        "appendix_rows": appendix_rows,
        "dev_nav": period_nav(dev_rows, "dev"),
        "appendix_nav": period_nav(appendix_rows, "appendix"),
    }


def build_samples(panel: pd.DataFrame) -> tuple[list[dict], dict, list[str]]:
    from train import feature_cloums_map, feature_engineer_func_map

    frame = panel.copy()
    frame["股票代码"] = frame["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d")
    errors = panel_date_errors(frame["日期"].tolist())
    if errors:
        raise RuntimeError("; ".join(errors))
    calendar = sorted(frame["日期"].unique())
    market = {
        (row["股票代码"], row["日期"]): {"volume": row["成交量"], "open": row["开盘"]}
        for row in frame[["股票代码", "日期", "成交量", "开盘"]].to_dict("records")
    }
    history = frame[frame["日期"] <= TRAIN_WINDOW_END]
    recent = history.sort_values(["股票代码", "日期"]).groupby("股票代码").tail(LEADER_LOOKBACK)
    average = recent.groupby("股票代码")["成交额"].mean().sort_values(ascending=False)
    keep = set(average.head(max(1, int(len(average) * LEADER_RATIO))).index)
    engineer = feature_engineer_func_map["158+39"]
    columns = feature_cloums_map["158+39"]
    samples_by_date: dict[str, list] = {}
    for code, group in frame[frame["股票代码"].isin(keep)].groupby("股票代码", sort=False):
        group = group.sort_values("日期")
        label = (group["开盘"].shift(-5) - group["开盘"].shift(-1)) / (group["开盘"].shift(-1) + 1e-12)
        features = engineer(group)
        features["label"] = label.to_numpy()
        features["日期"] = group["日期"].to_numpy()
        missing = [name for name in columns if name not in features.columns]
        if missing:
            raise RuntimeError(f"missing features: {missing[:8]}")
        matrix = features[columns].to_numpy(dtype=np.float32)
        labels = features["label"].to_numpy(dtype=np.float64)
        dates = features["日期"].astype(str).tolist()
        ends = set(window_end_dates(features, columns, 30))
        for index, date in enumerate(dates):
            if date not in ends:
                continue
            split = assign_split(date)
            if split == "ignore":
                continue
            window = matrix[index - 29 : index + 1]
            samples_by_date.setdefault(date, []).append((code, window, float(labels[index])))
    samples = []
    for date in sorted(samples_by_date):
        rows = samples_by_date[date]
        if len(rows) < 10:
            continue
        samples.append(
            {
                "date": date,
                "split": assign_split(date),
                "codes": [row[0] for row in rows],
                "x": np.stack([row[1] for row in rows]).astype(np.float32),
                "raw": np.asarray([row[2] for row in rows], dtype=np.float32),
            }
        )
    if any(sample["split"] == "train" and sample["date"] > TRAIN_WINDOW_END for sample in samples):
        raise RuntimeError("training window crossed the frozen cutoff")
    _fit_arrays(samples)
    return samples, market, calendar


def run_study(panel: pd.DataFrame) -> dict:
    samples, market, calendar = build_samples(panel)
    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            runs.append(train_one(arm, samples, market, calendar, seed))
    dev_means = {}
    epoch_seconds = {}
    overlap = {}
    for arm in ARMS:
        arm_runs = [run for run in runs if run["arm"] == arm]
        dev_means[arm] = float(np.mean([run["best_dev_excess"] for run in arm_runs]))
        epoch_seconds[arm] = float(np.median([float(np.median(run["epoch_seconds"])) for run in arm_runs]))
    status = apply_runtime_budget(decide(dev_means), epoch_seconds)
    seed_rows = [run for run in runs if run["seed"] == SEEDS[0]]
    by_arm = {run["arm"]: run for run in seed_rows}
    for left, right in (("transformer", "lstm"), ("transformer", "memory"), ("lstm", "memory")):
        scores = []
        left_rows = {row["date"]: row["ranked"] for row in by_arm[left]["dev_rows"]}
        for row in by_arm[right]["dev_rows"]:
            if row["date"] in left_rows:
                scores.append(top_k_jaccard(left_rows[row["date"]], row["ranked"]))
        overlap[f"{left}_vs_{right}"] = float(np.mean(scores)) if scores else None
    return {
        "status": status,
        "selection_rule": SELECTION_RULE,
        "nav_rule": NAV_RULE,
        "adopt_uses": ADOPT_USES,
        "limitation": HFQ_LIMIT_NOTE,
        "dev_means": dev_means,
        "median_epoch_seconds": epoch_seconds,
        "overlap_jaccard_seed42": overlap,
        "runs": [
            {
                "arm": run["arm"],
                "seed": run["seed"],
                "parameters": run["parameters"],
                "best_dev_excess": run["best_dev_excess"],
                "epochs": len(run["epoch_seconds"]),
                "mean_grad_norm": float(np.mean(run["grad_norms"])),
                "dev_nav": {key: value for key, value in run["dev_nav"].items() if key != "trades"},
                "appendix_nav": {key: value for key, value in run["appendix_nav"].items() if key != "trades"},
                "annualization_warning": run["dev_nav"]["annualization_warning"],
            }
            for run in runs
        ],
        "trades": [
            {"arm": run["arm"], "seed": run["seed"], "split": "dev", **trade}
            for run in runs
            for trade in run["dev_nav"]["trades"]
        ],
    }


def verify_panel_file(path: Path, digest: str) -> pd.DataFrame:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != digest:
        raise RuntimeError(f"panel sha256 {actual} != {digest}")
    frame = pd.read_csv(path, dtype={"股票代码": str})
    if len(frame) != PANEL_ROWS or frame["股票代码"].nunique() != PANEL_STOCKS:
        raise RuntimeError(f"panel shape {len(frame)} rows / {frame['股票代码'].nunique()} stocks")
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d")
    errors = panel_date_errors(frame["日期"].tolist())
    if errors:
        raise RuntimeError("; ".join(errors))
    if digest != PANEL_SHA256:
        raise RuntimeError("research digest does not match the frozen snapshot")
    return frame


def smoke_snapshot_entry(panel: pd.DataFrame, output_dir: Path) -> dict:
    """One real build plus one optimizer step per arm. Not a training run."""
    from config import config as model_config
    from train import CombinedRankingLoss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, market, calendar = build_samples(panel)
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    if not train_samples:
        raise RuntimeError("snapshot smoke produced no training windows")
    input_dim = int(train_samples[0]["x"].shape[-1])
    loss_fn = CombinedRankingLoss(
        k=model_config.get("topk_mle_k", 3),
        portfolio_weight=model_config.get("portfolio_loss_weight", 0.4),
        temperature=model_config.get("portfolio_temperature", 0.35),
    )
    steps = {}
    runs = []
    trades = []
    dev_means = {}
    for arm in ARMS:
        model = build_model(arm, input_dim, num_stocks=8, config=model_config)
        sample = train_samples[0]
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.from_numpy(sample["x"]).unsqueeze(0))
        target = torch.from_numpy(sample["train_target"]).unsqueeze(0)
        loss = loss_fn(prediction, target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        steps[arm] = {"loss_finite": bool(torch.isfinite(loss.detach())), "grad_norm": grad_norm}
        dev_rows = evaluate_split(model, samples, market, calendar, "dev")
        dev_means[arm] = mean_excess(dev_rows, "dev")
        nav = period_nav(dev_rows, "dev")
        runs.append(
            {
                "arm": arm,
                "parameters": count_parameters(model),
                "best_dev_excess": dev_means[arm],
                "dev_nav": {key: value for key, value in nav.items() if key != "trades"},
                "annualization_warning": nav["annualization_warning"],
            }
        )
        for trade in nav["trades"]:
            trades.append({"arm": arm, "split": "dev", **trade})
    summary = {
        "status": decide(dev_means),
        "asof": ASOF,
        "steps": steps,
        "dev_days": sum(sample["split"] == "dev" for sample in samples),
        "dev_means": dev_means,
        "runs": runs,
        "v4_weights_untouched": True,
        "smoke": True,
    }
    (output_dir / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(trades).to_csv(output_dir / "research_trades.csv", index=False)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = verify_panel_file(Path(args.panel), PANEL_SHA256)
    summary = run_study(frame)
    summary["wall_seconds"] = time.perf_counter() - started
    summary["asof"] = ASOF
    summary["v4_weights_untouched"] = True
    (output / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary["trades"]).to_csv(output / "research_trades.csv", index=False)
    print(json.dumps({"status": summary["status"], "dev_means": summary["dev_means"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
