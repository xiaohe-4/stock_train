#!/usr/bin/env python3
"""Export the full cross-section ranking and the official top-5 file.

This is the inference path for the Kaggle run. Feature construction, the
leader pool, the volatility filter, the ensemble mean, and the top-5 weights
are the ones implemented in app/code/predict.py at the pinned commit. The
extra file adds the stock name, the label horizon, and every scored name.
It does not fit a new model and it does not change config.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = "未来5个交易日开盘到开盘收益: (open[t+5]-open[t+1])/open[t+1]"
LABEL_FORMULA = "(open[t+5]-open[t+1])/open[t+1]"


def _load_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    code_column = "code" if "code" in frame.columns else "股票代码"
    name_column = "code_name" if "code_name" in frame.columns else "股票名称"
    names = {}
    for _, row in frame.iterrows():
        code = str(row[code_column]).replace("sh.", "").replace("sz.", "").replace(".0", "")
        code = code.zfill(6)
        names[code] = "" if pd.isna(row[name_column]) else str(row[name_column])
    return names


def _nth_trading_day(trading_days: list[str], asof: str, step: int) -> str:
    if asof not in trading_days:
        raise ValueError(f"{asof} is not in the trading calendar")
    index = trading_days.index(asof) + step
    if index < 0 or index >= len(trading_days):
        raise ValueError(f"cannot resolve t+{step} from {asof}")
    return trading_days[index]


def export_ranking(names_csv: Path, out_csv: Path, provenance_path: Path) -> pd.DataFrame:
    import multiprocessing as mp

    import joblib
    import torch

    mp.set_start_method("spawn", force=True)

    from config import config
    from model import StockTransformer
    from predict import (
        _find_model_root,
        _find_seed_dir,
        build_inference_sequences,
        preprocess_predict_data,
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required_seeds = [int(seed) for seed in provenance["required_seeds"]]
    if provenance.get("run_mode") == "full" and required_seeds != [42, 123]:
        raise RuntimeError(f"full ranking requires seeds [42, 123], got {required_seeds}")

    ensemble_seeds = config.get("ensemble_seeds", [42])
    data_file = os.path.join(config["data_path"], config.get("predict_file", "test.csv"))
    output_dir = _find_model_root(config["output_dir"])
    model_config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        for key in [
            "sequence_length",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "feature_num",
            "cross_stock_layers",
            "top_k_output",
            "top_k_weights",
            "predict_use_leader_pool",
        ]:
            if key in model_config:
                config[key] = model_config[key]

    def resolve(name: str) -> str:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            return path
        return os.path.join(_find_seed_dir(output_dir, ensemble_seeds[0]), name)

    scaler_path = resolve("scaler.pkl")
    stockid2idx_path = resolve("stockid2idx.pkl")
    medians_path = resolve("train_medians.pkl")
    leader_path = resolve("leader_stock_ids.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)
    if not os.path.exists(data_file):
        raise FileNotFoundError(data_file)

    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    latest_date = raw_df["日期"].max()
    asof = latest_date.strftime("%Y-%m-%d")
    if provenance.get("expected_asof_date") and asof != provenance["expected_asof_date"]:
        raise RuntimeError(f"predict file max date {asof} != expected {provenance['expected_asof_date']}")

    raw_map = joblib.load(stockid2idx_path)
    stockid2idx = {str(sid).replace(".0", "").zfill(6): int(idx) for sid, idx in raw_map.items()}
    stock_ids = sorted(stockid2idx.keys())
    if config.get("predict_use_leader_pool", True) and os.path.exists(leader_path):
        leader_stock_ids = {str(s).replace(".0", "").zfill(6) for s in joblib.load(leader_path)}
        stock_ids = [sid for sid in stock_ids if sid in leader_stock_ids]
        print(f"预测限制在训练龙头池: {len(stock_ids)} 只")
    elif config.get("predict_use_leader_pool", True):
        print("警告: 未找到 leader_stock_ids.pkl，回退为全市场预测")

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    if os.path.exists(medians_path):
        processed[features] = processed[features].fillna(joblib.load(medians_path))
    else:
        processed[features] = processed[features].fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        config["sequence_length"],
        stock_ids,
        latest_date,
    )

    excluded = []
    if "volatility_20" in features:
        vol_idx = features.index("volatility_20")
        last_vol = sequences_np[:, -1, vol_idx]
        vol_mean = float(np.nanmean(last_vol))
        vol_std = float(np.nanstd(last_vol))
        threshold = vol_mean + 2.0 * vol_std
        keep = last_vol <= threshold
        min_keep = max(config.get("top_k_output", 5) * 3, 20)
        if (not np.all(keep)) and int(keep.sum()) >= min_keep:
            excluded = [sid for sid, flag in zip(sequence_stock_ids, keep) if not flag]
            sequences_np = sequences_np[keep]
            sequence_stock_ids = [sid for sid, flag in zip(sequence_stock_ids, keep) if flag]
            print(f"风险过滤: 排除 {len(excluded)} 只，保留 {len(sequence_stock_ids)} 只")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if provenance.get("run_mode") == "full" and device.type != "cuda":
        raise RuntimeError("full ranking requires CUDA; refusing CPU scores")

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stockid2idx))
    model.to(device)
    model.eval()
    per_seed = []
    loaded = []
    for seed in ensemble_seeds:
        model_path = os.path.join(_find_seed_dir(output_dir, seed), "best_model.pth")
        if not os.path.exists(model_path):
            print(f"警告: seed={seed} 模型不存在，跳过")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        with torch.no_grad():
            scores = model(torch.from_numpy(sequences_np).unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
        per_seed.append(scores)
        loaded.append(int(seed))
        print(f"seed={seed} 模型已加载并预测")
    if loaded != required_seeds:
        raise RuntimeError(f"loaded seeds {loaded} != required {required_seeds}")

    scores = np.mean(per_seed, axis=0)
    order = np.argsort(scores)[::-1]
    top_k = int(config.get("top_k_output", 5))
    weights = [float(value) for value in config.get("top_k_weights")]
    if len(weights) != top_k:
        raise RuntimeError(f"top_k_weights length {len(weights)} != top_k_output {top_k}")
    if provenance.get("run_mode") == "full" and weights != [0.50, 0.47, 0.01, 0.01, 0.01]:
        raise RuntimeError(f"unexpected weights {weights}")

    last_bar = raw_df.groupby("股票代码")["日期"].max().dt.strftime("%Y-%m-%d").to_dict()
    names = _load_names(names_csv)
    trading_days = list(provenance.get("trading_days") or [])
    entry = _nth_trading_day(trading_days, asof, 1) if trading_days else ""
    exit_day = _nth_trading_day(trading_days, asof, 5) if trading_days else ""
    rows = []
    for rank, index in enumerate(order, start=1):
        code = sequence_stock_ids[index]
        rows.append(
            {
                "rank": rank,
                "stock_code": code,
                "stock_name": names.get(code, ""),
                "model_score": float(scores[index]),
                "horizon": HORIZON,
                "asof_date": asof,
                "entry_open_date": entry,
                "exit_open_date": exit_day,
                "label_formula": LABEL_FORMULA,
                "last_bar_date": last_bar.get(code, ""),
                "stale_bar": bool(last_bar.get(code, "") != asof),
                "weight": weights[rank - 1] if rank <= top_k else "",
                "run_mode": provenance["run_mode"],
                "commit": provenance["commit"],
                "seeds": ",".join(str(seed) for seed in loaded),
            }
        )
    ranking = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_csv, index=False, encoding="utf-8-sig")

    official = ranking.loc[ranking["rank"] <= top_k, ["stock_code", "weight"]].rename(columns={"stock_code": "stock_id"})
    official_path = Path("./output/result.csv")
    official_path.parent.mkdir(parents=True, exist_ok=True)
    official.to_csv(official_path, index=False)
    excluded_path = out_csv.with_name("volatility_excluded.csv")
    pd.DataFrame({"stock_code": excluded, "stock_name": [names.get(code, "") for code in excluded]}).to_csv(
        excluded_path, index=False, encoding="utf-8-sig"
    )
    print(f"预测日期: {asof}")
    print(f"参与排序股票数: {len(ranking)}")
    print(f"结果已写入: {official_path} 和 {out_csv}")
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    export_ranking(args.names, args.out, args.provenance)


if __name__ == "__main__":
    main()
