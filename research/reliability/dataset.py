"""Point-in-time samples from unadjusted bars. Labels stay outside the training clip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[2] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from research.reliability.folds import clip_bounds
from research.reliability.holding import resolve_holding


def _plain(code: str) -> str:
    return str(code).split(".")[-1].zfill(6)


def load_store(root) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    root = Path(root)
    calendar = json.loads((root / "calendar.json").read_text(encoding="utf-8"))
    membership = {}
    for date in calendar:
        rows = json.loads((root / "membership" / f"{date}.json").read_text(encoding="utf-8"))
        membership[date] = {_plain(row["code"]) for row in rows}
    frames = [pd.read_csv(path, dtype={"code": str}) for path in sorted((root / "bars").glob("*.csv"))]
    prices = pd.concat(frames, ignore_index=True)
    prices["code"] = prices["code"].map(_plain)
    prices["date"] = prices["date"].astype(str)
    for column in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn"):
        if column in prices.columns:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
    listings_raw = json.loads((root / "listings.json").read_text(encoding="utf-8"))
    listings = {_plain(code): {"ipoDate": row.get("ipoDate", "")} for code, row in listings_raw.items()}
    return prices, membership, listings, calendar


def _chinese(frame: pd.DataFrame) -> pd.DataFrame:
    preclose = frame["preclose"].replace(0, pd.NA)
    turn = frame["turn"] if "turn" in frame.columns else 0.0
    return pd.DataFrame(
        {
            "开盘": frame["open"].astype(float),
            "收盘": frame["close"].astype(float),
            "最高": frame["high"].astype(float),
            "最低": frame["low"].astype(float),
            "成交量": frame["volume"].astype(float),
            "成交额": frame["amount"].astype(float),
            "振幅": ((frame["high"] - frame["low"]) / preclose * 100).fillna(0),
            "涨跌额": (frame["close"] - frame["preclose"]).fillna(0),
            "涨跌幅": ((frame["close"] - frame["preclose"]) / preclose * 100).fillna(0),
            "换手率": pd.to_numeric(turn, errors="coerce").fillna(0),
        }
    )


def _matrix(group: pd.DataFrame, columns: list[str], engineer, cache_dir, code: str):
    dates = group["date"].astype(str).tolist()
    path = None if cache_dir is None else Path(cache_dir) / f"{code}.npz"
    if path is not None and path.exists():
        stored = np.load(path, allow_pickle=False)
        if stored["dates"].astype(str).tolist() == dates and stored["matrix"].shape[1] == len(columns):
            return stored["matrix"]
    features = engineer(_chinese(group))
    missing = [name for name in columns if name not in features.columns]
    if missing:
        raise RuntimeError(f"missing features: {missing[:8]}")
    matrix = features[columns].to_numpy(dtype=np.float32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dates=np.asarray(dates), matrix=matrix)
    return matrix


def build_samples(prices: pd.DataFrame, membership: dict, calendar: list[str], min_names: int = 10, cache_dir=None) -> list[dict]:
    from train import feature_cloums_map, feature_engineer_func_map

    columns = feature_cloums_map["158+39"]
    engineer = feature_engineer_func_map["158+39"]
    by_date: dict[str, list] = {}
    for code, group in prices.groupby("code", sort=False):
        group = group.drop_duplicates("date").sort_values("date")
        matrix = _matrix(group, columns, engineer, cache_dir, str(code))
        dates = group["date"].astype(str).tolist()
        opens = dict(zip(dates, group["open"].astype(float).tolist()))
        date_index = {date: index for index, date in enumerate(dates)}
        for date in dates:
            if code not in membership.get(date, set()):
                continue
            index = date_index[date]
            if index < 29:
                continue
            try:
                holding = resolve_holding(calendar, date)
            except ValueError:
                continue
            entry = opens.get(holding["entry_date"])
            exit_open = opens.get(holding["exit_date"])
            if entry is None or exit_open is None or entry <= 0 or entry != entry or exit_open != exit_open:
                continue
            window = matrix[index - 29 : index + 1]
            if window.shape != (30, len(columns)):
                continue
            by_date.setdefault(date, []).append((code, window, (exit_open - entry) / entry, holding))
    samples = []
    for date in calendar:
        rows = by_date.get(date, [])
        if len(rows) < min_names:
            continue
        holding = rows[0][3]
        samples.append(
            {
                "date": date,
                "codes": [row[0] for row in rows],
                "x": np.stack([row[1] for row in rows]).astype(np.float32),
                "raw": np.asarray([row[2] for row in rows], dtype=np.float32),
                "entry_date": holding["entry_date"],
                "exit_date": holding["exit_date"],
            }
        )
    return samples


def split_fold(samples: list[dict], fold: dict) -> dict:
    train = [sample for sample in samples if sample["date"] <= fold["train_end"] and sample["exit_date"] <= fold["train_end"]]
    dev = [sample for sample in samples if fold["dev_start"] <= sample["date"] <= fold["dev_end"]]
    test = [sample for sample in samples if fold["test_start"] <= sample["date"] <= fold["test_end"]]
    labels = [float(value) for sample in train for value in sample["raw"]]
    bounds = clip_bounds(labels) if labels else (0.0, 0.0)
    for sample in train:
        sample["train_target"] = np.clip(sample["raw"], bounds[0], bounds[1]).astype(np.float32)
    return {"train": train, "dev": dev, "test": test, "clip_bounds": bounds}
