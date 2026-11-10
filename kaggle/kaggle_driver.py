#!/usr/bin/env python3
"""Prepare data and run the pinned stock_train ranking on Kaggle.

full  uses the repository defaults: features 158+39, sequence 30, leader ratio
0.75, epochs 80, patience 12, seeds 42 and 123. smoke is a pipeline check with
one epoch and one seed in a separate directory. preflight does not train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from refresh_hs300 import (
    SHANGHAI,
    audit_panel,
    calendar_cap,
    completeness_errors,
    load_fetcher,
    trading_days_between,
)

PINNED_COMMIT = "c735797c86ec3908379f7f54b70953c6a62b23b9"
SOURCE_SHA256 = {
    "app/code/train.py": "82aef547c4dab9b42cadfb977dd9456fb00b4a80dc4b68f2881ec0dc4830d8da",
    "app/code/predict.py": "16a05cb892a89cde684326e69d061fa452008be23507779d63d5837d3696ea48",
    "app/code/config.py": "085148582a70eb9d73f25eaa0553dc34439a2023db2d5008c11f76c03574bfd9",
    "app/code/model.py": "8986b4df7df689a281872bde217a7de99bc0b6bb0ec47d55b6cc8064d5deeb13",
    "app/code/utils.py": "533763a9451021656a3590491e7dd0d7285e0d8457c006901848b9754e886f12",
    "app/code/test.py": "b8dcb25b67de558d533fe7fb98cd4c023dd3550901c1875f293cc12918144df4",
    "THU-BDC2026-main/get_stock_data.py": "73201a01825235e9b4e835bbd1929487e049842503a4aac5a93d95793fa83b87",
}
OFFICIAL_WEIGHTS = [0.50, 0.47, 0.01, 0.01, 0.01]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_sources(repo: Path) -> None:
    mismatches = []
    for relative, expected in SOURCE_SHA256.items():
        path = repo / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append(f"{relative}: {actual} != {expected}")
    if mismatches:
        raise RuntimeError("pinned source mismatch:\n" + "\n".join(mismatches))


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"HEAD {commit} != pinned {PINNED_COMMIT}")
    return commit


def resolve_cap(repo: Path, now: datetime) -> tuple[str, list[str]]:
    import socket

    socket.setdefaulttimeout(90)
    fetcher = load_fetcher(repo)
    fetcher.login()
    try:
        trading_days = trading_days_between("2024-01-01", "2027-06-30")
    finally:
        fetcher.logout()
    return calendar_cap(now, trading_days), trading_days


def _cap_from_calendar(now: datetime, trading_days: list[str]) -> str:
    if not trading_days:
        raise RuntimeError("trading calendar is empty")
    return calendar_cap(now, trading_days)


def acquire_panel(
    repo: Path,
    output_dir: Path,
    now: datetime,
    offline_panel: Path | None,
    offline_names: Path | None,
    offline_manifest: Path | None,
    allow_stale: bool,
) -> dict:
    if offline_panel is None:
        code = refresh_main_with_args(repo, output_dir / "baostock_fetch", now)
        manifest_path = output_dir / "baostock_fetch" / "hs300_data_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if code != 0 or not manifest.get("complete"):
            raise RuntimeError(f"BaoStock refresh is not complete: {manifest.get('completeness_errors')}")
        if not manifest.get("trading_days"):
            raise RuntimeError("BaoStock manifest has no trading calendar")
        return manifest

    embedded_days: list[str] = []
    if offline_manifest is not None and offline_manifest.is_file():
        embedded = json.loads(offline_manifest.read_text(encoding="utf-8"))
        embedded_days = list(embedded.get("trading_days") or [])
    if embedded_days:
        cap = _cap_from_calendar(now, embedded_days)
        trading_days = embedded_days
        print("using the snapshot trading calendar; no second market download", flush=True)
    else:
        cap, trading_days = resolve_cap(repo, now)
    print(f"Asia/Shanghai now {now.isoformat()}; complete-session cap {cap}", flush=True)

    panel = pd.read_csv(offline_panel, dtype={"股票代码": str}, encoding="utf-8-sig")
    constituents = pd.read_csv(offline_names, dtype=str, encoding="utf-8-sig")
    panel["日期"] = pd.to_datetime(panel["日期"], errors="coerce")
    if panel["日期"].isna().any():
        raise RuntimeError("offline snapshot has unparseable dates")
    actual_max = panel["日期"].max().strftime("%Y-%m-%d")
    later = sorted({day for day in panel["日期"].dt.strftime("%Y-%m-%d") if day > cap})
    audit = audit_panel(panel.assign(日期=panel["日期"].dt.strftime("%Y-%m-%d")), constituents, actual_max)
    errors = []
    if later:
        errors.append(f"offline snapshot contains dates after the complete-session cap {cap}: {later[:5]}")
    if actual_max != cap and not allow_stale:
        errors.append(
            f"offline snapshot max date {actual_max} != current complete session {cap}; refusing to substitute older data"
        )
    errors.extend([item for item in completeness_errors(audit, []) if "earlier than" not in item])
    if errors:
        raise RuntimeError("offline snapshot rejected: " + "; ".join(errors))
    snapshot_dir = output_dir / "offline_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    panel_path = snapshot_dir / "hs300_daily_hfq_20240101_complete.csv"
    names_path = snapshot_dir / "hs300_constituents.csv"
    panel.assign(日期=panel["日期"].dt.strftime("%Y-%m-%d")).to_csv(panel_path, index=False, encoding="utf-8-sig")
    constituents.to_csv(names_path, index=False, encoding="utf-8-sig")
    return {
        "source": "offline_snapshot",
        "complete": True,
        "stale_snapshot": actual_max != cap,
        "complete_session_cap": cap,
        "trading_days": trading_days,
        "asia_shanghai_now": now.isoformat(),
        "audit": audit,
        "completeness_errors": [],
        "files": {
            "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
            "constituents": {"path": str(names_path), "sha256": sha256_file(names_path)},
        },
    }


def refresh_main_with_args(repo: Path, output_dir: Path, now: datetime) -> int:
    argv = sys.argv
    sys.argv = [
        "refresh_hs300.py",
        "--repo-root",
        str(repo),
        "--output-dir",
        str(output_dir),
        "--now",
        now.isoformat(),
    ]
    try:
        from refresh_hs300 import main as refresh_main

        return refresh_main()
    finally:
        sys.argv = argv


def materialize_split(panel_csv: Path, app_dir: Path) -> dict:
    """Write the full panel to both train.csv and test.csv.

    predict.py needs sequence_length=30 and up to 60-day indicators, so the
    inference file cannot be the repository's last-five-session split. Both
    files contain every downloaded row through the verified max date. Training
    still builds its own validation tail internally.
    """
    frame = pd.read_csv(panel_csv, dtype={"股票代码": str}, encoding="utf-8-sig")
    frame["股票代码"] = frame["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    data_dir = app_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    frame.to_csv(train_path, index=False)
    frame.to_csv(test_path, index=False)
    return {
        "rows": int(len(frame)),
        "stocks": int(frame["股票代码"].nunique()),
        "min_date": frame["日期"].min(),
        "max_date": frame["日期"].max(),
        "rows_on_max_date": int((frame["日期"] == frame["日期"].max()).sum()),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "note": "train.csv and test.csv are the same full panel; test.csv is not a 5-day slice",
    }


def prepare_run_dir(repo: Path, mode: str) -> Path:
    app = repo / "app"
    if mode == "full":
        return app
    smoke = repo / "_smoke_run"
    if smoke.exists():
        shutil.rmtree(smoke)
    smoke.mkdir()
    (smoke / "code").symlink_to(app / "code", target_is_directory=True)
    (smoke / "data").symlink_to(app / "data", target_is_directory=True)
    (smoke / "model").mkdir()
    (smoke / "output").mkdir()
    (smoke / "temp").mkdir()
    return smoke


def install_runtime_exporter(repo: Path) -> None:
    source = Path(__file__).resolve().parent / "export_ranking.py"
    target = repo / "app" / "code" / "export_ranking.py"
    shutil.copyfile(source, target)


def finished_seeds(run_dir: Path) -> list[int]:
    done = []
    for seed in (42, 123):
        seed_dir = run_dir / "model" / "30_158+39" / f"seed_{seed}"
        if (seed_dir / "best_model.pth").is_file() and (seed_dir / "final_score.txt").is_file():
            done.append(seed)
    return done


def module_env(run_dir: Path) -> dict[str, str]:
    """Let `python -c` and spawn workers import app/code/*.py.

    cwd stays the app directory because config uses ./data and ./model.
    `python -c` only puts that cwd on sys.path, not the code subdirectory.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    code_dir = str((run_dir / "code").resolve())
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = code_dir if not previous else code_dir + os.pathsep + previous
    return env


def run_training(run_dir: Path, mode: str, resume: bool) -> None:
    env = module_env(run_dir)
    # Call train.main() per seed. Leaving ensemble_seeds unchanged keeps the
    # saved config.json equal to the repository default even when a finished
    # seed is skipped on resume.
    script = r"""
import multiprocessing as mp
from pathlib import Path
import config
import train

mp.set_start_method('spawn', force=True)
mode = %r
resume = %s
if mode == 'smoke':
    print('SMOKE TEST ONLY — NOT A RANKING', flush=True)
    config.config['num_epochs'] = 1
    config.config['early_stopping_patience'] = 1
    seeds = [42]
else:
    print('FULL RUN defaults: epochs=80 patience=12 seeds=[42, 123] leader_ratio=0.75 features=158+39', flush=True)
    seeds = [42, 123]
    if config.config.get('ensemble_seeds') != [42, 123] or config.config.get('num_epochs') != 80:
        raise SystemExit('repository training defaults were already changed; refusing full run')
for seed in seeds:
    seed_dir = Path(config.config['output_dir']) / f'seed_{seed}'
    finished = (seed_dir / 'best_model.pth').is_file() and (seed_dir / 'final_score.txt').is_file()
    if mode == 'full' and resume and finished:
        print(f'RESUME skip finished seed {seed}', flush=True)
        continue
    print(f'========== train seed {seed} ==========', flush=True)
    config.config['seed'] = seed
    train.main()
""" % (mode, "True" if resume else "False")
    subprocess.run([sys.executable, "-u", "-c", script], cwd=run_dir, env=env, check=True)


def run_export(run_dir: Path, names_csv: Path, out_csv: Path, provenance_path: Path) -> None:
    env = module_env(run_dir)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "code/export_ranking.py",
            "--names",
            str(names_csv),
            "--out",
            str(out_csv),
            "--provenance",
            str(provenance_path),
        ],
        cwd=run_dir,
        env=env,
        check=True,
    )


def assert_full_training_config(run_dir: Path) -> None:
    expected = {
        "num_epochs": 80,
        "feature_num": "158+39",
        "sequence_length": 30,
        "train_leader_ratio": 0.75,
        "early_stopping_patience": 12,
        "topk_mle_k": 3,
        "portfolio_loss_weight": 0.4,
        "portfolio_temperature": 0.35,
        "selection_metric": "weighted_port_return",
    }
    for seed in (42, 123):
        path = run_dir / "model" / "30_158+39" / f"seed_{seed}" / "config.json"
        if not path.is_file():
            raise RuntimeError(f"missing training config: {path}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f"seed {seed} {key}={saved.get(key)!r}, expected {value!r}")
        if [float(item) for item in saved.get("top_k_weights", [])] != OFFICIAL_WEIGHTS:
            raise RuntimeError(f"seed {seed} weights changed: {saved.get('top_k_weights')}")
        if saved.get("ensemble_seeds") != [42, 123]:
            raise RuntimeError(f"seed {seed} ensemble_seeds={saved.get('ensemble_seeds')}")


def verify_ranking(ranking_csv: Path, result_csv: Path, provenance: dict) -> list[str]:
    ranking = pd.read_csv(ranking_csv, dtype={"stock_code": str}, encoding="utf-8-sig")
    result = pd.read_csv(result_csv, dtype={"stock_id": str})
    errors = []
    if provenance["run_mode"] != "full":
        errors.append("run_mode is not full")
    if ranking.empty:
        return ["ranking is empty"]
    ranking["stock_code"] = ranking["stock_code"].astype(str).str.zfill(6)
    result["stock_id"] = result["stock_id"].astype(str).str.zfill(6)
    if not ranking["model_score"].map(lambda value: pd.notna(value) and abs(float(value)) < 1e6).all():
        errors.append("model scores are missing or non-finite")
    scores = ranking["model_score"].tolist()
    if scores != sorted(scores, reverse=True):
        errors.append("ranking is not sorted by model_score descending")
    if ranking["stock_code"].duplicated().any():
        errors.append("duplicate stock codes")
    if len(ranking) < 50:
        errors.append(f"only {len(ranking)} scored names; leader pool looks truncated")
    if set(ranking["asof_date"]) != {provenance["expected_asof_date"]}:
        errors.append("asof_date does not match the verified complete session")
    if set(ranking["horizon"]) != {"未来5个交易日开盘到开盘收益: (open[t+5]-open[t+1])/open[t+1]"}:
        errors.append("horizon text mismatch")
    if set(ranking["seeds"]) != {"42,123"}:
        errors.append("ranking was not produced by both seeds")
    trading_days = list(provenance.get("trading_days") or [])
    asof = provenance["expected_asof_date"]
    if trading_days and asof in trading_days:
        entry = trading_days[trading_days.index(asof) + 1]
        exit_day = trading_days[trading_days.index(asof) + 5]
        if set(ranking["entry_open_date"].astype(str)) != {entry}:
            errors.append(f"entry_open_date is not the next trading day {entry}")
        if set(ranking["exit_open_date"].astype(str)) != {exit_day}:
            errors.append(f"exit_open_date is not t+5 trading day {exit_day}")
    if ranking["stock_name"].fillna("").eq("").any():
        errors.append("at least one scored stock has no name")
    top = ranking.nsmallest(len(ranking), "rank").head(5)
    if top["stock_code"].tolist() != result["stock_id"].tolist():
        errors.append("result.csv order does not match ranking_full.csv")
    if [float(value) for value in result["weight"].tolist()] != OFFICIAL_WEIGHTS:
        errors.append(f"weights are not {OFFICIAL_WEIGHTS}")
    stale_mask = ranking["stale_bar"].astype(str).str.lower().isin(["true", "1"])
    if stale_mask.any():
        errors.append("scored names whose last bar is older than asof: " + ",".join(ranking.loc[stale_mask, "stock_code"].tolist()[:20]))
    return errors


def preflight(repo: Path, manifest: dict, mode: str) -> dict:
    import torch

    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_devices": [],
        "data_max_date": manifest["audit"]["max_date"],
        "data_rows": manifest["audit"]["rows"],
        "data_stocks": manifest["audit"]["stocks"],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["cuda_devices"].append({"index": index, "name": props.name, "total_memory": props.total_memory})
    try:
        import talib

        report["talib"] = talib.__version__
    except Exception as exc:
        report["talib_error"] = str(exc)
    if mode == "full":
        if not report["cuda"]:
            raise RuntimeError("full mode requires the Kaggle GPU accelerator; CUDA is not available")
        if "talib_error" in report:
            raise RuntimeError(f"TA-Lib is required by the feature code: {report['talib_error']}")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def run(mode: str, repo: Path, output_dir: Path, resume: bool, offline_panel: Path | None, offline_names: Path | None, offline_manifest: Path | None, allow_stale: bool, now: datetime) -> dict:
    if mode not in {"preflight", "smoke", "full"}:
        raise RuntimeError(f"unknown mode {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_pinned_sources(repo)
    commit = git_commit(repo)
    manifest = acquire_panel(repo, output_dir, now, offline_panel, offline_names, offline_manifest, allow_stale)
    asof = manifest["audit"]["max_date"]
    if manifest.get("stale_snapshot") and mode == "full":
        raise RuntimeError(
            f"snapshot max date {asof} is older than the complete-session cap {manifest.get('complete_session_cap')}; "
            "full mode will not train a ranking on substituted older data"
        )
    if manifest.get("dates_after_requested_end") or manifest["audit"].get("dates_after_requested_end"):
        raise RuntimeError("panel contains a date after the complete-session cap")
    split = materialize_split(Path(manifest["files"]["panel"]["path"]), repo / "app")
    if split["max_date"] != asof:
        raise RuntimeError("materialized split max date drifted")
    hardware = preflight(repo, manifest, mode)
    summary = {
        "mode": mode,
        "commit": commit,
        "asof_date": asof,
        "split": split,
        "hardware": hardware,
        "data_manifest_complete": manifest.get("complete"),
        "stale_snapshot": bool(manifest.get("stale_snapshot")),
        "source": manifest.get("source"),
    }
    if mode == "preflight":
        (output_dir / "preflight.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("PREFLIGHT ONLY — no model was trained and no ranking was produced", flush=True)
        return summary

    install_runtime_exporter(repo)
    run_dir = prepare_run_dir(repo, mode)
    run_training(run_dir, mode, resume=resume and mode == "full")
    if mode == "smoke":
        print("SMOKE TEST ONLY — NOT A RANKING. Full defaults were not used.", flush=True)
        (output_dir / "SMOKE_NOT_A_RANKING.txt").write_text(
            "This run used 1 epoch and seed 42 in _smoke_run. It is not the predicted ranking.\n",
            encoding="utf-8",
        )
        return summary

    provenance = {
        "run_mode": "full",
        "commit": commit,
        "required_seeds": [42, 123],
        "expected_asof_date": asof,
        "trading_days": manifest.get("trading_days") or [],
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False), encoding="utf-8")
    assert_full_training_config(run_dir)
    ranking_path = output_dir / "ranking_full.csv"
    run_export(run_dir, Path(manifest["files"]["constituents"]["path"]), ranking_path, provenance_path)
    errors = verify_ranking(ranking_path, run_dir / "output" / "result.csv", provenance)
    shutil.copyfile(run_dir / "output" / "result.csv", output_dir / "result.csv")
    verification = {
        "passed": not errors,
        "errors": errors,
        "criteria": [
            "RUN_MODE was full",
            "git HEAD and source hashes match c735797c86ec3908379f7f54b70953c6a62b23b9",
            "panel max date equals the Asia/Shanghai complete-session cap and has no later rows",
            "both seed_42 and seed_123 finished and were loaded",
            "features remain 158+39, sequence_length 30, leader ratio 0.75, weights [0.50, 0.47, 0.01, 0.01, 0.01]",
            "ranking_full.csv is sorted by model_score descending and includes code, name, score, horizon, asof_date",
            "result.csv top 5 codes and weights match the first five ranking rows",
            "no scored name has a last bar older than asof_date",
        ],
    }
    (output_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path = output_dir / ("VERIFY_PASS.txt" if not errors else "VERIFY_FAIL.txt")
    status_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["verification"] = verification
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("ranking verification failed: " + "; ".join(errors))
    print("FULL RANKING VERIFIED", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["preflight", "smoke", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline-panel", type=Path)
    parser.add_argument("--offline-names", type=Path)
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--allow-stale-snapshot", action="store_true")
    parser.add_argument("--now", default="")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(SHANGHAI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=SHANGHAI)
    if bool(args.offline_panel) != bool(args.offline_names):
        raise SystemExit("offline snapshot requires both --offline-panel and --offline-names")
    run(
        args.mode,
        args.repo,
        args.output_dir,
        args.resume,
        args.offline_panel,
        args.offline_names,
        args.offline_manifest,
        args.allow_stale_snapshot,
        now,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
