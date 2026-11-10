#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook from the reviewed helper scripts."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HELPERS = ["refresh_hs300.py", "export_ranking.py", "kaggle_driver.py"]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    blobs = {name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii") for name in HELPERS}
    intro = (ROOT / "NOTEBOOK_INTRO.md").read_text(encoding="utf-8")
    parameters = '''RUN_MODE = "full"  # full | smoke | preflight
RESUME_COMPLETED_SEEDS = False
USE_OFFLINE_SNAPSHOT = False
ALLOW_STALE_SNAPSHOT = False
OFFLINE_PANEL = "/kaggle/input/REPLACE_DATASET/hs300_daily_hfq_20240101_complete.csv"
OFFLINE_NAMES = "/kaggle/input/REPLACE_DATASET/hs300_constituents.csv"
OFFLINE_MANIFEST = "/kaggle/input/REPLACE_DATASET/hs300_data_manifest.json"
PINNED_COMMIT = "c735797c86ec3908379f7f54b70953c6a62b23b9"
REPO_URL = "https://github.com/xiaohe-4/stock_train.git"
print("mode", RUN_MODE)
assert RUN_MODE in {"full", "smoke", "preflight"}
if RUN_MODE != "full":
    print("THIS IS NOT THE FINAL RANKING")
'''
    setup = '''import base64
import subprocess
import sys
from pathlib import Path

KIT = Path("/kaggle/working/hs300_kaggle_kit")
REPO = Path("/kaggle/working/stock_train")
OUTPUT = Path("/kaggle/working/hs300_rank_output")
KIT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
BLOBS = %s
for name, payload in BLOBS.items():
    (KIT / name).write_bytes(base64.b64decode(payload))
    print("wrote", name, (KIT / name).stat().st_size)

if not (REPO / ".git").exists():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "remote", "add", "origin", REPO_URL])
subprocess.check_call(["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", PINNED_COMMIT])
subprocess.check_call(["git", "-C", str(REPO), "checkout", "--force", "--detach", "FETCH_HEAD"])
head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("HEAD", head)
if head != PINNED_COMMIT:
    raise SystemExit("checked out commit does not match the pin")
''' % json.dumps(blobs)
    install = '''import subprocess
import sys
# Keep the CUDA torch already installed on the Kaggle image.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "baostock==0.9.40",
    "pandas>=2.3.2",
    "scikit-learn>=1.7.2",
    "joblib>=1.5.2",
    "tensorboardX>=2.6.4",
    "tqdm>=4.67.1",
    "ta-lib>=0.6.8",
])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
'''
    execute = '''import subprocess
import sys
from pathlib import Path

command = [
    sys.executable, "-u", str(KIT / "kaggle_driver.py"),
    "--repo", str(REPO),
    "--output-dir", str(OUTPUT),
    "--mode", RUN_MODE,
]
if RESUME_COMPLETED_SEEDS:
    command.append("--resume")
if USE_OFFLINE_SNAPSHOT:
    command.extend([
        "--offline-panel", OFFLINE_PANEL,
        "--offline-names", OFFLINE_NAMES,
        "--offline-manifest", OFFLINE_MANIFEST,
    ])
if ALLOW_STALE_SNAPSHOT:
    command.append("--allow-stale-snapshot")
print(" ".join(command), flush=True)
subprocess.check_call(command)
'''
    show = '''from pathlib import Path
import pandas as pd
ranking_path = Path("/kaggle/working/hs300_rank_output/ranking_full.csv")
result_path = Path("/kaggle/working/hs300_rank_output/result.csv")
pass_path = Path("/kaggle/working/hs300_rank_output/VERIFY_PASS.txt")
if RUN_MODE != "full":
    print("No final ranking is displayed because RUN_MODE is", RUN_MODE)
elif not pass_path.is_file():
    raise SystemExit("VERIFY_PASS.txt is missing; do not use any CSV from this run as the ranking")
else:
    ranking = pd.read_csv(ranking_path, dtype={"stock_code": str}, encoding="utf-8-sig")
    result = pd.read_csv(result_path, dtype={"stock_id": str})
    print("asof", ranking["asof_date"].iloc[0])
    print("horizon", ranking["horizon"].iloc[0])
    print("entry", ranking["entry_open_date"].iloc[0], "exit", ranking["exit_open_date"].iloc[0])
    print(result.to_string(index=False))
    print(ranking.head(15).to_string(index=False))
'''
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {
                "accelerator": "NvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": [
            markdown_cell(intro),
            code_cell(parameters),
            code_cell(setup),
            code_cell(install),
            code_cell(execute),
            code_cell(show),
        ],
    }
    destination = ROOT / "hs300_5d_rank_kaggle.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(destination, destination.stat().st_size)


if __name__ == "__main__":
    main()
