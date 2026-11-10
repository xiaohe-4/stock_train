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
USE_OFFLINE_SNAPSHOT = True
ALLOW_STALE_SNAPSHOT = False
OFFLINE_PANEL = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv"
OFFLINE_NAMES = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv"
OFFLINE_MANIFEST = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json"
PINNED_COMMIT = "c735797c86ec3908379f7f54b70953c6a62b23b9"
REPO_URL = "https://github.com/xiaohe-4/stock_train.git"
print("mode", RUN_MODE)
assert RUN_MODE in {"full", "smoke", "preflight"}
if RUN_MODE != "full":
    print("THIS IS NOT THE FINAL RANKING")
from pathlib import Path
missing = [p for p in (OFFLINE_PANEL, OFFLINE_NAMES, OFFLINE_MANIFEST) if not Path(p).is_file()]
if USE_OFFLINE_SNAPSHOT and missing:
    raise SystemExit("attached snapshot files not found:\\n" + "\\n".join(missing))
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
    install = '''import importlib.util
import json
import subprocess
import sys

def _have(module):
    return importlib.util.find_spec(module) is not None

# Do not install or upgrade torch. Kaggle's CUDA build must stay as it is.
torch_before = None
if _have("torch"):
    import torch
    torch_before = (torch.__version__, bool(torch.cuda.is_available()), torch.cuda.device_count())
    print("torch before pip", torch_before)

# PyPI has baostock 0.9.4. Its protocol string is 00.9.40; there is no pip release 0.9.40.
# Install only missing scientific packages, with pins that have Python 3.13 wheels.
# only-if-needed avoids replacing the numpy that the existing torch was built against.
pins_if_missing = [
    ("pandas", "pandas==2.3.3"),
    ("sklearn", "scikit-learn==1.7.2"),
    ("joblib", "joblib==1.5.2"),
    ("tensorboardX", "tensorboardX==2.6.4"),
    ("tqdm", "tqdm==4.67.1"),
    ("talib", "ta-lib==0.6.8"),
]
cmd = [
    sys.executable, "-m", "pip", "install", "-q",
    "--upgrade-strategy", "only-if-needed",
    "baostock==0.9.4",
]
for module, pin in pins_if_missing:
    if not _have(module):
        cmd.append(pin)
print("pip install", " ".join(cmd[4:]), flush=True)
subprocess.check_call(cmd)

import baostock
import baostock.common.contants as cons
from importlib.metadata import version
if version("baostock") != "0.9.4":
    raise SystemExit("baostock dist is %s, expected 0.9.4" % version("baostock"))
if cons.BAOSTOCK_CLIENT_VERSION != "00.9.40":
    raise SystemExit("unexpected BaoStock protocol client %s" % cons.BAOSTOCK_CLIENT_VERSION)
print("baostock", version("baostock"), "protocol", cons.BAOSTOCK_CLIENT_VERSION)

if torch_before is not None:
    # A second import in this process is safe. Reloading torch is not: it registers triton again.
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import json, torch\\n"
            "print('TORCH_PROBE ' + json.dumps([torch.__version__, bool(torch.cuda.is_available()), int(torch.cuda.device_count())]))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    probed = [line for line in probe.stdout.splitlines() if line.startswith("TORCH_PROBE ")]
    if probe.returncode != 0 or len(probed) != 1:
        raise SystemExit("torch probe failed\\n" + probe.stdout + probe.stderr)
    torch_after = tuple(json.loads(probed[0].split(" ", 1)[1]))
    print("torch after pip", torch_after)
    if torch_after != torch_before:
        raise SystemExit("pip changed the existing torch install: %s -> %s" % (torch_before, torch_after))
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
