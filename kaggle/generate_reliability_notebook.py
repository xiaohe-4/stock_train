#!/usr/bin/env python3
"""Build the Kaggle notebook that audits strict mode and runs only the CPU smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle" / "hs300_reliability_evaluation.ipynb"


def cell(kind: str, source: str) -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def build(commit: str) -> dict:
    intro = """# 沪深300 可靠评估审计

这不是 v4 排名，也不会覆盖 v4 权重、`result.csv` 或已经完成的三臂研究输出。

- 当前数据集是后复权、当前 300 只成分。它不能变成严格的 point-in-time 数据。
- 本笔记本默认只做依赖检查、锁定提交、数据审计和 CPU 烟测。不跑 80 epoch。
- 研究模式的结论是 `abstain`。这不是新的收益，也不能当成可上线。
- 可比持有期仍是信号日 t 的下一交易日开盘进、t+5 开盘出，中间是 4 个交易日间隔。
- 严格模式需要按日历史成分、不复权行情和上市日。缺任何一项就停止，不用后复权猜测涨跌停。
"""
    params = f'''PINNED_COMMIT = "{commit}"
REPO_URL = "https://github.com/xiaohe-4/stock_train.git"
PANEL = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv"
NAMES = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv"
MANIFEST = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json"
OUTPUT = "/kaggle/working/hs300_reliability"
from pathlib import Path
missing = [path for path in (PANEL, NAMES, MANIFEST) if not Path(path).is_file()]
if missing:
    raise SystemExit("missing dataset files\\n" + "\\n".join(missing))
print("reliability commit", PINNED_COMMIT)
'''
    setup = '''import subprocess
from pathlib import Path
REPO = Path("/kaggle/working/stock_train")
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
if not (REPO / "research" / "reliability" / "smoke.py").is_file():
    raise SystemExit("pinned commit has no reliability smoke")
'''
    install = r'''import importlib.util
import json
import subprocess
import sys

def _have(module):
    return importlib.util.find_spec(module) is not None

torch_before = None
if _have("torch"):
    import torch
    torch_before = (torch.__version__, bool(torch.cuda.is_available()), torch.cuda.device_count())
    print("torch before pip", torch_before)

pins_if_missing = [
    ("pandas", "pandas==2.3.3"),
    ("sklearn", "scikit-learn==1.7.2"),
    ("joblib", "joblib==1.5.2"),
    ("tensorboardX", "tensorboardX==2.6.4"),
    ("tqdm", "tqdm==4.67.1"),
    ("talib", "ta-lib==0.6.8"),
]
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", "baostock==0.9.4"]
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
    probe = subprocess.run(
        [sys.executable, "-c", "import json, torch\nprint('TORCH_PROBE ' + json.dumps([torch.__version__, bool(torch.cuda.is_available()), int(torch.cuda.device_count())]))"],
        check=False, capture_output=True, text=True,
    )
    probed = [line for line in probe.stdout.splitlines() if line.startswith("TORCH_PROBE ")]
    if probe.returncode != 0 or len(probed) != 1:
        raise SystemExit("torch probe failed\n" + probe.stdout + probe.stderr)
    torch_after = tuple(json.loads(probed[0].split(" ", 1)[1]))
    print("torch after pip", torch_after)
    if torch_after != torch_before:
        raise SystemExit("pip changed the existing torch install: %s -> %s" % (torch_before, torch_after))
'''
    run = '''import subprocess
import sys
PROBE = "/kaggle/working/stock_train/research/reliability/fixtures/baostock_pit_probe_2026-11-10.json"
subprocess.check_call([
    sys.executable, "-m", "research.reliability.smoke",
    "--panel", PANEL,
    "--probe", PROBE,
    "--output-dir", OUTPUT,
], cwd="/kaggle/working/stock_train")
'''
    show = '''import json
from pathlib import Path
summary = json.loads(Path(OUTPUT, "research_summary.json").read_text(encoding="utf-8"))
print("status", summary["status"])
print("mode", summary["mode"], "strict", summary["strict"])
print("feature_map", summary["feature_map"])
print("blockers")
for item in summary["blockers"]:
    print("-", item)
print("raw_limit_check", summary["raw_limit_check"])
print("smoke", summary["smoke"], "replaces_v4", summary["replaces_v4"])
print("no new return is claimed")
'''
    return {
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
            cell("markdown", intro),
            cell("code", params),
            cell("code", setup),
            cell("code", install),
            cell("code", run),
            cell("code", show),
        ],
    }


def main() -> None:
    commit = sys.argv[1]
    DEST.write_text(json.dumps(build(commit), ensure_ascii=False, indent=1), encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
