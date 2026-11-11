#!/usr/bin/env python3
"""Build the Kaggle notebook that downloads point-in-time data and runs the strict study."""

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
    intro = """# 沪深300 严格可靠评估

这不是 v4 排名，也不会覆盖 v4 权重、`result.csv` 或已经完成的三臂研究输出。

- 笔记本自己用匿名 BaoStock 拉取 2024-01-02 至 2026-11-09 的每日成分、不复权日线和上市日。不需要原来的后复权数据集。
- 下载写到 `/kaggle/working/hs300_reliability/pit`，可以断点续传。审计不通过就停止，不会改用后复权继续。
- 审计通过后才做三折、种子 42、123 和 2024、等权、Ridge、浅层树和原 Transformer。内部 LSTM 不打开。
- 如果 `/kaggle/input` 里有一份校验通过的 `hs300_reliability/pit`（manifest 的 sha256 一致，且日历覆盖 2024-01-02 至 2026-11-09），就复用它并仍先审计。校验失败就重新下载。报告里的 `data_source` 写成 `reused` 或 `downloaded`。
- Transformer 早停只用训练集内部最后 20% 信号日，不看折的开发验证或测试。证据等级保持 already_observed。
- 特征只用训练折估计均值和标准差（标准差下限 1e-6），再截断到 ±10。非有限分数仍然直接失败。
- 可比持有期仍是 t+1 开盘进、t+5 开盘出，共 4 个交易日间隔。
- 总预算按单次 12 小时控制，程序在 11 小时处停止未完成的部分并标明未完成。
- 这里不写死任何收益数字。结果只来自本次运行写出的文件。
"""
    params = f'''PINNED_COMMIT = "{commit}"
REPO_URL = "https://github.com/xiaohe-4/stock_train.git"
PIT = "/kaggle/working/hs300_reliability/pit"
OUTPUT = "/kaggle/working/hs300_reliability/report"
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
    fetch = '''import json
import subprocess
import sys
completed = subprocess.run([
    sys.executable, "-m", "research.reliability.fetch_pit",
    "--output-dir", PIT,
    "--reuse-search", "/kaggle/input",
], cwd="/kaggle/working/stock_train", capture_output=True, text=True)
print(completed.stdout)
print(completed.stderr)
if completed.returncode != 0:
    raise SystemExit(completed.returncode)
lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
info = json.loads(lines[-1])
PIT = info["data_dir"]
DATA_SOURCE = info["data_source"]
DATA_SOURCE_PATH = info["data_dir"]
REUSE_REJECTION = info.get("reuse_rejection") or ""
print("data_source", DATA_SOURCE)
print("data_dir", PIT)
'''
    audit = '''import json
import subprocess
import sys
from pathlib import Path
completed = subprocess.run([
    sys.executable, "-m", "research.reliability.strict_run",
    "--data-dir", PIT,
    "--output-dir", OUTPUT,
    "--audit-only",
    "--data-source", DATA_SOURCE,
    "--data-source-path", DATA_SOURCE_PATH,
    "--reuse-rejection", REUSE_REJECTION,
], cwd="/kaggle/working/stock_train")
report = json.loads(Path(OUTPUT, "audit_report.json").read_text(encoding="utf-8"))
print("strict_ready", report["strict_ready"])
for item in report["blockers"]:
    print("-", item)
if completed.returncode != 0 or not report["strict_ready"]:
    raise SystemExit("strict audit failed; refusing to train")
'''
    run = '''import json
import subprocess
import sys
from pathlib import Path
completed = subprocess.run([
    sys.executable, "-m", "research.reliability.strict_run",
    "--data-dir", PIT,
    "--output-dir", OUTPUT,
    "--data-source", DATA_SOURCE,
    "--data-source-path", DATA_SOURCE_PATH,
    "--reuse-rejection", REUSE_REJECTION,
], cwd="/kaggle/working/stock_train")
print("returncode", completed.returncode)
summary_path = Path(OUTPUT, "summary.json")
if summary_path.is_file():
    print(summary_path.read_text(encoding="utf-8")[:2000])
if completed.returncode not in (0, 3):
    raise SystemExit(completed.returncode)
'''
    show = '''import json
from pathlib import Path
summary = json.loads(Path(OUTPUT, "summary.json").read_text(encoding="utf-8"))
print("status", summary["status"])
print("strict", summary["strict"], "trained", summary["trained"], "complete", summary["complete"])
print("selected_model", summary.get("selected_model"))
print("models", summary.get("models"))
print("feature_map", summary.get("feature_map"))
print("replaces_v4", summary.get("replaces_v4"))
print("holding_steps", summary.get("holding_steps"))
print("data_source", summary.get("data_source"))
print("seeds", summary.get("seeds"))
print("early_stop", summary.get("early_stop"))
print("nav_metrics_gated", summary.get("nav_metrics_gated"))
print(summary.get("sharpe_formula"))
print(summary.get("limitation"))
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
            cell("code", fetch),
            cell("code", audit),
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
