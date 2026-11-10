#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook for the three-arm research backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle" / "hs300_temporal_memory_research.ipynb"


def cell(kind: str, source: str) -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def build(commit: str) -> dict:
    intro = """# 沪深300 时间记忆研究回测

这是历史研究，不是 v4 排名，也不会覆盖 v4 权重或 `result.csv`。

- 数据固定为 as-of `2026-11-09` 的已核对快照。不按运行当天的时钟判断快照是否过期。
- 三臂从头训练：原 Transformer、单层 LSTM、第二层时间 block 内的 LSTM `(h, c)` 记忆。
- 开发验证是 `2026-06-01` 至 `2026-08-25`。`2026-09-09` 至 `2026-10-26` 只写入附录，不能决定采用，也不是 v4 的未见测试。
- 选模指标是开发验证日上重叠 5 日信号的平均超额，不把这些信号复利。
- 可交易净值预先规定为每 5 个信号日调仓一次。夏普用持有期收益，不用日收益。持有期少于 30 段时年化不稳定。
- 示例成本是 `0.0010 × 换手`。这不是真实全部交易成本。
- 面板是后复权。停牌按成交量为空或 0。涨跌停没有可靠的未复权价，正式净值不执行涨跌停拒单。
- 预计一张 T4 上约 30 到 90 分钟。脚本只用 `cuda:0`。若某个 seed 远超这个范围，先停止，不要改评估规则。
"""
    params = f'''PINNED_COMMIT = "{commit}"
REPO_URL = "https://github.com/xiaohe-4/stock_train.git"
PANEL = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv"
NAMES = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv"
MANIFEST = "/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json"
OUTPUT = "/kaggle/working/hs300_memory_research"
from pathlib import Path
missing = [path for path in (PANEL, NAMES, MANIFEST) if not Path(path).is_file()]
if missing:
    raise SystemExit("missing dataset files\\n" + "\\n".join(missing))
print("research commit", PINNED_COMMIT)
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
if not (REPO / "research" / "study.py").is_file():
    raise SystemExit("pinned commit has no research/study.py")
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
subprocess.check_call([
    sys.executable, "-m", "research.study",
    "--panel", PANEL,
    "--output-dir", OUTPUT,
], cwd="/kaggle/working/stock_train")
'''
    show = '''import json
from pathlib import Path
summary = json.loads(Path(OUTPUT, "research_summary.json").read_text(encoding="utf-8"))
print("status", summary["status"])
print("dev_means", summary["dev_means"])
print("median_epoch_seconds", summary["median_epoch_seconds"])
print("overlap", summary["overlap_jaccard_seed42"])
print(summary["limitation"])
print(summary["runs"][0]["annualization_warning"])
print("v4 untouched", summary["v4_weights_untouched"])
print("wall_seconds", summary["wall_seconds"])
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
    notebook = build(commit)
    DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
