# 时间记忆研究回测

笔记本：`kaggle/hs300_temporal_memory_research.ipynb`。它克隆的提交必须包含 `research/study.py`，不是 v4 的 `c735797`。

在 Kaggle 上打开 GPU，挂上已有数据集，Run All。不要改评估日期，也不要打开“按今天重新判断快照是否过期”。

数据集：

- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json`

输出在 `/kaggle/working/hs300_memory_research/`：`research_summary.json` 和 `research_trades.csv`。`status` 为 `research_candidate` 或 `do_not_adopt`。这不是新的交易榜单。

一张 T4 大约 30 到 90 分钟。只使用第一张 GPU。
