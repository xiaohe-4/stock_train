# 时间记忆研究回测

笔记本：`kaggle/hs300_temporal_memory_research.ipynb`。它锁定提交 `b711d7e5a30df39ba44f3d61643793bfcc43a534`，其中包含已修正的 `research/study.py`。不要改回 v4 的 `c735797`，也不要再提交锁定 `370a181` 的 version 1。version 1 在导入 `feature_cols_map` 时失败，没有进入训练。

在 Kaggle 上打开 GPU，挂上已有数据集，Run All。不要改评估日期，也不要打开“按今天重新判断快照是否过期”。

数据集：

- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json`

输出在 `/kaggle/working/hs300_memory_research/`：`research_summary.json` 和 `research_trades.csv`。`status` 为 `research_candidate` 或 `do_not_adopt`。这不是新的交易榜单。

一张 T4 大约 30 到 90 分钟。只使用第一张 GPU。
