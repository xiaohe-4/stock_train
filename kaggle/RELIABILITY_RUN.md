# 可靠评估

笔记本：`kaggle/hs300_reliability_evaluation.ipynb`。它锁定提交 `3d74ac65b6a9ff95f7d4d2b897825c0ba1be2976`，其中包含 `research/reliability/smoke.py`。不要改回 v4 的 `c735797`，也不要覆盖三臂研究笔记本。

在 Kaggle 上打开同一份后复权数据集，Run All。默认命令没有 `--full`，所以不会开始 80 epoch。输出在 `/kaggle/working/hs300_reliability/`。`status` 应为 `abstain`。这不是新收益，也不替换 v4。

数据集：

- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_daily_hfq_20240101_complete.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_constituents.csv`
- `/kaggle/input/datasets/caihuasheng/hs300-latest-complete-daily-snapshot/hs300_data_manifest.json`

严格模式还需要三份现在没有的数据，来源可以是匿名 BaoStock，不需要注册：

- 每个交易日的沪深300成分，2024-01-02 至 2026-11-09 共 689 个交易日。端点比较已看到 62 只被调出。
- 这些成分并集的不复权日线，字段包含前收、成交量、`tradestatus`、`isST`、`adjustflag=3`。
- `query_stock_basic` 的上市日。

成分查询大约 0.6 到 8 秒一天，全历史大约 30 到 90 分钟。按约 400 只股票再拉取行情和上市日，大约再 30 分钟。这是探测时的量级，不是一次完整下载的计时。探测样本和 sha256 在 `research/reliability/fixtures/`。

完整滚动训练仍是每折从头训练、种子 42 和 123、最多 80 epoch。一张 T4 上，仅 Transformer 三折的量级与已经完成的三臂研究同级或更高，可能是一到数小时。线性模型和树模型小得多。在严格数据齐备之前不要启动。
