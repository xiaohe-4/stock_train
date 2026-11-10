# 严格可靠评估

笔记本：`kaggle/hs300_reliability_evaluation.ipynb`。它锁定提交 `32176c31eb5f322348edb81af56de8f657cccb13`。这一提交包含匿名下载、审计和三折训练。不要改回 v4 的 `c735797`，也不要覆盖三臂研究笔记本。

在 Kaggle 上打开 GPU 和网络，不需要挂原来的后复权数据集。Run All。下载写到 `/kaggle/working/hs300_reliability/pit`，结果写到 `/kaggle/working/hs300_reliability/report`。

流程是：拉取、审计、只有 `strict_ready` 才训练。审计失败会停下并留下 `audit_report.json`。内部 LSTM 不打开。种子是 42 和 123，最多 80 epoch，patience 12，学习率 2e-4。模型是等权、Ridge、浅层树和原 Transformer。每个折、每个种子、成本 1/2/3 倍都有逐日净值和逐笔文件。`status` 只可能是 `abstain` 或 `process_only`，不会替换 v4。

程序从启动起 11 小时还没完成就会停，并把该次标成未完成。不要把未完成的文件当成严格结论。

预计时长，按 2026-11-10 的探测量级，不是这次的实测墙钟：

- 689 个交易日的成分查询大约 1.5 到 4 小时，含重试和每天约 0.25 秒限速。
- 大约 400 只股票的不复权日线和上市日大约再 0.5 到 1.5 小时。
- 特征缓存和三折、两颗种子的 Transformer 大约 2 到 6 小时。Ridge 和树模型远小于这项。
- 合计计划落在单次 12 小时以内。下载文件按日、按股票保存，同一会话里中断后可以续传。

本地验证没有跑这 80 epoch，也没有新的收益数字。
