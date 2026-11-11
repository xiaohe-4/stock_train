# 严格可靠评估

笔记本：`kaggle/hs300_reliability_evaluation.ipynb`。它锁定提交 `d88427f74174028a57134681c9b7004a017c6bff`。这一提交包含种子修复、三颗种子、3 倍成本和日净值指标、训练集内部早停，以及点时数据复用。不要改回 v4 的 `c735797`，也不要覆盖三臂研究笔记本。

上一轮已经在 Kaggle 跑完：`caihuasheng/hs300-reliability-evaluation` version 1，`scriptVersionId` 368132794，提交 `32176c31`，4621 秒，审计通过，决策 `abstain`。这次笔记本不改那次结论，也不编造新的收益。

在 Kaggle 上打开 GPU 和网络。Run All。结果写到 `/kaggle/working/hs300_reliability/report`。

## 复用上一轮的点时数据

上一轮把成分、不复权日线和上市日写在了 `/kaggle/working/hs300_reliability/pit`。Kaggle 会把 notebook 的 working 目录存成该版本的 Output。

1. 导入本仓库里的新 `kaggle/hs300_reliability_evaluation.ipynb`，不要重跑 version 1。
2. 在右侧 Add Input 里选 Notebooks，找到自己的 `hs300-reliability-evaluation`，版本选 Version 1（`scriptVersionId` 368132794）。
3. 挂上之后，那份 Output 出现在 `/kaggle/input/<输入名>/` 下面，其中应有 `hs300_reliability/pit/manifest.json`。输入名由 Kaggle 生成，代码会在 `/kaggle/input` 下递归查找，不用把名字写死。
4. 程序核对 manifest 里每个文件的 sha256，并确认日历盖住 2024-01-02 至 2026-11-09。通过就把 `data_source` 记为 `reused`，直接审计这份输入，不再向 BaoStock 下载。输入目录是只读的，不会回写它的 manifest。
5. 哈希不一致、文件缺失或日期盖不住时，`data_source` 记为 `downloaded`，仍下载到 `/kaggle/working/hs300_reliability/pit`。审计照常执行，不通过就停止，不会把坏缓存当成严格数据。

网络仍然要开着：需要克隆锁定的提交；缓存被拒绝时也要能下载。GPU 仍然要开着，训练在 `cuda:0` 上。

## 这一版跑什么

流程是：解析缓存或下载、审计、只有 `strict_ready` 才训练。审计失败会停下并留下 `audit_report.json`。内部 LSTM 不打开。种子是 42、123、2024。门槛不变：每个折、每个种子的 1 倍和 2 倍超额都要大于 0。3 倍超额写入 `stress_excess_3x`，只报告。

日净值还会写出模型累计收益、等权基准累计收益、最大回撤、年化夏普、日均换手和费用合计。夏普是 `mean(r) / sample_std(r) * sqrt(252)`，`r` 是含空仓日的逐日净值收益，标准差 `ddof=1`。这些不参与门槛。

Transformer 早停改为训练集内部最后 20% 信号日上的 top-5 标签均值，不用折的开发验证或测试来选 epoch，也没有按那两段调学习率或 dropout。正式折的证据等级仍是 `already_observed`。`status` 只可能是 `abstain` 或 `process_only`，不会替换 v4。

程序从启动起 11 小时还没完成就会停，并把该次标成未完成。不要把未完成的文件当成严格结论。

上一轮实测 4621 秒里，大约 58 分钟是下载。缓存校验通过时这一段可以省掉。训练比上一轮多一颗种子，GPU 时间会比上一轮去掉下载后的那一段更长。这不是新的实测墙钟。

本地验证没有跑这 80 epoch，也没有新的收益数字。
