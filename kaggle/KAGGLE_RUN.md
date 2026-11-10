# 沪深300五日排序：交给 Kaggle 执行

云主机没有 GPU，没有在 Kaggle 上运行，也没有生成排序。下面的文件可以直接上传。

## 要上传的文件

- `hs300_5d_rank_kaggle.ipynb`：新建 Kaggle 笔记本后导入。不要打开或覆盖已有的私有 GPU 检查笔记本。
- 现场下载失败时才另外上传这三个文件，做成私有 Dataset，并把笔记本第一个单元的 `USE_OFFLINE_SNAPSHOT` 改成 `True`，路径改成该 Dataset：
  - `hs300_daily_hfq_20240101_complete.csv`
  - `hs300_constituents.csv`
  - `hs300_data_manifest.json`

默认 `USE_OFFLINE_SNAPSHOT = False`。笔记本会自己连 BaoStock。快照日期如果早于当场算出的完整交易日，full 模式会拒绝训练。

## 已经核对的数据

- 来源：BaoStock `public-api.baostock.com:10030`，匿名登录，不需要 token。复权 `adjustflag=1`（后复权），字段和 `THU-BDC2026-main/get_stock_data.py` 的 `get_stock_history` 相同。
- 拉取时刻：2026-11-10 12:15 Asia/Shanghai。当天是交易日，午间没有完整日线。
- 实际最大日期：**2026-11-09**。300 只股票在这一天都有 K 线，没有更晚的日期。
- 区间：2024-01-02 至 2026-11-09，205,626 行，300 只，689 个交易日。成分股列表的 `updateDate` 是 2026-11-09。
- 面板 SHA256：`1bd7ff55788340b864e2f8c4d0c70f7ce7f344d181103fba4e85fee7a13526f2`
- 浦发银行 600000 在 2024-01-02 的后复权开盘 79.22708118、收盘 78.8685876，与仓库里旧文件同一天相同。
- 254 行历史停牌记录有价格、成交量为空，没有落在 2026-11-09。
- 仓库自带数据停在 2026-07-22，不能当作这次的最新完整日。

## 标签区间

仓库标签是 `(open[t+5] - open[t+1]) / open[t+1]`。

对 as-of 日 t = 2026-11-09：

- t+1 开盘日：2026-11-10
- t+5 开盘日：2026-11-16

模型只用到 2026-11-09 为止的日线，看不到 2026-11-10 的盘中数据。

## 原样保留的默认值

full 模式不改这些值：

- 代码提交：`c735797c86ec3908379f7f54b70953c6a62b23b9`
- 特征：`158+39`（列表长度 196），序列长度 30
- epoch 80，早停耐心 12，学习率 `2e-4`
- seed 42 和 123，两个模型的分数取平均
- 龙头池：训练集最近 30 日平均成交额的前 75%
- TopK-ListMLE 的 K=3，组合温度 0.35，损失权重 0.4
- 选模指标 `weighted_port_return`
- 前五名权重 `[0.50, 0.47, 0.01, 0.01, 0.01]`
- 推理时的波动率过滤仍是 `predict.py` 里的均值加两倍标准差

训练开始日期沿用脚本里的 2024-01-01，不是文件头注释里的 2015。`train.csv` 和 `test.csv` 都写成这份完整日线，因为原来的最后 5 个交易日切片不够 30 日序列。被打分的日期是文件最大日期，这一天还没有 t+5 标签。

## Kaggle 上怎么跑

1. 账号 `caihuasheng` 登录。新建笔记本，确认不是 GPU 检查笔记本，再导入 ipynb。
2. Accelerator 用 GPU T4。代码只用 `cuda:0`。Internet 打开。不要设为 Public。
3. 确认第一个单元 `RUN_MODE = "full"`，`RESUME_COMPLETED_SEEDS = False`。
4. Run All。
5. 会话还在时下载 `/kaggle/working/hs300_rank_output/` 里的 `ranking_full.csv`、`result.csv`、`VERIFY_PASS.txt`、`run_manifest.json`。

`smoke` 和 `preflight` 不会产出正式排序。

## 时间和资源

这台机器上的测量，不是 Kaggle 实测：

- 300 只股票下载大约 26 分钟。
- 单只特征大约 0.06 秒，300 只大约 20 秒。
- 模型 458,370 个参数。CPU 上一个 8×225×30×196 的 batch 大约 3.2 秒。按大约 80 个 batch、80 个 epoch、两个 seed 估算，CPU 大约 11 小时。这里没有 T4，不能把这个数字当成 Kaggle 时间。
- 日线和 checkpoint 远小于 20GB。不要重装 Kaggle 自带的 CUDA PyTorch。

第一个 epoch 结束后看日志。剩余时间大约是「该 epoch 秒数 × 剩余 epoch × 剩余 seed」。如果两个 seed 会超过 10 小时，让当前 seed 跑完并写出 `final_score.txt`，下一会话把 `RESUME_COMPLETED_SEEDS` 设为 `True`。这仍是两个完整 seed，不是把 epoch 或股票池改小。

## 怎样确认排序是真的

`VERIFY_PASS.txt` 出现，并且同时满足：模式是 full；HEAD 是上面的提交；as-of 日期等于当场的完整交易日且没有更晚的行；seed 42 和 123 都有 `best_model.pth` 和 `final_score.txt`；`ranking_full.csv` 按 `model_score` 降序，含代码、名称、分数、horizon、as-of；`result.csv` 的前五名和权重与排序前五行一致。

没有 `VERIFY_PASS.txt` 就还没有正式排序。
