# 内部 LSTM 记忆与研究回测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在第二层时间 Transformer block 内加入真正的 LSTM `(h_t, c_t)` 记忆，并准备三臂从头训练的 Kaggle 研究回测；不覆盖 v4 排名。

**Architecture:** 新代码放在 `research/`，不改 `app/code` 的 v4 哈希。窗口构造去掉“标签已删除后再丢 5 行”。三臂共用该数据和 `app/code/config.py` 的优化设置。选模只用开发验证日的平均超额（不复利）。可交易净值预先写成每 5 个交易日一次的非重叠调仓，步进净值，不用重叠 5 日收益计算日夏普。

**Tech Stack:** PyTorch、pandas、pytest。Kaggle 笔记本沿用已验证的 pip 策略和 `PYTHONPATH`，锁定包含 `research/` 的新提交。

## Global Constraints

- 只改第二层时间 block：注意力之后接 `nn.LSTM` 的完整序列、残差投影、再前馈。隐藏宽度 64。
- 单 score head。禁止外部 LSTM 再接 Transformer，禁止把两臂分数平均。
- 参数量必须在 389,614 到 527,126。单 LSTM 臂隐藏宽度 192，若实测越界只在开训前改宽度并写进 `research/protocol.py`。
- 种子只允许 42 和 123。最多 80 轮，patience 12。学习率 `2e-4`，warmup 5，权重衰减 `2e-3`。
- 示例成本 `0.0010 * 0.5 * L1(换手)`。不是真实全部成本。
- 开发验证窗口结束日 `2026-06-01` 至 `2026-08-25`。训练窗口结束日不晚于 `2026-05-22`。
- `2026-09-09` 至 `2026-10-26` 只作附录，不能决定采用，也不能称为 v4 的未见 holdout。
- 采用规则只比较三臂在开发验证上的平均超额。不因净值好坏改规则。
- 面板 as-of 固定 `2026-11-09`，300 只，205,626 行，sha256 `1bd7ff55788340b864e2f8c4d0c70f7ce7f344d181103fba4e85fee7a13526f2`。不用实时日期 guard。
- 后复权价格不能可靠判断涨跌停。停牌用成交量缺失或 0。涨跌停函数只在未复权合成数据上测试，正式净值不假装已执行涨跌停。
- T+1 由“信号日 t 的下一交易日开盘进入、第 5 个交易日开盘退出”体现。
- CPU 只做单元测试和一步烟测，不跑完整训练。

---

### Task 1: 窗口修复

**Files:**
- Create: `research/windows.py`
- Test: `tests/test_windows.py`

**Interfaces:**
- Produces: `legacy_drops_extra_five(n_labeled, end_idx) -> bool`；`fixed_window_end_indices(n_labeled, sequence_length) -> list[int]`

- [ ] 先写测试：40 日去掉最后 5 个标签后 `n_labeled=35`，`sequence_length=30`。旧谓词丢掉结束下标 34；新谓词保留 34。
- [ ] 运行 pytest 确认失败。
- [ ] 实现两个函数。
- [ ] 再运行 pytest 确认通过。

### Task 2: 三个模型

**Files:**
- Create: `research/models.py`
- Test: `tests/test_memory_block.py`

**Interfaces:**
- Produces: `TemporalLSTMMemoryBlock`，`build_model(arm, input_dim, num_stocks, config) -> nn.Module`，`count_parameters(model) -> int`，`benchmark_step(model, batch) -> dict`

- [ ] 测试形状 `[B,N,30,F] -> [B,N]`、LSTM 四门与 `c_t`、时间轴不串股票、batch 内状态独立、三臂参数量落在预算、一步反向梯度有限。
- [ ] 先看测试失败，再实现。`transformer` 使用现有 `StockTransformer`。`memory` 只替换第二层。`lstm` 用隐藏 192 的单层 LSTM 接原池化和跨股票注意力。

### Task 3: 非重叠净值与约束

**Files:**
- Create: `research/backtest.py`
- Create: `research/protocol.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `select_nonoverlapping(dates) -> list`；`period_nav(records) -> dict`；`mean_excess(records, split) -> float`；`decide(dev_means) -> str`；`entry_blocked(volume) -> bool`；`top_k_jaccard(a, b, k=5) -> float`

- [ ] 测试 10 个连续信号日、每日 5 日收益 0.01 时，净值只复利 2 段，夏普样本数为 2，并标记年化不稳。
- [ ] 测试附录分区不改变 `mean_excess(..., "dev")`。`decide` 不读取附录。
- [ ] 测试成交量为 0 时不能买入。涨跌停只在显式未复权样例上拦截封板。

### Task 4: 研究入口与笔记本

**Files:**
- Create: `research/study.py`
- Create: `kaggle/generate_research_notebook.py`
- Test: `tests/test_study_smoke.py`

- [ ] 烟测在随机小张量上对三臂各做一步，写出梯度范数和耗时，不读面板。
- [ ] 笔记本固定数据集路径、固定 as-of、校验 sha 与行数，克隆新提交，沿用 baostock `0.9.4` 与 torch 子进程探针。
- [ ] 生成 `kaggle/hs300_temporal_memory_research.ipynb` 并复制到 `/opt/cursor/artifacts/`。

### Task 5: 验证

- [ ] `python3 -m pytest tests/test_windows.py tests/test_memory_block.py tests/test_backtest.py tests/test_study_smoke.py -q`
- [ ] 不运行完整三臂训练。
