# 可靠评估实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不覆盖 v4 和已完成研究输出的前提下，把持有期、数据模式、滚动折、轻量对照、账户账本和 abstain 决策做成可测试的最小协议。

**Architecture:** 新代码只放在 `research/reliability/`。旧的 `research/study.py` 和 `research/protocol.py` 保持 version 2 的行为。严格模式缺数据就失败；研究模式用现有后复权快照，但结论只能是 abstain。

**Tech Stack:** Python 3.12、pandas、numpy、scikit-learn、现有 PyTorch `StockTransformer`、BaoStock 只作为已经落盘的探测样本，测试不访问网络。

## Global Constraints

- 可比持有配置名只有 `open_t_plus_1_to_open_t_plus_5`，持有间隔是 4 个交易日。
- 不修改 v4 的 `app/code/train.py` 导出名，研究特征导入使用 `feature_cloums_map`。
- 不改 version 2 的日期、预算、成本和采用规则。
- 决策状态只有 `abstain` 和 `process_only`。研究模式永远 `abstain`。没有 `adopt`。
- 不声称 version 2 的区间是独立最终证据。
- CPU 不跑完整 80 epoch。
- 不伪造收益。

---

### Task 1: 持有期

**Files:**
- Create: `research/reliability/holding.py`
- Test: `tests/test_reliability_holding.py`

**Interfaces:**
- Produces: `resolve_holding(trading_days: list[str], signal_date: str, holding_id: str = "open_t_plus_1_to_open_t_plus_5") -> dict`
- Produces: `rebalance_signals(trading_days: list[str], first: str, last: str) -> list[str]`

- [ ] 写失败测试：t 的进场是 t+1 开盘，出场是 t+5 开盘，`holding_steps == 4`；非基线配置名抛出 `ValueError`；相邻两次调仓之间，出场日是空仓日，下一次进场是下一个交易日。
- [ ] 运行 `python3 -m pytest tests/test_reliability_holding.py -q`，确认因为模块不存在而失败。
- [ ] 实现最小函数。
- [ ] 再跑同一命令，确认通过。

### Task 2: 涨跌停和数据审计

**Files:**
- Create: `research/reliability/limits.py`
- Create: `research/reliability/audit.py`
- Test: `tests/test_reliability_limits.py`

**Interfaces:**
- Produces: `limit_status(code: str, is_st, ipo_date, trade_date: str, trading_days: list[str]) -> dict`
- Produces: `reject_reason(bar: dict, side: str, status: dict) -> str | None`
- Produces: `audit_frame(prices, membership, listings) -> dict`

- [ ] 写失败测试：主板 10.06 元的 10% 涨停价是 11.07；封板买入拒绝；缺 `isST` 返回 `rule_unverified`；后复权中文列的审计 `strict_ready is False`，且原因包含幸存者偏差和缺少不复权字段；探测样本的审计不能把后复权快照标成严格。
- [ ] 先看失败，再实现，再看通过。

### Task 3: 折、成本和账本

**Files:**
- Create: `research/reliability/folds.py`
- Create: `research/reliability/costs.py`
- Create: `research/reliability/ledger.py`
- Test: `tests/test_reliability_ledger.py`

**Interfaces:**
- Produces: `FOLDS`，`purge_train(signals: list[dict], train_end: str) -> list[dict]`，`clip_bounds(labels: list[float]) -> tuple[float, float]`
- Produces: `charge(notional: float, side: str, multiplier: int = 1) -> float`
- Produces: `run_account(bars, signals, targets, multiplier: int = 1) -> dict`

- [ ] 写失败测试：出场日晚于训练截止日的标签被 purge；裁剪边界只用训练标签。账本包含空仓日；100 股一手；停牌不成交；费用可按 2 倍压力计算；同日买卖被拒绝。
- [ ] 先看失败，再实现，再看通过。

### Task 4: 对照、决策和前瞻接口

**Files:**
- Create: `research/reliability/baselines.py`
- Create: `research/reliability/decision.py`
- Create: `research/reliability/prospective.py`
- Test: `tests/test_reliability_decision.py`

**Interfaces:**
- Produces: `fit_linear(x, y)`，`fit_tree(x, y, seed: int)`，`equal_weights(codes: list[str]) -> dict[str, float]`
- Produces: `decide(mode: str, strict_ready: bool, dev_reports: list[dict]) -> str`
- Produces: `freeze_config(path, commit: str) -> dict`，`archive_predictions(path, rows: list[dict], audit: dict) -> None`

- [ ] 写失败测试：线性模型和树模型在小样本上给出有限预测；决策函数不读取测试折字段；两个种子符号相反时 `abstain`；研究模式即使开发验证超额为正也 `abstain`；归档在审计未通过时拒绝把 `strict=true` 写入。
- [ ] 先看失败，再实现，再看通过。

### Task 5: 真实数据烟测和笔记本

**Files:**
- Create: `research/reliability/fixtures/baostock_pit_probe_2026-11-10.json`
- Create: `research/reliability/fixtures/SOURCE.json`
- Create: `research/reliability/smoke.py`
- Test: `tests/test_reliability_smoke.py`
- Create: `kaggle/generate_reliability_notebook.py`
- Create: `kaggle/RELIABILITY_RUN.md`

**Interfaces:**
- Produces: `run_smoke(panel_path: Path, probe_path: Path, output_dir: Path) -> dict`

- [ ] 写失败测试：16 只真实快照股票经过真实 `build_samples` 和 `feature_cloums_map`；等权、线性、树和 Transformer 一步都有有限输出；研究模式结论是 `abstain`；逐日净值和逐笔文件存在；同一输出目录不写 `result.csv`。探测样本里的不复权行情能让严格规则拒绝封板，而不是退回后复权。
- [ ] 先看失败，再实现烟测入口。
- [ ] 跑 `python3 -m pytest tests/test_reliability_holding.py tests/test_reliability_limits.py tests/test_reliability_ledger.py tests/test_reliability_decision.py tests/test_reliability_smoke.py tests/test_windows.py tests/test_memory_block.py tests/test_backtest.py tests/test_study_smoke.py tests/test_study_entry.py -q`。
- [ ] 提交代码后，用该 commit 生成新笔记本。笔记本沿用 baostock 0.9.4、不 reload torch、`PYTHONPATH` 由仓库根运行 `python -m research.reliability.smoke`。默认不带 `--full`。

## Self-review

- 规格里的持有期、严格模式阻塞、三折、对照、账本、abstain、前瞻接口都有对应任务。
- 没有把完整 689 天下载写成已经完成。
- 决策函数的输入只有开发验证报告，测试折不进入函数签名。
