"""Frozen research settings. Do not change these after seeing backtest results."""

from __future__ import annotations

ASOF = "2026-11-09"
PANEL_ROWS = 205626
PANEL_STOCKS = 300
PANEL_SHA256 = "1bd7ff55788340b864e2f8c4d0c70f7ce7f344d181103fba4e85fee7a13526f2"

TRAIN_WINDOW_END = "2026-05-22"
DEV_VAL_START = "2026-06-01"
DEV_VAL_END = "2026-08-25"
APPENDIX_START = "2026-09-09"
APPENDIX_END = "2026-10-26"

SEEDS = (42, 123)
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-3
WARMUP_EPOCHS = 5
BATCH_SIZE = 8
WEIGHTS = (0.50, 0.47, 0.01, 0.01, 0.01)
EXAMPLE_COST_RATE = 0.0010
HOLDING_TRADING_DAYS = 5
PARAM_MIN = 389614
PARAM_MAX = 527126
BASELINE_PARAMETERS = 458370
LSTM_HIDDEN = 192
MEMORY_HIDDEN = 64
LEADER_RATIO = 0.75
LEADER_LOOKBACK = 30

SELECTION_RULE = "mean_overlapping_excess_not_compounded"
NAV_RULE = "nonoverlapping_every_5_trading_days"
ADOPT_USES = "selection_rule_only"
HFQ_LIMIT_NOTE = (
    "面板是后复权。涨跌停价带不能从后复权价格可靠还原，正式净值不执行涨跌停拒单。"
    "停牌按成交量缺失或为 0 处理。T+1 体现在下一交易日开盘才进入。"
)
