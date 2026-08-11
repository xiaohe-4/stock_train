#!/usr/bin/env bash
# 审核硬件的预测时限为 5 分钟；保留 5 秒缓冲。
set -euo pipefail

cd /app
timeout --signal=TERM --kill-after=5s 295s python /app/code/test.py
