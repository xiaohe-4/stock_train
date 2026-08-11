#!/usr/bin/env bash
# 审核硬件的训练时限为 8 小时；保留 10 分钟缓冲。
set -euo pipefail

cd /app
timeout --signal=TERM --kill-after=60s 28200s python /app/code/train.py
