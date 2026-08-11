#!/usr/bin/env bash
# 赛事 docker-compose 挂载此目录后执行的默认流程。
set -euo pipefail

/app/init.sh

# 复现审核从训练开始：默认训练全部固定 seed，再基于新生成模型推理。
# 仅在本地已有模型的快速推理场景下才允许显式关闭训练。
if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
    /app/train.sh
fi

/app/test.sh
