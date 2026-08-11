#!/usr/bin/env bash
# 赛事 docker-compose 挂载此目录后执行的默认流程。
set -euo pipefail

/app/init.sh

# 最终提交镜像须已包含 app/model 中的训练产物，因此默认仅推理。
# 复现训练时显式设置 RUN_TRAIN=1。
if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
    /app/train.sh
fi

/app/test.sh
