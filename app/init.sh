#!/usr/bin/env bash
# Docker 初始化：由 docker-compose 挂载 data 后调用。
set -euo pipefail

cd /app
mkdir -p /app/model /app/output /app/temp

echo "初始化完成：代码与模型位于镜像内，data/output/temp 由外部挂载。"
