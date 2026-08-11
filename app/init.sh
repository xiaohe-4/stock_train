#!/usr/bin/env bash
# Docker 容器入口：从挂载数据开始完整复现训练和推理。
set -euo pipefail

cd /app
mkdir -p /app/model /app/output /app/temp

for required_file in /app/data/train.csv /app/data/test.csv; do
    if [[ ! -s "${required_file}" ]]; then
        echo "缺少必需的挂载数据文件: ${required_file}" >&2
        exit 2
    fi
done

# 禁止运行时网络依赖；训练与推理所需的包已在镜像构建期安装。
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "开始确定性训练..."
/app/train.sh
echo "开始推理..."
/app/test.sh

test -s /app/output/result.csv
echo "复现完成: /app/output/result.csv"
