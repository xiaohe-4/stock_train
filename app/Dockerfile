FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# 依赖在构建镜像时安装；容器运行训练和推理时不下载安装任何包。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project \
    && rm -rf /root/.cache

COPY code /app/code
COPY init.sh train.sh test.sh readme.md /app/
RUN mkdir -p /app/data /app/model /app/output /app/temp \
    && chmod 755 /app/init.sh /app/train.sh /app/test.sh

ENTRYPOINT ["/app/init.sh"]
