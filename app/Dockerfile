FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# 依赖在构建镜像时安装；容器运行训练和推理时不下载安装任何包。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        python3 \
        python-is-python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN python3 -m pip install --no-cache-dir --break-system-packages uv \
    && uv sync --frozen --no-dev --no-install-project \
    && python3 -c "import talib, torch; print(f'TA-Lib={talib.__version__}, Torch={torch.__version__}')" \
    && rm -rf /root/.cache

COPY code /app/code
COPY init.sh train.sh test.sh readme.md /app/
COPY data /app/data
COPY model /app/model
RUN mkdir -p /app/data /app/model /app/output /app/temp \
    && chmod 755 /app/init.sh /app/train.sh /app/test.sh

CMD ["sleep", "infinity"]
