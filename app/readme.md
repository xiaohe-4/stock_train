# 代码说明

本项目为 2026 中国高校计算机大赛——大数据挑战赛的排序选股方案：在沪深 300 股票截面中，根据历史量价数据预测未来 5 个交易日收益靠前的股票，并输出不超过 5 只股票及其权重。

本目录是最终提交内容，可直接构建名为 `bdc2026` 的 Docker 镜像。模型代码位于 `code/`，完整算法说明包含在本 README 中。

## 环境配置

- Python：`>=3.10, <3.13`
- PyTorch：`>=2.6.0`（Linux/Windows 使用 CUDA 12.8 源）
- pandas：`>=2.3.2`；scikit-learn：`>=1.7.2`；joblib：`>=1.5.2`
- tensorboard：`>=2.20.0`；tensorboardX：`>=2.6.4`
- 其他依赖：akshare、baostock、ta-lib、tqdm 等，完整列表见 `pyproject.toml`

```bash
cd app
uv sync
```

训练自动优先使用 CUDA，无 CUDA 时回退至 MPS 或 CPU。

### Docker 提交与离线运行

提交镜像不在运行时安装依赖或下载模型/数据。准备好 `app/data/train.csv`、`app/data/test.csv` 后，在仓库根目录执行：

```bash
docker build -t bdc2026 ./app
docker save -o 队伍名称.tar bdc2026
```

导出的镜像名称必须为 `bdc2026`，提交文件命名为 `队伍名称.tar`，且不应再次压缩。示例 `docker-compose.yml` 会挂载 `app/data`、`app/output`、`app/temp` 到容器对应目录；镜像内保留 `code`、`model`、`init.sh`、`train.sh`、`test.sh`。

导出最终镜像前，必须执行一次训练，或将已经验证过的训练产物复制到 `app/model/`。其中至少应包含各 seed 的 `best_model.pth`、`scaler.pkl`、`stockid2idx.pkl`、`train_medians.pkl`、`leader_stock_ids.pkl` 与 `config.json`。Dockerfile 会将该目录复制进镜像；缺失这些文件时，`test.sh` 会明确报错，不能用于最终提交。

## 数据

### 数据与辅助数据

- 训练数据为赛事提供或按赛事格式整理的沪深 300 历史日频量价数据，默认路径为 `data/train.csv`；推理数据为挂载的 `data/test.csv`；
- 数据包含股票代码、日期、开盘/收盘/最高/最低价、成交量、成交额、换手率等字段；
- `data/stock_data.csv` 是原始汇总数据；`data/split_train_test.py` 可按日期切分为 `train.csv` 和 `test.csv`；
- `get_stock_data.py` 与根目录 `get_stock/` 用于更新量价数据与生成周度统计。

未使用赛事数据以外的公开辅助数据、新闻数据、财务数据或人工标注数据。技术指标完全由当前及历史量价数据计算，不引入未来时点信息。

### 特征与标签

默认特征为 `158+39`：158 个 Alpha158 风格量价因子（K 线、动量、波动、量能及多周期统计）加 39 个技术指标（均线、RSI、MACD、KDJ、布林带、ATR、多周期收益与价差）。

每只股票使用过去 30 个交易日的特征序列。监督标签为未来 5 日开盘到开盘收益：

```text
未来 5 日收益 r_i = (第 t+5 日开盘价 - 第 t+1 日开盘价) / 第 t+1 日开盘价
```

标签截断在训练集 1%～99% 分位数之间，以减弱异常价格的影响。

## 预训练模型

未使用预训练模型。`code/model.py` 的 `StockTransformer` 采用 Xavier 随机初始化，从头端到端训练。

## 算法

### 整体思路介绍

将任务建模为同一交易日股票截面上的 **listwise 排序**，而非逐股收益回归。模型输出每只股票的分数，按分数取 Top-5 并按配置分配权重。

```text
历史量价 → 因子工程 → 龙头池筛选与序列构造
        → 时序 Transformer → 股票间横截面注意力 → 排序分数
        → TopK-ListMLE + 软组合收益 → 双 seed 集成 → Top-5 输出
```

### 针对性问题解决方案与创新点

1. **Top-5 目标对齐**：当前方案自定义权重为 `[0.50, 0.47, 0.01, 0.01, 0.01]`，主动重仓前 2 名；因此采用 TopK-ListMLE（`K=3`）专注头部排序。该权重不是赛事官方给定参数，应根据最终提交规则调整 `config.py` 中的 `top_k_weights`。
2. **可微组合收益**：对预测分数低温 softmax，近似重仓高分股票，并直接最大化该软组合的真实未来收益。
3. **时序与横截面联合建模**：Transformer 提取个股历史状态；Cross-Stock Attention 学习同日股票间相对强弱关系。
4. **龙头池降噪**：仅保留训练集最近 30 日平均成交额前 75% 的股票，降低低流动性股票的噪声；验证与推理复用同一股票宇宙。
5. **padding 掩码与双 seed 集成**：无效补齐股票不参与横截面注意力；对 seed 42 与 123 的结果取均值降低方差。

### 网络结构

`code/model.py` 中的 `StockTransformer` 依次包括：

1. 输入 `[batch, 股票数, 30, 特征数]` 的线性投影与正弦位置编码；
2. 两层 Transformer Encoder，提取每只股票的时序表征；
3. `FeatureAttention`，在时间维进行可学习加权聚合；
4. 两层 `CrossStockAttention`（多头注意力、残差、LayerNorm），建模横截面关系，并通过 mask 屏蔽 padding；
5. Ranking MLP 与 score head，输出每只股票的标量排序分数。

### 损失函数

训练目标为：

```text
总损失 = TopK-ListMLE 损失 - λ × 平均软组合收益
软组合收益 = Σ[softmax(预测分数 / 温度) × 真实未来收益]
```

- `TopK-ListMLE`：按真实收益降序，仅计算前 `K=3` 名的 Plackett–Luce 负对数似然；每一步分母包含全体剩余股票，保证从完整截面中选对头部股票；
- 软组合收益项：温度 `T=0.35`，权重 `λ=0.4`；
- 优化器：AdamW，配合梯度裁剪、5 epoch warmup 和余弦退火。

### 数据扩增

未使用会破坏金融时间顺序的样本复制、打乱或翻转增强。仅进行训练集标准化、训练集列中位数填充缺失值和极端标签截断。

### 模型集成

用 `ensemble_seeds=[42, 123]` 独立训练两次；推理时对两个模型的原始排序分数取均值，再选择 Top-5。

### 算法的其他细节

- `d_model=128`、4 注意力头、时序层数 2、横截面注意力层数 2；
- `dropout=0.35`、`weight_decay=2e-3`，抑制金融短周期数据上的过拟合；
- 同时记录 Top-5 等权指标和加权组合指标；
- 最优模型按 `weighted_port_return` 保存，与当前自定义组合权重下的绝对组合收益一致。

## 训练流程

在容器内的 `/app`（或本目录）下执行：

```bash
python code/train.py
# 或 bash train.sh
```

`code/train.py` 的流程：

1. 固定随机种子，读取 `data/train.csv`；
2. 按时间划训练集与最近约 2 个月验证集，为验证首个样本保留 30 日上下文；
3. 仅用训练集最近 30 日平均成交额建立龙头池，验证集同步筛选；
4. 并行生成技术因子，构造未来 5 日收益标签；
5. 使用训练集拟合中位数填充器及 `StandardScaler`，并应用到验证集；
6. 按日期将全部有效股票组成一个排序样本，`collate_fn` 补齐变长截面；
7. 对两个 seed 分别训练，采用联合排序损失反向传播；
8. 每 epoch 在验证集计算加权组合收益，保存最佳 `best_model.pth`；连续 12 个 epoch 无提升则早停。

模型、标准化器、股票索引、龙头池及配置保存在 `model/<sequence_length>_<feature_num>/seed_<seed>/`。训练脚本以 7 小时 50 分钟为上限，满足 8 小时审核时限并保留缓冲。

## 推理流程

训练完成后，在容器内的 `/app`（或本目录）下执行：

```bash
python code/predict.py
# 或 bash test.sh
```

`code/predict.py` 的流程：

1. 加载训练保存的配置、中位数、标准化器、股票映射及龙头池；
2. 读取 `config['data_path']/test.csv` 截至最新交易日的数据，构造每只股票最近 30 日序列；
3. 按配置应用龙头池；若龙头池文件缺失则回退为全市场预测；
4. 加载每个 seed 的 `best_model.pth` 并平均排序分数；
5. 选 Top-5，按 `config.py` 中自定义的 `top_k_weights`（默认 `[0.50, 0.47, 0.01, 0.01, 0.01]`）分配权重；
6. 输出 `output/result.csv`，包含 `stock_id` 与 `weight` 两列。

若需对新的赛事测试文件推理，应保证字段与训练数据一致，并将其作为 `config['data_path']/test.csv` 挂载；也可通过 `config.py` 的 `predict_file` 修改文件名。

## 其他注意事项

- 数据必须按股票代码和日期升序排列，否则基于 `shift(-1)`、`shift(-5)` 的标签会错位；
- 龙头池只能用训练期数据构建，不能使用验证期或待预测期数据，否则会泄露未来信息；
- 标准化器、缺失值中位数和股票代码映射必须复用训练阶段文件，不能在测试数据上重新拟合；
- 推理股票至少需要 30 个交易日历史；不足窗口者会被跳过；
- 原有训练逻辑固定 Python、NumPy、PyTorch/CUDA 随机源，并设置 `cudnn.deterministic=True`、`cudnn.benchmark=False`；未改变模型结构、损失、超参或既有 seed 集成方式。复现审核应在相同数据与配置下完整训练两次，比较 `weighted_port_return` 的绝对差，要求不超过 `0.002`；
- 推理脚本以 295 秒为硬时限，满足 5 分钟审核要求；训练和推理过程不调用网络接口；
- 市场非平稳且周度涨幅噪声很高，本项目输出的是算法预测，不构成投资建议。
