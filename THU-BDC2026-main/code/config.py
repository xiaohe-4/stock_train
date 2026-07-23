# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 8,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 80,       # 排序任务可能需要更多epochs
    'learning_rate': 5e-4,  # 配合warmup使用较高的学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 1.0,
    'use_clip': True,       # 启用梯度裁剪
    'warmup_epochs': 10,    # 学习率预热轮数

    'pairwise_weight': 1.0,           # 配对损失权重
    'base_weight': 1.0,               # 非top-k样本权重
    'top5_weight': 3.0,               # top-5样本权重（加大，突出选股目标）
    'topk_loss_weight': 0.5,          # soft top-k 收益对齐权重
    'loss_temperature': 0.5,          # soft top-k 温度（越小越接近 hard top-k）
    'listwise_temperature': 1.0,      # ListNet 温度

    'cross_stock_layers': 3,          # 股票间交互注意力层数（已接入模型）
    'recent_bias_strength': 1.5,      # 时序聚合的近期偏置强度
    'early_stopping_patience': 15,    # Early Stopping 容忍轮数
    'val_months': 2,                  # 验证集使用最近 N 个月

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
