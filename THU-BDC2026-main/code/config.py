# 配置参数
sequence_length = 30
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'batch_size': 8,
    'num_epochs': 80,
    'learning_rate': 2e-4,
    'dropout': 0.35,
    'feature_num': feature_num,
    'max_grad_norm': 1.0,
    'use_clip': True,
    'warmup_epochs': 5,
    'weight_decay': 2e-3,

    'cross_stock_layers': 2,
    'early_stopping_patience': 12,

    'seed': 42,
    # 双 seed 集成，预测时取均值
    'ensemble_seeds': [42, 123],

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',

    # 85% 测试周变差，回调到 0.75 折中
    'train_leader_ratio': 0.75,
    'leader_lookback_days': 30,
    'predict_use_leader_pool': True,

    # 权重不变
    'top_k_output': 5,
    'top_k_weights': [0.50, 0.47, 0.01, 0.01, 0.01],
    # 更贴近重仓 Top2/Top3
    'topk_mle_k': 3,
    # ListMLE + 可微组合收益辅助项
    'portfolio_loss_weight': 0.4,
    'portfolio_temperature': 0.35,
    # 选模主指标：绝对加权收益（与赛方评分同口径）
    'selection_metric': 'weighted_port_return',
}
