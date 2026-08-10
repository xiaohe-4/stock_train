"""
打印当前模型对所有股票的评分排名前30的结果（仅打印，不写文件）。
复用 predict.py 里已经写好的加载/预处理/集成预测逻辑，保持口径一致
（包括风险过滤：剔除近20日波动率超过2σ的极端股票）。

用法:
    python print_top30.py          # 默认打印前30
    python print_top30.py 50       # 也可以传参数指定打印前N只
"""
import os
import sys
import json
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch

from config import config
from model import StockTransformer
from predict import (
	preprocess_predict_data,
	build_inference_sequences,
	_find_seed_dir,
	_find_model_root,
)


def main(top_n=30):
	ensemble_seeds = config.get('ensemble_seeds', [42])
	data_file = os.path.join(config['data_path'], 'train.csv')
	output_dir = _find_model_root(config['output_dir'])

	# 从模型目录加载训练时的 config.json，保证 sequence_length 等参数与训练一致
	model_config_path = os.path.join(output_dir, 'config.json')
	if os.path.exists(model_config_path):
		with open(model_config_path, 'r') as f:
			model_config = json.load(f)
	else:
		model_config = None
		for seed in ensemble_seeds:
			sc = os.path.join(output_dir, f'seed_{seed}', 'config.json')
			if os.path.exists(sc):
				with open(sc, 'r') as f:
					model_config = json.load(f)
				break
	if model_config:
		for key in ['sequence_length', 'd_model', 'nhead', 'num_layers',
					'dim_feedforward', 'dropout', 'feature_num']:
			if key in model_config:
				config[key] = model_config[key]

	# 共享 artifacts：根目录优先，找不到再去第一个 seed 子目录找
	def _resolve(name):
		path = os.path.join(output_dir, name)
		if os.path.exists(path):
			return path
		return os.path.join(_find_seed_dir(output_dir, ensemble_seeds[0]), name)

	scaler_path = _resolve('scaler.pkl')
	stockid2idx_path = _resolve('stockid2idx.pkl')
	medians_path = _resolve('train_medians.pkl')

	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	if os.path.exists(stockid2idx_path):
		stockid2idx = joblib.load(stockid2idx_path)
	else:
		stock_ids = sorted(raw_df['股票代码'].unique())
		stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
	stock_ids = sorted(stockid2idx.keys())

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)

	if os.path.exists(medians_path):
		train_medians = joblib.load(medians_path)
		processed[features] = processed[features].fillna(train_medians)
	else:
		processed[features] = processed[features].fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed, features, sequence_length, stock_ids, latest_date,
	)

	# 风险过滤：与 predict.py 保持一致，排除近20日波动率超过2σ的极端股票
	if 'volatility_20' in features:
		vol_idx = features.index('volatility_20')
		last_vol = sequences_np[:, -1, vol_idx]
		vol_mean = float(np.nanmean(last_vol))
		vol_std = float(np.nanstd(last_vol))
		threshold = vol_mean + 2.0 * vol_std
		keep = last_vol <= threshold
		if not np.all(keep):
			n_excluded = int((~keep).sum())
			sequences_np = sequences_np[keep]
			sequence_stock_ids = [sid for i, sid in enumerate(sequence_stock_ids) if keep[i]]
			print(f'风险过滤: 排除 {n_excluded} 只极端波动股票，保留 {len(sequence_stock_ids)} 只')

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stockid2idx))
	model.to(device)
	model.eval()

	all_scores = []
	loaded_seeds = []
	for seed in ensemble_seeds:
		seed_dir = _find_seed_dir(output_dir, seed)
		model_path = os.path.join(seed_dir, 'best_model.pth')
		if not os.path.exists(model_path):
			print(f'警告: seed={seed} 模型不存在，跳过')
			continue
		model.load_state_dict(torch.load(model_path, map_location=device))
		with torch.no_grad():
			x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
			scores = model(x).squeeze(0).detach().cpu().numpy()
			all_scores.append(scores)
		loaded_seeds.append(seed)

	if not all_scores:
		raise FileNotFoundError('未找到任何可集成的模型 — 请先训练至少一个 seed')

	scores = np.mean(all_scores, axis=0)
	order = np.argsort(scores)[::-1]
	top_n = min(top_n, len(order))

	print(f'\n集成模型: {loaded_seeds} ({len(loaded_seeds)} 个)')
	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(sequence_stock_ids)}')
	print(f'\n评分最高的前 {top_n} 只股票:')
	print(f'{"排名":<6}{"股票代码":<12}{"评分":>12}')
	for rank, idx in enumerate(order[:top_n], start=1):
		print(f'{rank:<6}{sequence_stock_ids[idx]:<12}{scores[idx]:>12.6f}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
	main(top_n=n)
