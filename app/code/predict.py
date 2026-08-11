import json
import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def _find_seed_dir(base_dir, seed):
	"""定位模型目录: 优先 seed_{seed}/ 子目录，回退到 base_dir（兼容旧结构）"""
	seed_dir = os.path.join(base_dir, f'seed_{seed}')
	if os.path.isdir(seed_dir):
		return seed_dir
	return base_dir


def _find_model_root(output_dir):
	"""查找可用模型根目录: 若 output_dir 下无 scaler.pkl，自动回退到已有的模型目录"""
	# 检查 seed 子目录
	ensemble_seeds = config.get('ensemble_seeds', [42])
	for seed in ensemble_seeds:
		seed_dir = os.path.join(output_dir, f'seed_{seed}')
		if os.path.isdir(seed_dir):
			sc = os.path.join(seed_dir, 'scaler.pkl')
			if os.path.exists(sc):
				return output_dir

	# 检查根目录
	if os.path.exists(os.path.join(output_dir, 'scaler.pkl')):
		return output_dir

	# 回退：扫描 ./model/ 下其他子目录
	model_root = os.path.join(os.path.dirname(__file__), '..', 'model')
	if os.path.isdir(model_root):
		for d in sorted(os.listdir(model_root), reverse=True):
			dd = os.path.join(model_root, d)
			if not os.path.isdir(dd):
				continue
			for seed in ensemble_seeds:
				sc = os.path.join(dd, f'seed_{seed}', 'scaler.pkl')
				if os.path.exists(sc):
					print(f'自动回退模型目录: {output_dir} → {dd}')
					return dd
			if os.path.exists(os.path.join(dd, 'scaler.pkl')):
				print(f'自动回退模型目录: {output_dir} → {dd}')
				return dd

	raise FileNotFoundError(
		f'未找到任何可用的模型（含 scaler.pkl）。\n'
		f'  搜索: {output_dir}\n'
		f'  请先运行 train.py 完成训练。'
	)


def main():
	ensemble_seeds = config.get('ensemble_seeds', [42])
	data_file = os.path.join(config['data_path'], config.get('predict_file', 'test.csv'))
	output_path = os.path.join('./output/', 'result.csv')
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	output_dir = _find_model_root(config['output_dir'])

	# 从模型目录加载 config.json，覆盖当前 config 中的关键参数（sequence_length 等）
	model_config_path = os.path.join(output_dir, 'config.json')
	if os.path.exists(model_config_path):
		with open(model_config_path, 'r') as f:
			model_config = json.load(f)
		for key in ['sequence_length', 'd_model', 'nhead', 'num_layers',
					'dim_feedforward', 'dropout', 'feature_num', 'cross_stock_layers',
					'top_k_output', 'top_k_weights', 'predict_use_leader_pool']:
			if key in model_config:
				config[key] = model_config[key]
		print(f'从 {model_config_path} 加载模型参数 (seq_len={config["sequence_length"]})')
	else:
		# 回退: 尝试从 seed 子目录加载
		for seed in ensemble_seeds:
			sc = os.path.join(output_dir, f'seed_{seed}', 'config.json')
			if os.path.exists(sc):
				with open(sc, 'r') as f:
					model_config = json.load(f)
				for key in ['sequence_length', 'd_model', 'nhead', 'num_layers',
							'dim_feedforward', 'dropout', 'feature_num', 'cross_stock_layers',
							'top_k_output', 'top_k_weights', 'predict_use_leader_pool']:
					if key in model_config:
						config[key] = model_config[key]
				print(f'从 {sc} 加载模型参数 (seq_len={config["sequence_length"]})')
				break

	# 共享 artifacts 优先从根目录加载，找不到再从 seed 子目录找
	seed0_dir = _find_seed_dir(output_dir, ensemble_seeds[0])
	scaler_path = os.path.join(output_dir, 'scaler.pkl')
	if not os.path.exists(scaler_path):
		scaler_path = os.path.join(seed0_dir, 'scaler.pkl')
	stockid2idx_path = os.path.join(output_dir, 'stockid2idx.pkl')
	if not os.path.exists(stockid2idx_path):
		stockid2idx_path = os.path.join(seed0_dir, 'stockid2idx.pkl')
	medians_path = os.path.join(output_dir, 'train_medians.pkl')
	if not os.path.exists(medians_path):
		medians_path = os.path.join(seed0_dir, 'train_medians.pkl')
	leader_path = os.path.join(output_dir, 'leader_stock_ids.pkl')
	if not os.path.exists(leader_path):
		leader_path = os.path.join(seed0_dir, 'leader_stock_ids.pkl')

	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	if not os.path.exists(data_file):
		raise FileNotFoundError(f'未找到推理数据: {data_file}')
	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	if os.path.exists(stockid2idx_path):
		raw_map = joblib.load(stockid2idx_path)
		# 兼容旧映射中的短码（如 '1'），统一成 6 位
		stockid2idx = {str(sid).zfill(6): int(idx) for sid, idx in raw_map.items()}
		print(f'已加载训练时的股票映射，共 {len(stockid2idx)} 只股票')
	else:
		stock_ids = sorted(raw_df['股票代码'].unique())
		stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
		print(f'未找到训练时的股票映射，从数据重建，共 {len(stockid2idx)} 只股票')

	# 与训练同一股票宇宙：优先使用训练保存的龙头池
	stock_ids = sorted(stockid2idx.keys())
	if config.get('predict_use_leader_pool', True) and os.path.exists(leader_path):
		leader_stock_ids = {str(s).zfill(6) for s in joblib.load(leader_path)}
		stock_ids = [sid for sid in stock_ids if sid in leader_stock_ids]
		print(f'预测限制在训练龙头池: {len(stock_ids)} 只')
	elif config.get('predict_use_leader_pool', True):
		print('警告: 未找到 leader_stock_ids.pkl，回退为全市场预测')

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)

	if os.path.exists(medians_path):
		train_medians = joblib.load(medians_path)
		processed[features] = processed[features].fillna(train_medians)
		print('使用训练集中位数填充缺失值')
	else:
		processed[features] = processed[features].fillna(0.0)
		print('警告: 未找到训练集中位数文件，使用0填充缺失值')

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	# ---- 风险过滤：仅在候选池内排除极端波动股票 ----
	if 'volatility_20' in features:
		vol_idx = features.index('volatility_20')
		last_vol = sequences_np[:, -1, vol_idx]
		vol_mean = float(np.nanmean(last_vol))
		vol_std = float(np.nanstd(last_vol))
		threshold = vol_mean + 2.0 * vol_std
		keep = last_vol <= threshold
		# 至少保留足够股票用于 top_k 输出
		min_keep = max(config.get('top_k_output', 5) * 3, 20)
		if (not np.all(keep)) and int(keep.sum()) >= min_keep:
			n_excluded = int((~keep).sum())
			sequences_np = sequences_np[keep]
			sequence_stock_ids = [sid for i, sid in enumerate(sequence_stock_ids) if keep[i]]
			print(f'风险过滤: 排除 {n_excluded} 只极端波动股票 (阈值={threshold:.4f})，保留 {len(sequence_stock_ids)} 只')
		else:
			print(f'风险过滤: 跳过或无需排除 (阈值={threshold:.4f})，保留 {len(sequence_stock_ids)} 只')

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	# ---- 多模型集成：加载每个 seed 的模型，取评分均值 ----
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
			x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
			scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]
			all_scores.append(scores)
		loaded_seeds.append(seed)
		print(f'seed={seed} 模型已加载并预测')

	if not all_scores:
		raise FileNotFoundError('未找到任何可集成的模型 — 请先训练至少一个 seed')

	scores = np.mean(all_scores, axis=0)  # 多模型取均值
	print(f'集成模型: {loaded_seeds} ({len(loaded_seeds)} 个)')

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]

	# 评分最高的 top_k_output 只股票，按 top_k_weights 分配权重（评分第1名拿第1个权重，依此类推）
	top_k = config.get('top_k_output', 2)
	if len(ranked_stock_ids) < top_k:
		raise ValueError(f'可预测股票不足{top_k}只，当前仅有 {len(ranked_stock_ids)} 只')

	top_k_weights = config.get('top_k_weights', [1.0 / top_k] * top_k)  # 未配置时回退为等权重
	if len(top_k_weights) != top_k:
		raise ValueError(f'top_k_weights 长度({len(top_k_weights)})与 top_k_output({top_k})不一致')

	# 取评分最高的 top_k 只股票，按配置权重分配
	top_k_indices = order[:top_k]
	top_k_scores = scores[top_k_indices]  # 模型评分
	weights = list(top_k_weights)

	top_k_stocks = ranked_stock_ids[:top_k]
	output_df = pd.DataFrame({
		'stock_id': top_k_stocks,
		'weight': weights,
	})
	output_df.to_csv(output_path, index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'评分最高的{top_k}只股票:')
	for i, (sid, sc, w) in enumerate(zip(top_k_stocks, top_k_scores, weights)):
		print(f'  Top{i+1}: {sid}  评分={sc:.6f}  权重={w}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
