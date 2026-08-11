import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    # 截断极端收益率，防止梯度不稳定
    lower = processed['label'].quantile(0.01)
    upper = processed['label'].quantile(0.99)
    processed['label'] = processed['label'].clip(lower=lower, upper=upper)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# Top-K ListMLE: 只优化收益率最高的 K 只股票的排序
# 预期初始 loss ≈ log(K!)（K=10 时约 15），远优于全量 ListMLE 的 log(N!)≈363
class TopKListMLELoss(nn.Module):
    """
    Top-K List-wise Maximum Likelihood Estimation.
    只对收益率最高的 top_k 只股票做 Plackett-Luce 排序优化，
    分母始终用全量股票，确保"选对"top-k 的约束。
    
    y_pred: [batch, N] 模型输出的原始评分
    y_true: [batch, N] 真实未来收益率（越高越好）
    """
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        N = y_pred.size(1)
        k = min(self.k, N)
        # 按真实收益率降序排序
        _, indices = torch.sort(y_true, dim=1, descending=True)
        sorted_pred = y_pred.gather(1, indices)  # [batch, N]
        # 只对 top-k 计算 loss，但分母用全部剩余股票（含 top-k 之外的）
        topk_pred = sorted_pred[:, :k]  # [batch, K]
        log_cumsum = torch.logcumsumexp(sorted_pred.flip(1), dim=1).flip(1)[:, :k]  # [batch, K]
        # loss = -Σ_i [s_i - log Σ_{j∈remaining} exp(s_j)]
        loss = -(topk_pred - log_cumsum).sum(dim=1).mean()
        return loss


class CombinedRankingLoss(nn.Module):
    """
    TopK-ListMLE + 可微软组合收益。
    软组合用低温 softmax 近似“重仓高分股票”，直接拉高组合期望收益。
    """
    def __init__(self, k=3, portfolio_weight=0.4, temperature=0.35):
        super().__init__()
        self.listmle = TopKListMLELoss(k=k)
        self.portfolio_weight = float(portfolio_weight)
        self.temperature = max(float(temperature), 1e-3)

    def forward(self, y_pred, y_true):
        mle = self.listmle(y_pred, y_true)
        weights = torch.softmax(y_pred / self.temperature, dim=1)
        port_ret = (weights * y_true).sum(dim=1).mean()
        return mle - self.portfolio_weight * port_ret

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算评估指标：Top-K 等权收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top K
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top K (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random K (Expected Value)
        random_return_sum = k * valid_true.mean().item()
        
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics


def calculate_weighted_portfolio_metrics(y_pred, y_true, masks, weights):
    """
    按提交权重评估组合收益（与 predict.py 的 top_k_weights 对齐）。
    weights 按模型评分从高到低依次分配。
    """
    weights = list(weights)
    k = len(weights)
    if k == 0:
        return {
            'weighted_port_return': 0.0,
            'weighted_oracle_return': 0.0,
            'weighted_random_return': 0.0,
            'weighted_final_score': 0.0,
        }

    w = torch.tensor(weights, dtype=y_pred.dtype, device=y_pred.device)
    batch_size = y_pred.size(0)
    port_list, oracle_list, random_list, score_list = [], [], [], []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        if valid_indices.numel() < k:
            continue
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        _, true_indices = torch.topk(valid_true, k)

        port_ret = (valid_true[pred_indices] * w).sum().item()
        oracle_ret = (valid_true[true_indices] * w).sum().item()
        # 权重和为 1 时，随机组合期望收益 = 截面均值
        random_ret = valid_true.mean().item()
        denom = oracle_ret - random_ret
        final_score = (port_ret - random_ret) / (denom + 1e-12) if abs(denom) > 1e-6 else 0.0

        port_list.append(port_ret)
        oracle_list.append(oracle_ret)
        random_list.append(random_ret)
        score_list.append(final_score)

    return {
        'weighted_port_return': float(np.mean(port_list)) if port_list else 0.0,
        'weighted_oracle_return': float(np.mean(oracle_list)) if oracle_list else 0.0,
        'weighted_random_return': float(np.mean(random_list)) if random_list else 0.0,
        'weighted_final_score': float(np.mean(score_list)) if score_list else 0.0,
    }


def _build_batch_metrics(masked_outputs, masked_targets, masks):
    """统一组装训练/验证日志指标。"""
    eval_k = config.get('top_k_output', 5)
    weights = config.get('top_k_weights', [1.0 / eval_k] * eval_k)
    metrics_k5 = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
    metrics_keval = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=eval_k)
    metrics = {f'{name}_k5': v for name, v in metrics_k5.items()}
    metrics.update({f'{name}_k{eval_k}': v for name, v in metrics_keval.items()})
    metrics.update(calculate_weighted_portfolio_metrics(masked_outputs, masked_targets, masks, weights))
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(self.relevance_scores[idx]),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(sequences, masks)  # [batch, max_stocks] 预测分数，masks 用于屏蔽跨股票注意力中的 padding 股票
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和真实收益率
            valid_pred = masked_outputs[i][valid_indices]
            valid_true = masked_targets[i][valid_indices]
            
            if len(valid_pred) > 1:
                # ListMLE: 直接对真实收益率排序
                loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if config.get('use_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标（含与提交权重对齐的 weighted_final_score）
            with torch.no_grad():
                metrics = _build_batch_metrics(masked_outputs, masked_targets, masks)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs = model(sequences, masks)  # masks 用于屏蔽跨股票注意力中的 padding 股票
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    # ListMLE: 直接对真实收益率排序
                    loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            metrics = _build_batch_metrics(masked_outputs, masked_targets, masks)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def filter_leader_stocks(df, leader_ratio=0.40, lookback_days=30, volume_col='成交额'):
    """
    龙头股筛选：按近 lookback_days 个交易日的平均成交额排名，
    保留前 leader_ratio 比例的股票作为龙头股训练集。
    
    返回: 筛选后的 DataFrame（仅保留龙头股数据）
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 取每只股票最近 lookback_days 天的数据
    recent = df.groupby('股票代码').tail(lookback_days)

    # 计算每只股票近 lookback_days 天的平均成交额
    avg_volume = recent.groupby('股票代码')[volume_col].mean().sort_values(ascending=False)

    # 取前 leader_ratio
    cutoff = max(1, int(len(avg_volume) * leader_ratio))
    leader_stock_ids = avg_volume.head(cutoff).index.tolist()

    print(f"龙头股筛选: 全量 {len(avg_volume)} 只 → 保留前 {cutoff} 只 (比例={leader_ratio})")
    print(f"龙头股top5: {leader_stock_ids[:5]}")
    print(f"龙头股平均成交额范围: {avg_volume.iloc[cutoff-1]:.0f} ~ {avg_volume.iloc[0]:.0f}")

    # 恢复日期格式
    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')

    # 只保留龙头股的数据
    leader_df = df[df['股票代码'].isin(leader_stock_ids)].copy()

    return leader_df, leader_stock_ids


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    # 用最后约 2 个月做验证，并额外回退 sequence_length 个交易日作为序列上下文
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后约2个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

# 主程序
def main():
    seed = config.get('seed', 42)
    set_seed(seed)
    output_dir = config['output_dir']
    model_dir = os.path.join(output_dir, f'seed_{seed}')
    os.makedirs(model_dir, exist_ok=True)
    # 保存配置文件到 seed 子目录
    data_path = config['data_path']
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    # 统一为 6 位代码，与 get_stock_data / predict 保持一致
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)

    # 1.0 先构建全量股票映射（保持训练/预测时 instrument 索引一致）
    all_stock_ids = sorted(full_df['股票代码'].unique())
    full_stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}

    # 1.1 先划分再筛龙头：龙头池只看训练集近期成交额，避免用验证期信息定池
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
    leader_ratio = config.get('train_leader_ratio', 0.40)
    leader_lookback = config.get('leader_lookback_days', 30)
    train_df, leader_stock_ids = filter_leader_stocks(
        train_df,
        leader_ratio=leader_ratio,
        lookback_days=leader_lookback,
    )
    # 验证集与训练集使用同一股票宇宙，保证截面注意力分布一致
    val_df = val_df[val_df['股票代码'].isin(leader_stock_ids)].copy()
    print(f"验证集对齐龙头池后股票数: {val_df['股票代码'].nunique()}")

    stockid2idx = full_stockid2idx
    num_stocks = len(stockid2idx)

    # 保存映射与龙头池，供预测时复用同一宇宙
    joblib.dump(stockid2idx, os.path.join(model_dir, 'stockid2idx.pkl'))
    joblib.dump([str(s).zfill(6) for s in leader_stock_ids], os.path.join(model_dir, 'leader_stock_ids.pkl'))
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 用训练集中位数填充NaN，保留更多样本
    train_medians = train_data[features].median()
    train_data[features] = train_data[features].fillna(train_medians)
    val_data[features] = val_data[features].fillna(train_medians)
    joblib.dump(train_medians, os.path.join(model_dir, 'train_medians.pkl'))
    # 丢弃残余NaN（如某列全为NaN的情况）
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    
    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = CombinedRankingLoss(
        k=config.get('topk_mle_k', 3),
        portfolio_weight=config.get('portfolio_loss_weight', 0.4),
        temperature=config.get('portfolio_temperature', 0.35),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-3),
    )
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'] - warmup_epochs, eta_min=config['learning_rate'] * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 15)

        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 选模：默认用加权组合绝对收益（与赛方评分同口径）
            selection_metric = config.get('selection_metric', 'weighted_port_return')
            current_score = eval_metrics.get(selection_metric, eval_metrics.get('weighted_port_return', 0.0))
            current_port = eval_metrics.get('weighted_port_return', 0.0)
            current_final = eval_metrics.get('weighted_final_score', 0.0)
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
                print(
                    f"保存最佳模型 - {selection_metric}: {best_score:.4f}, "
                    f"port={current_port:.4f}, final={current_final:.4f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"\nEarly stopping at epoch {epoch+1}, best epoch: {best_epoch}, "
                        f"best {selection_metric}: {best_score:.4f}"
                    )
                    break
        print(
            f"\n训练完成！最佳 epoch: {best_epoch}, "
            f"最佳 {config.get('selection_metric', 'weighted_port_return')}: {best_score:.4f}"
        )
        with open(os.path.join(model_dir, 'final_score.txt'), 'w') as f:
            f.write(
                f"Best epoch: {best_epoch}\n"
                f"Best {config.get('selection_metric', 'weighted_port_return')}: {best_score:.6f}\n"
            )

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    seeds = config.get('ensemble_seeds', [config.get('seed', 42)])
    all_scores = []
    for seed in seeds:
        config['seed'] = seed
        print(f"\n========== 开始训练 seed={seed} ==========")
        score = main()
        all_scores.append((seed, score))
        print(f"========== seed={seed} 完成: {score:.4f} ==========")
    print("\n########## 全部训练完成 ##########")
    for seed, score in all_scores:
        print(f"  seed={seed}: {score:.4f}")