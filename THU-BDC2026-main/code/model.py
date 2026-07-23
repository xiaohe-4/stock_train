import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码，突出时序先后关系。"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossStockAttention(nn.Module):
    """股票间交互注意力：支持 padding mask，避免无效股票污染排序。"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features, key_padding_mask=None):
        # stock_features: [batch, num_stocks, d_model]
        # key_padding_mask: [batch, num_stocks], True 表示需要忽略的位置
        attended, _ = self.cross_attention(
            stock_features,
            stock_features,
            stock_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(stock_features + self.dropout(attended))
        x = self.norm2(x + self.ffn(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class TemporalPooling(nn.Module):
    """
    时序聚合：注意力池化 + 近期偏置。
    周度涨幅更依赖近期边际资金变化，因此对最近交易日给予可学习偏置。
    """

    def __init__(self, d_model, dropout=0.1, recent_bias_strength=1.5):
        super().__init__()
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.recent_bias_strength = recent_bias_strength
        self.dropout = nn.Dropout(dropout)
        # 可学习门控：在全局注意力聚合与“最近时刻”表征之间权衡
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        seq_len = x.size(1)
        scores = self.score_proj(x).squeeze(-1)  # [B*N, L]

        # 线性近期偏置：越靠近窗口末端权重越大
        recent_bias = torch.linspace(
            0.0, self.recent_bias_strength, steps=seq_len, device=x.device, dtype=x.dtype
        )
        weights = torch.softmax(scores + recent_bias.unsqueeze(0), dim=1)  # [B*N, L]
        attended = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B*N, d_model]
        last_state = x[:, -1, :]

        gate = self.gate(torch.cat([attended, last_state], dim=-1))
        pooled = gate * attended + (1.0 - gate) * last_state
        return self.dropout(pooled)


class StockTransformer(nn.Module):
    """
    排序学习选股模型：
    1) Transformer 提取单票时序特征
    2) 近期偏置时序聚合
    3) 多层股票间交互（带 mask）
    4) 门控融合个股表征与横截面交互表征后输出排序分数
    """

    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super().__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        d_model = config['d_model']
        dropout = config['dropout']
        nhead = config['nhead']
        cross_layers = max(1, int(config.get('cross_stock_layers', 1)))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, config['sequence_length'])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=config['dim_feedforward'],
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config['num_layers']
        )

        self.temporal_pooling = TemporalPooling(
            d_model,
            dropout=dropout,
            recent_bias_strength=float(config.get('recent_bias_strength', 1.5)),
        )

        self.cross_stock_layers = nn.ModuleList(
            [CrossStockAttention(d_model, nhead, dropout) for _ in range(cross_layers)]
        )

        # 个股时序特征 vs 横截面交互特征 的门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.ranking_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, stock_mask=None):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        stock_mask: [batch, num_stocks]，1=有效股票，0=padding；可为 None
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        temporal_features = self.temporal_encoder(src_proj)
        stock_features = self.temporal_pooling(temporal_features)
        stock_features = stock_features.view(batch_size, num_stocks, -1)

        key_padding_mask = None
        if stock_mask is not None:
            # MultiheadAttention: True 表示忽略
            key_padding_mask = stock_mask <= 0
            stock_features = stock_features.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        interactive = stock_features
        for layer in self.cross_stock_layers:
            interactive = layer(interactive, key_padding_mask=key_padding_mask)

        gate = self.fusion_gate(torch.cat([stock_features, interactive], dim=-1))
        fused = gate * interactive + (1.0 - gate) * stock_features

        fused = fused.view(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(fused)
        scores = self.score_head(ranking_features).view(batch_size, num_stocks)

        if stock_mask is not None:
            scores = scores.masked_fill(stock_mask <= 0, -1e9)

        return scores
