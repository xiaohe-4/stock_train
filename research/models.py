"""Three research arms. The memory arm edits only the second temporal block."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "app" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    CrossStockAttention,
    FeatureAttention,
    PositionalEncoding,
    StockTransformer,
)
from research.protocol import LSTM_HIDDEN, MEMORY_HIDDEN


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class TemporalLSTMMemoryBlock(nn.Module):
    """Post-norm time attention, then a real LSTM cell, then the feed-forward."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, lstm_hidden: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.memory_proj = nn.Linear(lstm_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_mem = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        u = self.norm1(x + attended)
        hidden, _ = self.lstm(u)
        memory = self.memory_proj(hidden)
        y = self.norm_mem(u + self.dropout(memory))
        feed = self.linear2(self.ff_dropout(F.relu(self.linear1(y))))
        return self.norm2(y + feed)


class _RankerBase(nn.Module):
    def __init__(self, input_dim: int, config: dict, num_stocks: int):
        super().__init__()
        d_model = config["d_model"]
        dropout = config["dropout"]
        self.config = config
        self.num_stocks = num_stocks
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, config["sequence_length"])
        self.feature_attention = FeatureAttention(d_model, dropout)
        n_cross = max(1, int(config.get("cross_stock_layers", 1)))
        self.cross_stock_attention = nn.ModuleList(
            [CrossStockAttention(d_model, config["nhead"], dropout) for _ in range(n_cross)]
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
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _project(self, src: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, stocks, _length, _features = src.shape
        flat = src.reshape(batch * stocks, src.size(2), src.size(3))
        projected = self.pos_encoder(self.input_proj(flat))
        return projected, batch, stocks

    def _scores_from_pooled(self, pooled: torch.Tensor, batch: int, stocks: int, mask) -> torch.Tensor:
        stock_features = pooled.view(batch, stocks, -1)
        key_padding_mask = mask.eq(0) if mask is not None else None
        interactive = stock_features
        for layer in self.cross_stock_attention:
            interactive = layer(interactive, key_padding_mask=key_padding_mask)
        ranked = self.ranking_layers(interactive.reshape(batch * stocks, -1))
        return self.score_head(ranked).view(batch, stocks)


class MemoryTransformer(_RankerBase):
    def __init__(self, input_dim: int, config: dict, num_stocks: int):
        super().__init__(input_dim, config, num_stocks)
        d_model = config["d_model"]
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.memory_block = TemporalLSTMMemoryBlock(
            d_model=d_model,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            lstm_hidden=int(config.get("memory_hidden", MEMORY_HIDDEN)),
        )

    def encode_temporal(self, src: torch.Tensor) -> torch.Tensor:
        projected, batch, stocks = self._project(src)
        hidden = self.memory_block(self.temporal_layer(projected))
        pooled = self.feature_attention(hidden)
        return pooled.view(batch, stocks, -1)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        projected, batch, stocks = self._project(src)
        hidden = self.memory_block(self.temporal_layer(projected))
        pooled = self.feature_attention(hidden)
        return self._scores_from_pooled(pooled, batch, stocks, mask)


class LSTMOnlyRanker(_RankerBase):
    def __init__(self, input_dim: int, config: dict, num_stocks: int):
        super().__init__(input_dim, config, num_stocks)
        hidden = int(config.get("lstm_hidden", LSTM_HIDDEN))
        d_model = config["d_model"]
        self.lstm = nn.LSTM(d_model, hidden, num_layers=1, batch_first=True)
        self.lstm_proj = nn.Linear(hidden, d_model)

    def encode_temporal(self, src: torch.Tensor) -> torch.Tensor:
        projected, batch, stocks = self._project(src)
        sequence, _ = self.lstm(projected)
        pooled = self.feature_attention(self.lstm_proj(sequence))
        return pooled.view(batch, stocks, -1)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        projected, batch, stocks = self._project(src)
        sequence, _ = self.lstm(projected)
        pooled = self.feature_attention(self.lstm_proj(sequence))
        return self._scores_from_pooled(pooled, batch, stocks, mask)


def build_model(arm: str, input_dim: int, num_stocks: int, config: dict) -> nn.Module:
    if arm == "transformer":
        return StockTransformer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    if arm == "memory":
        return MemoryTransformer(input_dim, config, num_stocks)
    if arm == "lstm":
        return LSTMOnlyRanker(input_dim, config, num_stocks)
    raise ValueError(f"unknown arm {arm}")


def benchmark_step(model: nn.Module, batch: torch.Tensor) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = torch.randn(batch.shape[0], batch.shape[1], device=batch.device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)
    loss = (prediction - target).pow(2).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if not torch.isfinite(loss.detach()):
        raise RuntimeError("non-finite loss")
    return {"seconds": time.perf_counter() - started, "grad_norm": float(grad_norm)}
