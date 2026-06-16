import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from .base import BaseModel


class _TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        x = self.output_head(x[:, -1, :])
        return x


class TransformerModel(BaseModel):
    """Transformer 时间序列预测模型 (PyTorch)"""
    def __init__(self, lookback: int = 60, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, epochs: int = 40, lr: float = 1e-3):
        super().__init__("Transformer")
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self._scaler = MinMaxScaler()
        self._model: _TimeSeriesTransformer | None = None
        self._last_sequence: np.ndarray | None = None

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray):
        scaled = self._scaler.fit_transform(data.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(scaled)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        self._model = _TimeSeriesTransformer(
            d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
        )
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self._model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()

        self._last_sequence = scaled[-self.lookback:]

    @torch.no_grad()
    def predict(self, steps: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("请先调用 train()")
        self._model.eval()
        results = []
        seq = self._last_sequence.copy()
        for _ in range(steps):
            x = torch.tensor(seq, dtype=torch.float32).view(1, -1, 1)
            pred = self._model(x).item()
            results.append(pred)
            seq = np.append(seq[1:], pred)
        results = np.array(results).reshape(-1, 1)
        return self._scaler.inverse_transform(results).flatten()
