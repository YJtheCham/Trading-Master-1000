import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from .base import BaseModel


class _LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMModel(BaseModel):
    """LSTM 时间序列预测 (PyTorch)"""
    def __init__(self, lookback: int = 60, hidden: int = 50,
                 epochs: int = 80, lr: float = 1e-3, batch_size: int = 32):
        super().__init__("LSTM")
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._scaler = MinMaxScaler()
        self._net: _LSTMNet | None = None
        self._last_sequence: np.ndarray | None = None

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return (torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1))

    def train(self, data: np.ndarray):
        scaled = self._scaler.fit_transform(data.reshape(-1, 1)).flatten()
        X, y_t = self._create_sequences(scaled)

        self._net = _LSTMNet(input_size=1, hidden_size=self.hidden)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        n = len(X)
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                end = min(i + self.batch_size, n)
                xb, yb = X[i:end], y_t[i:end]
                optimizer.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # 早停：loss 稳定后提前结束
            if epoch > 10 and total_loss < 1e-6:
                break

        self._last_sequence = scaled[-self.lookback:]

    @torch.no_grad()
    def predict(self, steps: int) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("请先调用 train()")
        self._net.eval()
        results = []
        seq = self._last_sequence.copy()
        for _ in range(steps):
            x = torch.tensor(seq, dtype=torch.float32).view(1, -1, 1)
            pred = self._net(x).item()
            results.append(pred)
            seq = np.append(seq[1:], pred)
        results = np.array(results).reshape(-1, 1)
        return self._scaler.inverse_transform(results).flatten()
