import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .gbdt import _get_all_cols, _train_cols


class _LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMModel(BaseModel):

    def __init__(self, lookback: int = 60, hidden: int = 64,
                 dropout: float = 0.2, epochs: int = 80, lr: float = 1e-3,
                 batch_size: int = 32):
        super().__init__("LSTM")
        self.lookback = lookback
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._all_cols = None
        self._x_cols = None
        self._close_idx = None
        self._x_to_all = None
        self._scaler = StandardScaler()
        self._net = None
        self._last_rows = None
        self._close_mean = 0.0
        self._close_std = 1.0
        self._tuned_params = None

    def get_param_info(self):
        info = super().get_param_info()
        info["params"] = {
            "lookback": self.lookback,
            "hidden_size": self.hidden,
            "num_layers": (self._net.lstm.num_layers if self._net and hasattr(self._net, 'lstm') else 2),
            "dropout": self.dropout,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        if self._x_cols:
            info["features"] = self._x_cols
        return info

    def train(self, data: pd.DataFrame):
        self._all_cols = _get_all_cols(data)
        self._x_cols = _train_cols(self._all_cols)
        self._close_idx = self._all_cols.index("Close")
        self._x_to_all = [self._all_cols.index(c) for c in self._x_cols]

        X_full = data[self._all_cols].values.astype(np.float32)
        X_full = self._scaler.fit_transform(X_full)
        self._close_mean = float(self._scaler.mean_[self._close_idx])
        self._close_std = float(self._scaler.scale_[self._close_idx])

        seq_x, seq_y = [], []
        for i in range(self.lookback, len(X_full)):
            seq_x.append(X_full[i - self.lookback:i, self._x_to_all])
            seq_y.append(X_full[i, self._close_idx])
        seq_x = torch.tensor(np.array(seq_x), dtype=torch.float32)
        seq_y = torch.tensor(np.array(seq_y), dtype=torch.float32).unsqueeze(-1)

        self._net = _LSTMNet(input_size=len(self._x_cols), hidden_size=self.hidden)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = len(seq_x)

        self._net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:min(i + self.batch_size, n)]
                optimizer.zero_grad()
                loss = loss_fn(self._net(seq_x[idx]), seq_y[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch > 20 and total_loss < 1e-5:
                break

        self._last_rows = X_full.copy()

    @torch.no_grad()
    def predict(self, steps: int) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("请先调用 train()")
        self._net.eval()

        results = []
        full = self._last_rows[-self.lookback:].copy()
        last_close_norm = float(full[-1, self._close_idx])

        for _ in range(steps):
            x = torch.tensor(full[-self.lookback:, self._x_to_all],
                             dtype=torch.float32).unsqueeze(0)
            pred_norm = self._net(x).item()
            pred_price = pred_norm * self._close_std + self._close_mean
            results.append(pred_price)

            new_row = full[-1].copy()
            chg = (pred_norm - last_close_norm) / max(abs(last_close_norm), 0.01)
            new_row[self._close_idx] = pred_norm
            col_map = {c: i for i, c in enumerate(self._all_cols)}
            for c in ["Open", "High", "Low"]:
                if c in col_map:
                    new_row[col_map[c]] = pred_norm
            if "ret_1d" in col_map:
                new_row[col_map["ret_1d"]] = chg
            for k in [3, 5, 10, 20]:
                if f"ret_{k}d" in col_map:
                    new_row[col_map[f"ret_{k}d"]] = chg
            full = np.vstack([full, new_row])
            last_close_norm = pred_norm

        return np.array(results)
