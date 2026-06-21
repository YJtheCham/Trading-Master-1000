import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .gbdt import _get_all_cols, _train_cols


class _TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])


class TransformerModel(BaseModel):

    def __init__(self, lookback: int = 60, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 128,
                 dropout: float = 0.1, epochs: int = 40, lr: float = 1e-3):
        super().__init__("Transformer")
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self._d_model = d_model
        self._nhead = nhead
        self._num_layers = num_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
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
            "d_model": self._d_model,
            "nhead": self._nhead,
            "num_layers": self._num_layers,
            "dim_feedforward": self._dim_feedforward,
            "dropout": self._dropout,
            "learning_rate": self.lr,
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

        effective_lookback = min(self.lookback, len(X_full) - 2)
        if effective_lookback < 10:
            effective_lookback = 10
        if len(X_full) <= effective_lookback:
            raise ValueError(f"Transformer: 数据量不足 (需要>{effective_lookback}行, 实际{len(X_full)}行)")

        seq_x, seq_y = [], []
        for i in range(effective_lookback, len(X_full)):
            seq_x.append(X_full[i - effective_lookback:i, self._x_to_all])
            seq_y.append(X_full[i, self._close_idx])

        if len(seq_x) == 0:
            raise ValueError(f"Transformer: 无法构建训练序列 (lookback={effective_lookback}, 数据长度={len(X_full)})")

        seq_x = torch.tensor(np.array(seq_x), dtype=torch.float32)
        seq_y = torch.tensor(np.array(seq_y), dtype=torch.float32).unsqueeze(-1)

        input_size = len(self._x_cols)
        d_model = self._d_model
        nhead = self._nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead)
            if d_model == 0:
                d_model = nhead

        self._net = _TimeSeriesTransformer(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=self._num_layers,
        )
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._net(seq_x), seq_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            optimizer.step()

        self._last_rows = X_full.copy()
        self._effective_lookback = effective_lookback

    @torch.no_grad()
    def predict(self, steps: int) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("请先调用 train()")
        self._net.eval()

        lookback = getattr(self, '_effective_lookback', self.lookback)

        results = []
        full = self._last_rows[-lookback:].copy()
        last_close_norm = float(full[-1, self._close_idx])

        for _ in range(steps):
            x = torch.tensor(full[-lookback:, self._x_to_all],
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
                    prev_key = "ret_2d" if k == 3 else f"ret_{k-1}d"
                    if k == 3 and "ret_2d" not in col_map:
                        prev_key = "ret_1d"
                    if prev_key in col_map:
                        new_row[col_map[f"ret_{k}d"]] = (1 + new_row[col_map[prev_key]]) * (1 + chg) - 1
                    else:
                        new_row[col_map[f"ret_{k}d"]] = chg
            full = np.vstack([full, new_row])
            last_close_norm = pred_norm

        return np.array(results)
