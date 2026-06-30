import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel
from .gbdt import _get_all_cols, _train_cols


class XGBoostModel(BaseModel):

    def __init__(self):
        super().__init__("XGBoost")
        self._model = None
        self._all_cols = None
        self._x_cols = None
        self._close_idx = None
        self._x_to_all = None
        self._last_rows = None
        self._last_close = None
        self._tuned_params = None

    def get_param_info(self):
        info = super().get_param_info()
        if self._model:
            info["params"] = {
                "n_estimators": self._model.n_estimators,
                "learning_rate": round(self._model.learning_rate, 4),
                "max_depth": self._model.max_depth,
                "subsample": round(self._model.subsample, 4),
                "colsample_bytree": round(self._model.colsample_bytree, 4) if self._model.colsample_bytree else 0.8,
            }
        if self._x_cols:
            info["features"] = self._x_cols
        return info

    def train(self, data):
        if isinstance(data, np.ndarray):
            self._train_simple(data)
        else:
            self._train_full(data)

    def _train_simple(self, prices: np.ndarray):
        lookback = 20
        X, y = [], []
        for i in range(lookback, len(prices)):
            feats = []
            for lag in [1, 2, 3, 5, 10, 20]:
                idx = i - lag
                feats.append(prices[idx] if idx >= 0 else prices[i])
            window = prices[max(0, i - 20):i]
            feats.append(float(np.mean(window)) if len(window) > 0 else 0)
            feats.append(float(np.std(window)) if len(window) > 1 else 0)
            feats.append(float(np.median(window)) if len(window) > 0 else 0)
            feats.append(float(prices[i - 1] - prices[i - min(i, 5)]) if i > 0 else 0)
            feats.append(float(prices[i - 1] - prices[i - min(i, 2)]) if i > 1 else 0)
            X.append(feats)
            y.append(prices[i])
        X, y = np.array(X), np.array(y)

        defaults = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 7,
                     "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0}
        params = {**defaults, **(self._tuned_params or {})}
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X, y)
        self._last_rows = X.copy()
        self._last_close = y.copy()
        self._x_to_all = list(range(X.shape[1]))
        self._all_cols = [f"feat_{i}" for i in range(X.shape[1])]

    def _train_full(self, data: pd.DataFrame):
        from .features import engineer_features
        if "Date" in data.columns and "macd" not in data.columns:
            data = engineer_features(data)
        self._all_cols = _get_all_cols(data)
        self._x_cols = _train_cols(self._all_cols)
        self._close_idx = self._all_cols.index("Close")
        self._x_to_all = [self._all_cols.index(c) for c in self._x_cols]

        X = data[self._x_cols].values if self._x_cols else data[["Close"]].values
        y = data["Close"].values
        X, y = X[:-1], y[1:]  # features[t] → predict Close[t+1]

        defaults = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 7,
                     "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0}
        params = {**defaults, **(self._tuned_params or {})}
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X, y)
        self._last_rows = data[self._all_cols].values[:-1].copy()
        self._last_close = data["Close"].values[1:].copy()

    def predict(self, steps: int) -> np.ndarray:
        results = []
        full = self._last_rows[-1].copy()
        last_price = self._last_close[-1]

        for _ in range(steps):
            x = full[self._x_to_all].reshape(1, -1)
            pred = float(self._model.predict(x)[0])
            results.append(pred)
            full = self._roll(full, pred, last_price)
            last_price = pred

        return np.array(results)

    def _roll(self, row: np.ndarray, new_price: float, old_price: float) -> np.ndarray:
        r = row.copy()
        if len(r) <= 12:
            r[:-1] = r[1:]
            r[-1] = new_price
            return r
        ret_1d = (new_price - old_price) / max(abs(old_price), 0.01)
        col_map = {c: i for i, c in enumerate(self._all_cols)}
        for c in ["Open", "High", "Low", "Close"]:
            if c in col_map:
                r[col_map[c]] = new_price
        if "ret_1d" in col_map:
            r[col_map["ret_1d"]] = ret_1d
        for k in [3, 5, 10, 20]:
            if f"ret_{k}d" in col_map:
                prev_key = "ret_2d" if k == 3 else f"ret_{k-1}d"
                if k == 3 and "ret_2d" not in col_map:
                    prev_key = "ret_1d"
                if prev_key in col_map:
                    r[col_map[f"ret_{k}d"]] = (1 + r[col_map[prev_key]]) * (1 + ret_1d) - 1
                else:
                    r[col_map[f"ret_{k}d"]] = ret_1d
        return r
