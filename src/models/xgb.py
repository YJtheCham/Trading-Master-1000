import numpy as np
import xgboost as xgb

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost 回归模型，使用滞后特征与滚动统计量"""
    def __init__(self, lookback: int = 20):
        super().__init__("XGBoost")
        self.lookback = lookback
        self._model = None
        self._last_data = None

    def _features_at(self, data: np.ndarray, i: int) -> list:
        feats = []
        for lag in [1, 2, 3, 5, 10, 20]:
            idx = i - lag
            feats.append(data[idx] if idx >= 0 else data[i])
        window = data[max(0, i - 20):i]
        feats.append(float(np.mean(window)))
        feats.append(float(np.std(window)))
        feats.append(float(np.median(window)))
        feats.append(float(data[i - 1] - data[i - min(i, 5)]))
        feats.append(float(data[i - 1] - data[i - min(i, 2)]))
        return feats

    def _build_features(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(self._features_at(data, i))
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray):
        X, y = self._build_features(data)
        self._model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y)
        self._last_data = data.copy()

    def predict(self, steps: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("请先调用 train()")
        results = []
        data = self._last_data.copy()
        for _ in range(steps):
            feats = np.array(self._features_at(data, len(data))).reshape(1, -1)
            pred = self._model.predict(feats)[0]
            results.append(pred)
            data = np.append(data, pred)
        return np.array(results)
