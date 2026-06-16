import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .base import BaseModel


class GBDTModel(BaseModel):
    def __init__(self, lookback: int = 20):
        super().__init__("GBDT")
        self.lookback = lookback
        self._model = None
        self._last_data = None

    def _build_features(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            feats = self._features_at(data, i)
            X.append(feats)
            y.append(data[i])
        return np.array(X), np.array(y)

    def _features_at(self, data: np.ndarray, i: int) -> list:
        feats = []
        for lag in [1, 2, 3, 5, 10, 20]:
            idx = i - lag
            feats.append(data[idx] if idx >= 0 else data[i])
        window = data[max(0, i - 20):i]
        feats.append(float(np.mean(window)))
        feats.append(float(np.std(window)))
        feats.append(float(data[i - 1] - data[i - min(i, 5)]))
        return feats

    def train(self, data: np.ndarray):
        X, y = self._build_features(data)
        self._model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42,
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
