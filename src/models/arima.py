import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

from .base import BaseModel


class ArimaModel(BaseModel):
    def __init__(self, order=(5, 1, 0)):
        super().__init__("ARIMA")
        self.order = order
        self.model = None
        self._fitted = None

    def train(self, data):
        if isinstance(data, pd.DataFrame):
            data = data["Close"].values
        self.model = ARIMAModel(data, order=self.order)
        self._fitted = self.model.fit()

    def predict(self, steps: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("请先调用 train()")
        result = self._fitted.forecast(steps)
        return np.asarray(result)
