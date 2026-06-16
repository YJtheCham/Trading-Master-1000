from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class PredictionResult:
    def __init__(self, model_name: str, forecast: np.ndarray, history: np.ndarray,
                 dates: list, forecast_dates: list, metrics: dict):
        self.model_name = model_name
        self.forecast = forecast
        self.history = history
        self.dates = dates
        self.forecast_dates = forecast_dates
        self.metrics = metrics


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, data: np.ndarray):
        ...

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        ...

    def run(self, df: pd.DataFrame, price_col: str = "Close", steps: int = 30,
            test_ratio: float = 0.2) -> PredictionResult:
        prices = df[price_col].values
        train_size = int(len(prices) * (1 - test_ratio))
        train_data = prices[:train_size]
        test_data = prices[train_size:]

        # 1) 用 80% 数据训练, 评估模型准确性
        self.train(train_data)
        test_forecast = self.predict(len(test_data))
        metrics = self._calc_metrics(test_data, test_forecast)

        # 2) 用全部数据重新训练, 预测真正的未来
        self.train(prices)
        forecast = self.predict(steps)

        dates = df["Date"].tolist()
        last_date = pd.Timestamp(dates[-1])
        forecast_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(steps)]

        return PredictionResult(
            model_name=self.name,
            forecast=forecast,
            history=prices,
            dates=dates,
            forecast_dates=forecast_dates,
            metrics=metrics,
        )

    def _calc_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}
