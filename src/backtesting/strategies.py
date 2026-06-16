from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """策略基类: 每个交易日返回信号 -1(卖出) / 0(持有) / 1(买入)"""

    def __init__(self, name: str = "strategy"):
        self.name = name

    @abstractmethod
    def init(self, df: pd.DataFrame):
        """回测开始前调用，用于计算指标"""

    @abstractmethod
    def next(self, i: int, df: pd.DataFrame) -> int:
        """第 i 天返回信号"""


class SignalArrayStrategy(BaseStrategy):
    """使用预生成的信号数组 (如从预测模型得到)"""
    def __init__(self, signals: np.ndarray, name: str = "signal"):
        super().__init__(name)
        self.signals = signals

    def init(self, df: pd.DataFrame):
        pass

    def next(self, i: int, df: pd.DataFrame) -> int:
        if i >= len(self.signals):
            return 0
        return int(self.signals[i])


class PredictionStrategy(BaseStrategy):
    """一次性预测 (旧版, 仅训练一次) — 建议改用 RollingPredictionStrategy"""
    def __init__(self, model, threshold_buy: float = 0.02,
                 threshold_sell: float = -0.02, forecast_steps: int = 5,
                 name: str = "prediction"):
        super().__init__(name)
        self.model = model
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.forecast_steps = forecast_steps
        self._forecasts: dict[int, float] = {}

    def init(self, df: pd.DataFrame):
        self._forecasts = {}
        train_end = int(len(df) * 0.7)
        if train_end < 60:
            return
        train = df.iloc[:train_end]
        prices = train["Close"].values
        self.model.train(prices)
        forecast = self.model.predict(self.forecast_steps)
        for j in range(self.forecast_steps):
            self._forecasts[train_end + j] = forecast[j]

    def next(self, i: int, df: pd.DataFrame) -> int:
        if i not in self._forecasts:
            return 0
        pred = self._forecasts[i]
        current = df.iloc[i]["Close"]
        change = (pred - current) / current
        if change > self.threshold_buy:
            return 1
        elif change < self.threshold_sell:
            return -1
        return 0


class RollingPredictionStrategy(BaseStrategy):
    """滚动预测策略 — Walk-Forward Optimization

    流程:
      1. warmup 天后进入测试期
      2. 每 retrain_freq 天, 用截至当天的全部数据训练模型
      3. 预测未来 forecast_steps 天, 缓存起来
      4. 每个交易日检查缓存, 用预测涨幅 vs 阈值产生信号
      5. 缓存过期 → 等待下一次重训

    参数:
      model          : 预测模型 (需有 .train()/.predict())
      warmup         : 初始训练天数 (默认120)
      retrain_freq   : 每多少天重训一次 (默认20, 即月频)
      forecast_steps : 每次预测未来多少天 (默认5)
      threshold_buy  : 预测涨幅超过此值 → 买入 (默认1.5%)
      threshold_sell : 预测涨幅低于此值 → 卖出 (默认-1.5%)
    """
    def __init__(self, model, warmup: int = 120, retrain_freq: int = 20,
                 forecast_steps: int = 5, threshold_buy: float = 0.015,
                 threshold_sell: float = -0.015, name: str = "rolling_prediction"):
        super().__init__(name)
        self.model = model
        self.warmup = warmup
        self.retrain_freq = retrain_freq
        self.forecast_steps = forecast_steps
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self._forecasts: dict[int, float] = {}
        self._last_train_idx = 0

    def init(self, df: pd.DataFrame):
        self._forecasts = {}
        self._last_train_idx = 0

    def _retrain_and_forecast(self, df: pd.DataFrame, i: int):
        """用 data[:i] 训练, 预测 [i, i+forecast_steps) 并缓存"""
        prices = df.iloc[:i]["Close"].values
        if len(prices) < self.warmup:
            return
        try:
            self.model.train(prices)
            preds = self.model.predict(self.forecast_steps)
            for j, p in enumerate(preds):
                self._forecasts[i + j] = p
            self._last_train_idx = i
        except Exception:
            self._last_train_idx = i  # 即使失败也推进, 避免死循环

    def next(self, i: int, df: pd.DataFrame) -> int:
        # 没到 warmup → 不交易
        if i < self.warmup:
            return 0

        # 判断需要重训: 首次 / 达到重训频率 / 缓存过期(当天不在缓存中)
        stale = i not in self._forecasts
        due = (i - self._last_train_idx) >= self.retrain_freq
        if stale or due:
            self._retrain_and_forecast(df, i)
            stale = i not in self._forecasts  # 重训后还不在 → 无信号

        if stale:
            return 0

        pred = self._forecasts[i]
        current = df.iloc[i]["Close"]
        change = (pred - current) / current

        if change > self.threshold_buy:
            return 1
        elif change < self.threshold_sell:
            return -1
        return 0


class MovingAverageCrossStrategy(BaseStrategy):
    """双均线交叉策略"""
    def __init__(self, short: int = 20, long: int = 60, name: str = "ma_cross"):
        super().__init__(name)
        self.short = short
        self.long = long
        self._ma_short: Optional[np.ndarray] = None
        self._ma_long: Optional[np.ndarray] = None

    def init(self, df: pd.DataFrame):
        prices = df["Close"].values
        self._ma_short = pd.Series(prices).rolling(self.short).mean().values
        self._ma_long = pd.Series(prices).rolling(self.long).mean().values

    def next(self, i: int, df: pd.DataFrame) -> int:
        if self._ma_short is None or self._ma_long is None:
            return 0
        if i < 1 or np.isnan(self._ma_short[i]) or np.isnan(self._ma_long[i]):
            return 0
        prev_short = self._ma_short[i - 1]
        prev_long = self._ma_long[i - 1]
        curr_short = self._ma_short[i]
        curr_long = self._ma_long[i]
        if prev_short <= prev_long and curr_short > curr_long:
            return 1
        elif prev_short >= prev_long and curr_short < curr_long:
            return -1
        return 0


class RSIStrategy(BaseStrategy):
    """RSI 均值回归策略: 超卖买入, 超买卖出"""
    def __init__(self, window: int = 14, oversold: int = 30,
                 overbought: int = 70, name: str = "rsi"):
        super().__init__(name)
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self._rsi: Optional[np.ndarray] = None

    def init(self, df: pd.DataFrame):
        prices = df["Close"].values
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(self.window).mean().values
        avg_loss = pd.Series(loss).rolling(self.window).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        self._rsi = np.concatenate([[np.nan], rsi])

    def next(self, i: int, df: pd.DataFrame) -> int:
        if self._rsi is None or i < self.window + 1 or np.isnan(self._rsi[i]):
            return 0
        if self._rsi[i] < self.oversold:
            return 1
        elif self._rsi[i] > self.overbought:
            return -1
        return 0


class ChannelBreakoutStrategy(BaseStrategy):
    """通道突破: 价格突破 N 日高点买入, 跌破 M 日低点卖出"""
    def __init__(self, high_lookback: int = 20, low_lookback: int = 10,
                 name: str = "channel"):
        super().__init__(name)
        self.high_lb = high_lookback
        self.low_lb = low_lookback
        self._high_ch: Optional[np.ndarray] = None
        self._low_ch: Optional[np.ndarray] = None

    def init(self, df: pd.DataFrame):
        prices = df["Close"].values
        self._high_ch = pd.Series(prices).rolling(self.high_lb).max().values
        self._low_ch = pd.Series(prices).rolling(self.low_lb).min().values

    def next(self, i: int, df: pd.DataFrame) -> int:
        if self._high_ch is None or i < max(self.high_lb, self.low_lb) + 1:
            return 0
        price = df.iloc[i]["Close"]
        prev = df.iloc[i - 1]["Close"]
        if price > self._high_ch[i - 1] and prev <= self._high_ch[i - 1]:
            return 1
        if price < self._low_ch[i - 1] and prev >= self._low_ch[i - 1]:
            return -1
        return 0


class BollingerStrategy(BaseStrategy):
    """布林带均值回归: 触及下轨买入, 触及上轨卖出"""
    def __init__(self, window: int = 20, std: int = 2,
                 name: str = "bollinger"):
        super().__init__(name)
        self.window = window
        self.std = std
        self._upper: Optional[np.ndarray] = None
        self._lower: Optional[np.ndarray] = None

    def init(self, df: pd.DataFrame):
        prices = df["Close"].values
        ma = pd.Series(prices).rolling(self.window).mean().values
        sd = pd.Series(prices).rolling(self.window).std().values
        self._upper = ma + self.std * sd
        self._lower = ma - self.std * sd

    def next(self, i: int, df: pd.DataFrame) -> int:
        if self._upper is None or i < self.window or np.isnan(self._lower[i]):
            return 0
        price = df.iloc[i]["Close"]
        if price <= self._lower[i]:
            return 1
        if price >= self._upper[i]:
            return -1
        return 0
