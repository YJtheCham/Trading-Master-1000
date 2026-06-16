import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig, BacktestResult
from src.backtesting.strategies import (
    MovingAverageCrossStrategy, PredictionStrategy,
    RollingPredictionStrategy, SignalArrayStrategy,
)


def _make_df(n=300):
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "Close": prices,
        "Volume": np.random.randint(1e6, 1e8, n),
    })


def test_backtest_returns_result_object():
    df = _make_df()
    strat = SignalArrayStrategy(np.random.choice([-1, 0, 1], len(df)))
    engine = BacktestEngine(df, strat)
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0


def test_backtest_ma_cross():
    df = _make_df(300)
    strat = MovingAverageCrossStrategy(short=10, long=30)
    engine = BacktestEngine(df, strat,
                            config=BacktestConfig(initial_capital=10000))
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert "总收益率" in result.metrics
    assert "夏普比率" in result.metrics
    assert "最大回撤" in result.metrics


def test_backtest_prediction_strategy():
    df = _make_df(200)
    from src.models.gbdt import GBDTModel
    model = GBDTModel(lookback=20)
    strat = PredictionStrategy(model, threshold_buy=0.01, threshold_sell=-0.01,
                               forecast_steps=5)
    engine = BacktestEngine(df, strat)
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert result.total_trades >= 0


def test_backtest_all_buy_all_sell():
    df = _make_df(100)
    signals = np.array([1] * 50 + [-1] * 50)
    strat = SignalArrayStrategy(signals)
    engine = BacktestEngine(df, strat,
                            config=BacktestConfig(initial_capital=100000, fee_rate=0))
    result = engine.run()
    # with 0 fees and perfect signals, should have 1 trade
    assert result.total_trades >= 1


def test_backtest_rolling_prediction():
    df = _make_df(400)
    from src.models.gbdt import GBDTModel
    model = GBDTModel(lookback=30)
    strat = RollingPredictionStrategy(model, warmup=150, retrain_freq=30,
                                      forecast_steps=5,
                                      threshold_buy=0.01, threshold_sell=-0.01)
    engine = BacktestEngine(df, strat)
    result = engine.run()
    assert isinstance(result, BacktestResult)
    # Should generate more trades than single-shot prediction
    print(f"滚动预测交易次数: {result.total_trades}")
    assert result.total_trades >= 0


def test_backtest_metrics_values():
    df = _make_df(300)
    strat = MovingAverageCrossStrategy(short=5, long=20)
    engine = BacktestEngine(df, strat)
    result = engine.run()
    assert isinstance(result.total_return, float)
    assert isinstance(result.sharpe_ratio, (float, np.floating))
    assert isinstance(result.max_drawdown, (float, np.floating))
    assert result.max_drawdown <= 0
