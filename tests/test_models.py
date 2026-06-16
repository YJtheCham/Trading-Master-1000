import numpy as np
import pandas as pd

from src.models.arima import ArimaModel
from src.models.gbdt import GBDTModel
from src.models.xgb import XGBoostModel

from src.risk.metrics import calc_all_risk_metrics, var, cvar, sharpe_ratio, max_drawdown


def _make_sample_df(n=200):
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Date": dates, "Close": prices, "Volume": np.random.randint(1e6, 1e8, n)})


def test_arima_predict():
    df = _make_sample_df(150)
    model = ArimaModel(order=(3, 1, 2))
    result = model.run(df, steps=10)
    assert len(result.forecast) == 10
    assert result.model_name == "ARIMA"
    assert "MAE" in result.metrics


def test_gbdt_predict():
    df = _make_sample_df(150)
    model = GBDTModel(lookback=10)
    result = model.run(df, steps=10)
    assert len(result.forecast) == 10
    assert result.model_name == "GBDT"


def test_xgboost_predict():
    df = _make_sample_df(150)
    model = XGBoostModel(lookback=10)
    result = model.run(df, steps=10)
    assert len(result.forecast) == 10
    assert result.model_name == "XGBoost"


def test_transformer_predict():
    from src.models.transformer_model import TransformerModel
    df = _make_sample_df(80)
    model = TransformerModel(lookback=20, epochs=10)
    result = model.run(df, steps=5)
    assert len(result.forecast) == 5
    assert result.model_name == "Transformer"


def test_risk_metrics():
    df = _make_sample_df(200)
    risk = calc_all_risk_metrics(df)
    assert "VaR(95%)" in risk
    assert "SharpeRatio" in risk
    assert "MaxDrawdown" in risk
    assert risk["MaxDrawdown"] <= 0

    prices = df["Close"].values
    returns = np.diff(prices) / prices[:-1]
    assert var(returns) < 0
    assert cvar(returns) < 0
    assert isinstance(sharpe_ratio(returns), float)
    assert max_drawdown(prices) <= 0
