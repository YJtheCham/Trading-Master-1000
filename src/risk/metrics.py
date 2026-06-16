import numpy as np
import pandas as pd


def var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk 历史模拟法"""
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)"""
    threshold = np.percentile(returns, (1 - confidence) * 100)
    return float(returns[returns <= threshold].mean())


def sharpe_ratio(returns: np.ndarray, rf: float = 0.02, periods: int = 252) -> float:
    """夏普比率"""
    excess = returns.mean() * periods - rf
    vol = returns.std() * np.sqrt(periods)
    return float(excess / vol) if vol != 0 else 0.0


def max_drawdown(prices: np.ndarray) -> float:
    """最大回撤"""
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    return float(np.min(dd))


def calc_all_risk_metrics(df: pd.DataFrame, price_col: str = "Close") -> dict:
    prices = df[price_col].values
    returns = np.diff(prices) / prices[:-1]

    return {
        "VaR(95%)": round(var(returns), 4),
        "CVaR(95%)": round(cvar(returns), 4),
        "SharpeRatio": round(sharpe_ratio(returns), 4),
        "MaxDrawdown": round(max_drawdown(prices), 4),
        "Volatility": round(float(returns.std() * np.sqrt(252)), 4),
    }
