"""
自动扫描引擎: 全模型预测 + 全策略回测 → 结构化报告
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    model: str
    forecast: np.ndarray = field(default_factory=lambda: np.array([]))
    direction: str = ""
    final_price: float = 0.0
    pct_change: float = 0.0
    mape: float = 0.0
    error: str = ""
    data_source: str = ""
    model_params: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy: str
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    error: str = ""
    strategy_params: dict = field(default_factory=dict)


def scan_predictions(df: pd.DataFrame, steps: int = 30, data_source: str = "",
                     light: bool = False) -> list[ModelResult]:
    """运行预测模型, light=True 跳过 LSTM/Transformer (省内存)"""
    from src.models.factory import run_models
    if light:
        model_names = ["arima", "gbdt", "xgboost"]
    else:
        model_names = None  # all models
    results = run_models(df, steps=steps, data_source=data_source, model_names=model_names)
    output = []
    for name, r in results.items():
        mr = ModelResult(model=name)
        if "error" in r.metrics:
            mr.error = str(r.metrics["error"])
        elif len(r.forecast) > 0:
            mr.forecast = r.forecast
            mr.final_price = float(r.forecast[-1])
            mr.pct_change = (r.forecast[-1] - r.history[-1]) / r.history[-1] * 100
            mr.direction = "📈 看涨" if mr.pct_change > 0 else "📉 看跌"
            mr.mape = float(r.metrics.get("MAPE", 0))
            mr.data_source = r.data_source
            mr.model_params = r.model_params if isinstance(r.model_params, dict) else {}
            mr.feature_names = list(r.feature_names) if r.feature_names else []
        output.append(mr)
    return output


def scan_strategies(df: pd.DataFrame, symbol: str, market: str, capital: float,
                    light: bool = False) -> list[StrategyResult]:
    """运行全部回测策略, 返回结果列表. light=True 跳过滚动预测(省内存)"""
    from src.backtesting.engine import BacktestEngine
    from src.backtesting.models import BacktestConfig
    from src.backtesting.strategies import (
        MovingAverageCrossStrategy, RSIStrategy,
        ChannelBreakoutStrategy, BollingerStrategy,
        RollingPredictionStrategy,
    )
    from src.models.gbdt import GBDTModel
    import gc

    cfg = BacktestConfig(initial_capital=capital, market=market)
    strategies = [
        ("双均线(5/20)",  MovingAverageCrossStrategy(5, 20)),
        ("双均线(10/30)", MovingAverageCrossStrategy(10, 30)),
        ("双均线(20/60)", MovingAverageCrossStrategy(20, 60)),
        ("RSI(14)",       RSIStrategy(14, 30, 70)),
        ("通道突破(20/10)", ChannelBreakoutStrategy(20, 10)),
        ("布林带(20/2)",   BollingerStrategy(20, 2)),
    ]
    if not light:
        strategies.extend([
            ("滚动预测(月频)", RollingPredictionStrategy(
                GBDTModel(), warmup=200, retrain_freq=20,
                threshold_buy=0.015, threshold_sell=-0.015)),
            ("滚动预测(周频)", RollingPredictionStrategy(
                GBDTModel(), warmup=200, retrain_freq=5,
                threshold_buy=0.01, threshold_sell=-0.01)),
        ])

    output = []
    for name, strat in strategies:
        sr = StrategyResult(strategy=name)
        # 捕获策略参数
        try:
            strat_params = {k: v for k, v in strat.__dict__.items() 
                           if not k.startswith('_') and k != 'name'}
            sr.strategy_params = strat_params
        except Exception:
            pass
        try:
            engine = BacktestEngine(df, strat, cfg)
            result = engine.run()
            sr.total_return = result.total_return
            sr.annual_return = result.annual_return
            sr.sharpe = result.sharpe_ratio
            sr.max_dd = result.max_drawdown
            sr.win_rate = result.win_rate
            sr.profit_factor = result.profit_factor
            sr.total_trades = result.total_trades
        except Exception as e:
            sr.error = str(e)[:80]
        output.append(sr)
    return output


def generate_report(stock_name: str, symbol: str, market: str,
                    models: list[ModelResult],
                    strategies: list[StrategyResult],
                    current_price: float,
                    risk: dict,
                    forecast_steps: int = 30) -> str:
    """生成结构化分析报告 (纯数据, 供 LLM 消费)"""
    lines = [f"## {stock_name} ({market}:{symbol}) 综合扫描报告"]
    lines.append(f"当前价格: {current_price:.2f}")
    lines.append(f"扫描时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 风控
    lines.append("### 风控指标")
    for k, v in risk.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # 预测
    lines.append(f"### 多模型预测结果 (未来{forecast_steps}天)")
    for m in models:
        if m.error:
            lines.append(f"- {m.model}: ❌ {m.error}")
        else:
            lines.append(f"- {m.model}: {m.direction} → 预测末价{m.final_price:.2f} "
                         f"({m.pct_change:+.1f}%) MAPE={m.mape:.1f}%")
    valid_m = [m for m in models if not m.error]
    if valid_m:
        up = sum(1 for m in valid_m if m.pct_change > 0)
        lines.append(f"  模型共识: {up}/{len(valid_m)} 看涨")
    lines.append("")

    # 回测
    lines.append("### 策略回测对比")
    best_strat = None
    for s in strategies:
        if s.error:
            lines.append(f"- {s.strategy}: ❌ {s.error}")
        else:
            lines.append(f"- {s.strategy}: 总收益{s.total_return*100:+.1f}% "
                         f"夏普{s.sharpe:.2f} 回撤{s.max_dd*100:.1f}% "
                         f"胜率{s.win_rate*100:.0f}% 交易{s.total_trades}次")
            if not best_strat or s.total_return > best_strat.total_return:
                best_strat = s
    if best_strat:
        lines.append(f"  最佳策略: {best_strat.strategy} "
                     f"(收益{best_strat.total_return*100:+.1f}%)")
    lines.append("")

    return "\n".join(lines)
